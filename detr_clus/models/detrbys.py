36# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from torchvision import transforms

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from sklearn.cluster import KMeans
from info_nce import InfoNCE, info_nce
from collections import Counter
from  scipy.spatial.distance import cdist

device = torch.device('cuda:0')


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0] #tensor（6，2，100，256）
        feature = hs[-1]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} #只拿decoder最后一层的输出
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out, feature

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

class SetCriterion0(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)  #拿到了需要的框

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, tem, k1, k2):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.tem = tem
        self.k1 = k1
        self.k2 = k2
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_clus(self, outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all):
        if len(indices_all) == 0:
            losses = {}
            losses['loss_clus'] = 0.0
        else:
            idx = self._get_src_permutation_idx(indices)
            # idx5 = self._get_src_permutation_idx(indices_all)
            # idx5_list = list(idx5)
            # for i in range(idx5_list[0].shape[0]):
            #     if idx5_list[0][i] % 2 == 0:#属于batch0
            #         idx5_list[0][i] = 0
            #     else:
            #         idx5_list[0][i] = 1

            batch = idx[0]

            # print("num_boxes:", num_boxes)
            # print("batch:", batch.shape[0])
            # f_num = batch.shape[0]
            # first_num = 0
            # first = []
            # second = []
            # for i in range(f_num):
            #     if idx[0][i] == 0:
            #         first.append(idx[1][i].tolist())
            #     else:
            #         second.append(idx[1][i].tolist())

            # first5 = []
            # second5 = []
            # num5 = idx5_list[0].shape[0]
            # for i in range(num5):
            #     if idx5_list[0][i] == 0:
            #         first5.append(idx5[1][i].tolist())
            #     else:
            #         second5.append(idx5[1][i].tolist())
            # first5 = sorted(first5)
            # first5_set = set(first5)
            # first5 = list(first5_set)
            # second5 = sorted(second5)
            # second5_set = set(second5)
            # second5 = list(second5_set)
            first5 = indices_all[0][1]
            first_num = len(first5)
            first5 = sorted(first5)
            second5 = indices_all[1][1]
            second_num = len(second5)
            second5 = sorted(second5)
            first5_tensor = torch.tensor(first5)
            second5_tensor = torch.tensor(second5)
            batch_tensor = torch.cat((first5_tensor,second5_tensor))
            zeros_tensor1 = torch.zeros(first_num)
            # zeros_tensor1 = zeros_tensor1.to(torch.int)
            ones_tensor1 = torch.ones(second_num)
            # ones_tensor1 = ones_tensor1.to(torch.int)
            z_batch1 = torch.cat((zeros_tensor1, ones_tensor1))
            z_batch1 = z_batch1.to(torch.long)
            batch_tensor = batch_tensor.to(torch.long)
            idx5_tuple = (z_batch1, batch_tensor)
            f_num = first_num + second_num
            # print("f_num:", f_num)
            


            #
            #
            first5_neg = indices_all[0][0]
            first_num_neg = len(first5_neg)
            first5_neg = sorted(first5_neg)
            second5_neg = indices_all[1][0]
            second_num_neg = len(second5_neg)
            second5_neg = sorted(second5_neg)
            first5_tensor_neg = torch.tensor(first5_neg)
            second5_tensor_neg = torch.tensor(second5_neg)
            batch_tensor_neg = torch.cat((first5_tensor_neg, second5_tensor_neg))
            zeros_tensor1_neg = torch.zeros(first_num_neg)
            # zeros_tensor1 = zeros_tensor1.to(torch.int)
            ones_tensor1_neg = torch.ones(second_num_neg)
            # ones_tensor1 = ones_tensor1.to(torch.int)
            z_batch1_neg = torch.cat((zeros_tensor1_neg, ones_tensor1_neg))
            z_batch1_neg = z_batch1_neg.to(torch.long)
            batch_tensor_neg = batch_tensor_neg.to(torch.long)
            idx5_tuple_neg = (z_batch1_neg, batch_tensor_neg)

            f_num_neg = first_num_neg + second_num_neg
            # print("f_num_neg", f_num_neg)
            

            src_boxes_neg = outputs['pred_boxes'][idx5_tuple_neg]
            src_boxes_xy_neg = box_ops.box_cxcywh_to_xyxy(src_boxes_neg)
            feature_expand = feature.unsqueeze(dim=1)
            img_w = feature.shape[1]
            img_h = feature.shape[2]
            scale = [img_w, img_h, img_w, img_h]
            # src_feature = feature[idx]

            # src_box_first = src_boxes[0:first_num:1,]#获取前first_num个框
            # src_box_sec = src_boxes[first_num:int(num_boxes):1,]
            # # src_feature_first = src_feature[0:first_num:1,]
            # # src_feature_sec = src_feature[first_num:int(num_boxes):1,]
            #
            # batch_box = []
            # batch_box.append(src_box_first)
            # batch_box.append(src_box_sec)
            # feature_num = feature.shape[0]
            # feature_expand = feature.unsqueeze(dim=1)

            l_box_neg = []
            if f_num_neg * f_num > 0:
                for i in range(f_num_neg):
                    box = []
                    box.append(z_batch1_neg[i].item())
                    for j in range(4):
                        box.append(src_boxes_xy_neg[i][j].item() * scale[j])
                    l_box_neg.append(box)

                l_box_tensor_neg = torch.tensor(l_box_neg)
                # a = l_box_tensor.size(1)

                # try_box = torch.tensor([[0.0, 0.0, 34.8, 34.6, 81.4]])
                # f_box = torchvision.ops.roi_align(input=feature_expand, boxes=try_box.to(device),
                #                                   output_size=(7, 7))
                f_box_neg = torchvision.ops.roi_align(input=feature_expand, boxes=l_box_tensor_neg.to(device),
                                                  output_size=(7, 7))
                f_box_de_neg = f_box_neg.squeeze(dim=1)
                num_boxes = f_num_neg
                neg_num = f_num_neg
                f_neg = f_box_de_neg.reshape(num_boxes, -1)

            #
            # num_allbox = outputs['pred_boxes'].shape[1]
            #
            # zero_num = num_allbox - len(first5)
            # one_num =  num_allbox - len(second5)
            # zeros_tensor = torch.zeros(zero_num)
            # ones_tensor = torch.ones(one_num)
            # neg_batch = torch.cat((zeros_tensor,ones_tensor))
            # neg_boxes = []
            # for j in range(num_allbox):
            #     if j not in first5:
            #         neg_boxes.append(outputs['pred_boxes'][0][j].detach().cpu())
            # for j in range(num_allbox):
            #     if j not in second5:
            #         neg_boxes.append(outputs['pred_boxes'][1][j].detach().cpu())
            #
            # # neg_boxes_array = np.array(neg_boxes, dtype='float32')
            # neg_boxes_tensor = torch.stack(neg_boxes)
            # neg_num = neg_boxes_tensor.shape[0]
            # neg_boxes_xy = box_ops.box_cxcywh_to_xyxy(neg_boxes_tensor)
            # feature_expand = feature.unsqueeze(dim=1)
            # img_w = feature.shape[1]
            # img_h = feature.shape[2]
            # scale = [img_w, img_h, img_w, img_h]
            # l_box_neg = []
            # if neg_num > 0:
            #     for i in range(neg_num):
            #         box = []
            #         box.append(neg_batch[i].item())
            #         for j in range(4):
            #             box.append(neg_boxes_xy[i][j].item() * scale[j])
            #         l_box_neg.append(box)
            #
            #
            #     l_box_neg_tensor = torch.tensor(l_box_neg)
            #     f_box_neg = torchvision.ops.roi_align(input=feature_expand, boxes=l_box_neg_tensor.to(device),
            #                                       output_size=(7, 7))
            #     f_box_de_neg = f_box_neg.squeeze(dim=1)
            #     # num_boxes = f_num
            #     f_neg = f_box_de_neg.reshape(neg_num, -1)
            # for i in range(num_allbox):
            #     if

            #
            #
                src_boxes = outputs['pred_boxes'][idx5_tuple]
                src_boxes_xy = box_ops.box_cxcywh_to_xyxy(src_boxes)
                feature_expand = feature.unsqueeze(dim=1)
                img_w = feature.shape[1]
                img_h = feature.shape[2]
                scale = [img_w, img_h, img_w, img_h]
            # src_feature = feature[idx]

            # src_box_first = src_boxes[0:first_num:1,]#获取前first_num个框
            # src_box_sec = src_boxes[first_num:int(num_boxes):1,]
            # # src_feature_first = src_feature[0:first_num:1,]
            # # src_feature_sec = src_feature[first_num:int(num_boxes):1,]
            #
            # batch_box = []
            # batch_box.append(src_box_first)
            # batch_box.append(src_box_sec)
            # feature_num = feature.shape[0]
            # feature_expand = feature.unsqueeze(dim=1)


                l_box = []
            
                for i in range(f_num):
                    box = []
                    box.append(z_batch1[i].item())
                    for j in range(4):
                        box.append(src_boxes_xy[i][j].item() * scale[j])
                    l_box.append(box)

                l_box_tensor = torch.tensor(l_box)
                # a = l_box_tensor.size(1)

                # try_box = torch.tensor([[0.0, 0.0, 34.8, 34.6, 81.4]])
                # f_box = torchvision.ops.roi_align(input=feature_expand, boxes=try_box.to(device),
                #                                   output_size=(7, 7))
                f_box = torchvision.ops.roi_align(input=feature_expand, boxes=l_box_tensor.to(device),
                                                  output_size=(7, 7))
                f_box_de = f_box.squeeze(dim=1)
                num_boxes = f_num
                f = f_box_de.reshape(num_boxes, -1)

                # print()
                idx_stand = self._get_src_permutation_idx(indices_stand)
                src_boxes_stand = outputs_stand['pred_boxes'][idx_stand]
                src_boxes_stand_xy = box_ops.box_cxcywh_to_xyxy(src_boxes_stand)
                l_box_stand = []
                feature_expand_stand = feature_stand.unsqueeze(dim=1)
                s_num = src_boxes_stand.shape[0]
                for i in range(src_boxes_stand.shape[0]):
                    box_stand = []
                    box_stand.append(i)
                    for j in range(4):
                        box_stand.append(src_boxes_stand_xy[i][j].item() * scale[j])
                    l_box_stand.append(box_stand)
                l_box_stand_tensor = torch.tensor(l_box_stand)
                f_box_stand = torchvision.ops.roi_align(input=feature_expand_stand, boxes=l_box_stand_tensor.to(device),
                                                        output_size=(7, 7))
                f_box_stand_de = f_box_stand.squeeze(dim=1)
                f_stand = f_box_stand_de.reshape(src_boxes_stand.shape[0], -1)

                #
                p1f = []
                p0f = []
                psample = []
                pzheng = []
                pfu = []
                sf = []
                # p0_tensor = []
                for i in range(f_num):
                    psample.append(f[i])
                    p1f.append(f[i].tolist())
                for i in range(neg_num):
                    # p0_tensor.append(f_neg[i])
                    p0f.append(f_neg[i].tolist())
                # for i in range(pred_num):
                #     # pf.append(pred_features[i].tolist())
                #     # tpf.append(pred_features[i].tolist())
                #     if i in proposal_id_list:
                #         psample.append(pred_features[i])
                #         p1f.append(pred_features[i].tolist())
                #         tpb.append(0)
                #     else:
                #         p0f.append(pred_features[i].tolist())
                #
                #         tpb.append(-1)
                p1f_array = np.array(p1f, dtype='float64')
                p0f_array = np.array(p0f, dtype='float64')

                for i in range(s_num):
                    sf.append(f_stand[i].tolist())
                    # tpf.append(pred_features_stand[i].tolist())
                    # tpb.append(1)
                sf_array = np.array(sf, dtype='float64')

                # tpf_array = np.array(tpf, dtype='float16')

                # tsne = TSNE(n_components=2, random_state=4)
                # reduced_target = tsne.fit_transform(tpf_array)
                # plt.scatter(reduced_target[:, 0], reduced_target[:, 1], c=cluser_labels, cmap='rainbow')
                # plt.title('t-SNE Visualization')

                # num_PCA = 2
                # pca = PCA(n_components=num_PCA)
                #
                # # 对数据进行降维转换
                # reduced_target = pca.fit_transform(sf_array)
                #
                # reduced_pred = pca.fit_transform(pf_array)
                # rtx = []
                # rty = []
                # for i in range(standard_num):
                #     rtx.append(reduced_target[i][0])
                #     rty.append(reduced_target[i][1])
                #
                # rpx = []
                # rpy = []
                # for i in range(pred_num):
                #     rpx.append(reduced_pred[i][0])
                #     rpy.append(reduced_pred[i][1])
                #
                # plt.scatter(rtx, rty, c='c', edgecolors='r')
                # plt.show()
                # plt.plot(rtx, rty, 'o', rpx, rpy, 'o');
                # plt.show()
                # 输出降维后的结果
                # print(reduced_data)
                # k = tf_array.shape[0]  # target有几个框
                # k1 = sf_array.shape[0]
                # # d = pf_array.shape[1]#特征维度
                # tpf_num = tpf_array.shape[0]

                # kmeans = KMeans(n_clusters=k, init=tf_array)
                # 对于每个样本（若簇中有标准字库）找到与他同类且最近的标准字库样本

                #
                p1_num = p1f_array.shape[0]
                p0_num = p0f_array.shape[0]
                # k1 = p0_num-100
                k1 = self.k1
                # k2 = 8
                kmeans = KMeans(n_clusters=k1)
                    # mbk = MiniBatchKMeans(
                    #     n_clusters=k1,  # 要形成的簇的数量
                    #     init='k-means++',  # 初始化质心的方法：'k-means++' 或 'random'
                    #     max_iter=100,  # 单次运行中执行的最大迭代次数
                    #     batch_size=100,  # 每批样本数量
                    #     verbose=0,  # 是否详细输出日志信息；0为不输出
                    #     compute_labels=False,  # 是否计算标签
                    #     random_state=None,  # 随机种子，用于初始化质心
                    #     tol=0.0,  # 相对于惯性的容忍度，用于声明收敛
                    #     max_no_improvement=10,  # 如果连续这么多批都没有改善聚类效果，则提前终止算法
                    #     init_size=None,  # 初始化时使用的样本数；如果为None，则默认为n_clusters * 3
                    #     n_init=3,  # 使用不同质心种子运行算法的次数
                    #     reassignment_ratio=0.01  # 控制重新分配的阈值
                    # )
                    # mbk.fit(p0f_array)

                kmeans.fit(p0f_array)
                #
                # # cluster_centers = mbk.cluster_centers_
                cluster_centers = kmeans.cluster_centers_
                # # cluster_centers_tensor = torch.from_numpy(cluster_centers)
                # # cluster_centers_tensor.to(device)
                #
                cluster_centers1 = cluster_centers.astype(np.float32)
                cluster_centers_tensor1 = torch.from_numpy(cluster_centers1)
                # k1 = 10
                # initial_medoids1 = kmeans_plusplus_initializer(p0f_array, k1).initialize()
                # initial_medoids1 = [np.where(np.all(p0f_array == center, axis=1))[0][0] for center in initial_medoids1]
                # kmedoids_instance1 = kmedoids(p0f_array, initial_medoids1)  # 假设我们想要将数据分为2个聚类
                #
                #
                # # initial_medoids1 = [int(index) for index in initial_medoids1]
                # # 执行聚类
                # kmedoids_instance1.process()
                # clusters1 = kmedoids_instance1.get_clusters()
                # medoids1 = kmedoids_instance1.get_medoids()

                

                #
                # buqifu = p0f_array[:92]
                # fu_all_array = np.concatenate((buqifu, cluster_centers1), axis=0)
                #
                # fu_all_tensor = torch.from_numpy(fu_all_array)
                #

                #
                # p0_num = p0f_array.shape[0]
                # for k in range(p0_num):
                #     if k in medoids1:
                #         pfu.append(p0f_array[k])

                # pfucentors = np.array(pfu, dtype='float32')
                # pfucentors_tensor = torch.from_numpy(pfucentors)
                #
                sf_num = sf_array.shape[0]
                # k2 = sf_num - 10
                # k2 = 1
                k2 = self.k2

                # initial_medoids = kmeans_plusplus_initializer(sf_array, k2).initialize()
                # initial_medoids = [np.where(np.all(sf_array == center, axis=1))[0][0] for center in initial_medoids]
                #
                # kmedoids_instance = kmedoids(sf_array, initial_medoids)  # 假设我们想要将数据分为2个聚类
                #
                # # 执行聚类
                #
                # kmedoids_instance.process()
                # clusters = kmedoids_instance.get_clusters()
                # medoids = kmedoids_instance.get_medoids()

                kmeans = KMeans(n_clusters=k2)

                kmeans.fit(sf_array)
                cluster_centers = kmeans.cluster_centers_
                cluster_centers = np.mean(cluster_centers, axis=0, keepdims=True)

                #
                # sf_num = sf_array.shape[0]
                for k in range(p1_num):
                    pzheng.append(cluster_centers[0])
                pzhengcentors = np.array(pzheng, dtype='float32')
                # pzhengcentors_tensor = torch.stack(pzhengcentors, dim=0)
                pzhengcentors_tensor = torch.from_numpy(pzhengcentors)

                #
                # distances = np.linalg.norm(p1f_array[:, np.newaxis, :] - pzhengcentors[np.newaxis, :, :], axis=-1)
                #
                # # 找到每个样本与中心点之间距离最小的中心点索引
                # closest_centroid_indices = np.argmin(distances, axis=1)
                # c_num = closest_centroid_indices.shape[0]
                # pz = []
                # for k in range(c_num):
                #     pz.append(pzhengcentors[closest_centroid_indices[k]])
                # pz_array = np.array(pz, dtype='float32')
                # pz_array_tensor = torch.from_numpy(pz_array)#最终对应每个sample的正样本

                #
                psample_tensor = torch.stack(psample, dim=0)
                #
                # loss = InfoNCE(temperature=0.005, negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                loss = InfoNCE(temperature=self.tem, negative_mode='unpaired')
                # batch_size, num_negative, embedding_size = pred_num, k, d
                query = psample_tensor
                # query = torch.randn(batch_size, embedding_size)
                # positive_key = pz_array_tensor
                positive_key = pzhengcentors_tensor
                #negative_keys = f_neg
                negative_keys = cluster_centers_tensor1
                # negative_keys = fu_all_tensor.float()
                output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
                output = output * 0.01

                #

                # print()
                # pf = []
                # tpf = []
                # for q in range(num_boxes):
                #     pf.append(f[q].tolist())
                #     tpf.append(f[q].tolist())
                # pf_array = np.array(pf, dtype='float16')
                # sf = []
                # for w in range(s_num):
                #     sf.append(f_stand[w].tolist())
                #     tpf.append(f_stand[w].tolist())
                #
                # sf_array = np.array(sf, dtype='float16')
                # tpf_array = np.array(tpf, dtype='float16')
                # tpf_num = tpf_array.shape[0]
                #
                # outlossall = 0.0
                # kmeans = KMeans(n_clusters=tpf_num - 10)
                # kmeans.fit(tpf_array)
                #
                # cluster_centers = kmeans.cluster_centers_
                # cluster_centers_tensor = torch.from_numpy(cluster_centers)
                # cluster_centers_tensor.to(device)
                #
                # cluster_centers1 = cluster_centers.astype(np.float32)
                # cluster_centers_tensor1 = torch.from_numpy(cluster_centers1)
                # cluster_centers_tensor1.to(device)
                #
                # cluser_labels = kmeans.predict(tpf_array)
                # s_cluslabel = []  # target的聚类label
                # p_cluslabel = []
                #
                # for ia in range(num_boxes):
                #     p_cluslabel.append(cluser_labels[ia])
                #
                # #
                # for i in range(s_num):
                #     s_cluslabel.append(cluser_labels[i + num_boxes])
                #
                # o = Counter(s_cluslabel)
                # o1 = dict(o)
                # smin_key = min(o1.items(), key=lambda x: x[1])[0]
                # smax_key = max(o1.items(), key=lambda x: x[1])[0]
                # w = Counter(p_cluslabel)
                # w1 = dict(w)
                # pmin_key = min(w1.items(), key=lambda x: x[1])[0]
                # pmax_key = max(w1.items(), key=lambda x: x[1])[0]
                # pcentors = []
                # # pfu = []
                # noicentors = []
                # noifu = []
                # psample = []
                # noisample = []
                #
                # for i in range(num_boxes):
                #     if p_cluslabel[i] not in s_cluslabel:  # mei有标准字库
                #         noicentors.append(cluster_centers[p_cluslabel[i]])
                #         noisample.append(f[i])
                #
                # for i in range(num_boxes):
                #     if p_cluslabel[i] in s_cluslabel:  # 有标准字库
                #         k = p_cluslabel[i]
                #         indices = [j for j, x in enumerate(s_cluslabel) if x == k]  # 找到与该样本在同一类的标准字库样本
                #         # indices_new = [y + pred_num for y in indices]
                #         a = sf_array[indices]
                #         b = pf_array[i]
                #         b = b.reshape(1, -1)
                #         dis = cdist(b, a, 'euclidean')[0]
                #         closest_index = np.argmin(dis)  # 找到在同一类且最近的标准字库样本
                #         pcentors.append(a[closest_index])  # 该样本的正样本
                #         psample.append(f[i])
                #         # dis = distance.euclidean(pf_array[i], sf_array[indices])
                #         # print(distance)
                #         # print(" ")
                # if len(pcentors) == 0:
                #     ppcentor = []
                #     ppsample = []
                #     pfu = []
                #     ppsample.append(cluster_centers[pmax_key])
                #     ppcentor.append(cluster_centers[smax_key])
                #     pfu.append(cluster_centers[smin_key])
                #     ppsample = np.array(ppsample, dtype='float32')
                #     ppsample_tensor = torch.from_numpy(ppsample)
                #     ppcentor = np.array(ppcentor, dtype='float32')
                #     ppcentor_tensor = torch.from_numpy(ppcentor)
                #     pfu = np.array(pfu, dtype='float32')
                #     pfu_tensor = torch.from_numpy(pfu)
                #     loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                #     # batch_size, num_negative, embedding_size = pred_num, k, d
                #     query = ppsample_tensor
                #     # query = torch.randn(batch_size, embedding_size)
                #     positive_key = ppcentor_tensor
                #     negative_keys = pfu_tensor
                #     output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
                #
                #     # p_centors = []
                #     # for i in p_cluslabel:
                #     #     p_centors.append(cluster_centers[i])
                #     # p_centors = np.array(p_centors,dtype='float32')
                #     # p_centors_tensor = torch.from_numpy(p_centors)
                #     # loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                #     # # batch_size, num_negative, embedding_size = pred_num, k, d
                #     # query = pred_features
                #     # # query = torch.randn(batch_size, embedding_size)
                #     # positive_key = p_centors_tensor
                #     # negative_keys = cluster_centers_tensor1
                #     # output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
                #
                #     # output = 0
                # elif len(pcentors) != 0 and len(noicentors) == 0:
                #     pcentors = np.array(pcentors, dtype='float32')
                #     pcentors_tensor = torch.from_numpy(pcentors)
                #
                #     # psample = np.array(psample,dtype='float32')
                #     psample_tensor = torch.stack(psample, dim=0)
                #     noicentors.append(cluster_centers[smin_key])
                #     noicentors = np.array(noicentors, dtype='float32')
                #     noicentors_tensor = torch.from_numpy(noicentors)
                #
                #     # for i in range(pred_num):
                #     #     if p_cluslabel[i] == min_key:  # mei有标准字库
                #     #         noicentors.append(cluster_centers[p_cluslabel[i]])
                #     #         noisample.append(pred_features[i])
                #     # noicentors = np.array(noicentors, dtype='float32')
                #     # noicentors_tensor = torch.from_numpy(noicentors)
                #     # # noisample = np.array(noisample,dtype='float32')
                #     # noisample_tensor = torch.stack(noisample, dim=0)
                #
                #     loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                #     # batch_size, num_negative, embedding_size = pred_num, k, d
                #     query = psample_tensor
                #     # query = torch.randn(batch_size, embedding_size)
                #     positive_key = pcentors_tensor
                #     negative_keys = noicentors_tensor
                #     output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
                #
                #     # query1 = noisample_tensor
                #     # # query = torch.randn(batch_size, embedding_size)
                #     # positive_key1 = noicentors_tensor
                #     # negative_keys1 = pcentors_tensor
                #     # output1 = loss(query1.to(device), positive_key1.to(device), negative_keys1.to(device))
                #
                #     output = output
                # else:
                #     # pcentors = pcentors.cpu()
                #     pcentors = np.array(pcentors, dtype='float32')
                #     pcentors_tensor = torch.from_numpy(pcentors)
                #
                #     # psample = np.array(psample,dtype='float32')
                #     psample_tensor = torch.stack(psample, dim=0)
                #
                #     noicentors = np.array(noicentors, dtype='float32')
                #     noicentors_tensor = torch.from_numpy(noicentors)
                #     # noisample = np.array(noisample,dtype='float32')
                #     noisample_tensor = torch.stack(noisample, dim=0)
                #
                #     loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                #     # batch_size, num_negative, embedding_size = pred_num, k, d
                #     query = psample_tensor
                #     # query = torch.randn(batch_size, embedding_size)
                #     positive_key = pcentors_tensor
                #     negative_keys = noicentors_tensor
                #     output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
                #
                #     query1 = noisample_tensor
                #     # query = torch.randn(batch_size, embedding_size)
                #     positive_key1 = noicentors_tensor
                #     negative_keys1 = pcentors_tensor
                #     output1 = loss(query1.to(device), positive_key1.to(device), negative_keys1.to(device))
                #
                #     output = (output + output1) / 2
                #

                # p_centors = []
                # for ix in p_cluslabel:
                #     p_centors.append(cluster_centers[ix])
                # p_centors = np.array(p_centors, dtype='float32')
                # p_centors_tensor = torch.from_numpy(p_centors)
                #
                # loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                # # batch_size, num_negative, embedding_size = pred_num, k, d
                # query = f
                # # query = torch.randn(batch_size, embedding_size)
                # positive_key = p_centors_tensor
                # negative_keys = cluster_centers_tensor1
                # output_loss = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
                # # outlossall.append(output_loss)
                # outlossall = outlossall + output_loss
                losses = {}
                losses['loss_clus'] = output

            else:
                losses = {}
                losses['loss_clus'] = 0.0

        return losses










    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'clus': self.loss_clus,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all, **kwargs)

    def forward(self, outputs, targets, outputs_stand, t_stand, feature, feature_stand, epoch):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        outputs_stand_without_aux = {k: v for k, v in outputs_stand.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, indices_all = self.matcher(outputs_without_aux, targets, 1, epoch)  #拿到了需要的框

        indices_stand = self.matcher(outputs_stand_without_aux, t_stand)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, outputs_stand, t_stand, feature, feature_stand, indices, indices_stand, num_boxes, indices_all)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 1 if args.dataset_file != 'coco' else 1
    # if args.dataset_file == "coco_panoptic":
    #     # for panoptic, we just add a num_classes that is large enough to hold
    #     # max_obj_id + 1, but the exact value doesn't really matter
    #     num_classes = 1
    num_classes = 2
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    # weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef, 'loss_clus': 1}
    # weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_clus': 1}
    weight_dict['loss_giou'] = 2
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'clus']
    if args.masks:
        losses += ["masks"]

    #
    weight_dict1 = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict1['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict1["loss_mask"] = args.mask_loss_coef
        weight_dict1["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict1 = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict1.update({k + f'_{i}': v for k, v in weight_dict1.items()})
        weight_dict1.update(aux_weight_dict1)

    losses1 = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses1 += ["masks"]
    #
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, tem=args.tem, k1=args.k1, k2=args.k2)
    criterion.to(device)
    criterion_eval = SetCriterion0(num_classes, matcher=matcher, weight_dict=weight_dict1,
                             eos_coef=args.eos_coef, losses=losses1)
    criterion_eval.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors, criterion_eval
