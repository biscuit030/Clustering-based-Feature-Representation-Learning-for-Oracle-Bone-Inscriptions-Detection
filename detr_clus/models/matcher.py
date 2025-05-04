# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import heapq
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def find_smallest_values_and_indices(self, lst, n):
        # 使用enumerate将列表转换为(key, value)对，其中key为索引
        indexed_list = list(enumerate(lst))

        # 使用heapq.nsmallest找到前n个最小值的(key, value)对
        smallest_items = heapq.nsmallest(n, indexed_list, key=lambda x: x[1])

        # 提取最小值和对应的索引
        smallest_values = [item[1] for item in smallest_items]
        indices = [item[0] for item in smallest_items]

        return indices, smallest_values

    @torch.no_grad()
    def forward(self, outputs, targets, key=0, epoch=-1):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        #







        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        if key == 0 and epoch == -1:
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        else:
            if 12 < epoch < 20:
                k = 0.1
            elif 20 <= epoch < 25:
                k = 0.15
            elif 25 <= epoch < 30:
                k = 0.2
            elif 30 <= epoch < 35:
                k = 0.25
            elif 35 <= epoch < 40:
                k = 0.3
            elif 40 <= epoch < 45:
                k = 0.35
            elif 45 <= epoch < 50:
                k = 0.4

            tensors_split_batch = torch.split(cost_giou, 100, dim=0)
            indies_all_batch = []

            # 检查拆分后的张量
            for i in range(2):

                indices_all = []
                giou = tensors_split_batch[i]#(100,7)

                tensors_split = torch.split(giou, 1, dim=0)#tuple100
                min_values = []
                if tensors_split[0].numel() > 0:
                    for t in tensors_split:
                        min_values.append(t.squeeze().min())#匹配得最好的框
                        #在每个与之最匹配的框中找正样本（>0.7转换后应该是<-0.7）和负样本(<0.3转换后应该是>-0.3)
                    # indices_fu = [i for i, tensor in enumerate(min_values) if tensor > -k and tensor < 0]
                    # indices_zheng = [i for i, tensor in enumerate(min_values) if tensor <= -k]
                    indices_fu = [i for i, tensor in enumerate(min_values) if tensor > -0.2]
                    indices_zheng = [i for i, tensor in enumerate(min_values) if tensor <= -0.2]
                    # indies, smallkey = self.find_smallest_values_and_indices(min_values, 10)
                    indices_all.append(indices_fu)
                    indices_all.append(indices_zheng)
                    indies_all_batch.append(indices_all)


            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                    indices], indies_all_batch



def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
