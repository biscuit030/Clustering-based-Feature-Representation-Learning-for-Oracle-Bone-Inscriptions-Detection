from typing import Optional, List, Dict, Tuple
import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from . import det_utils
from . import boxes as box_ops
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from info_nce import InfoNCE, info_nce
from scipy.spatial import distance
from collections import Counter
from  scipy.spatial.distance import cdist
import warnings
import time
from sklearn.manifold import TSNE
from datetime import datetime
from sklearn.cluster import MiniBatchKMeans
# from pyclustering.cluster.kmedoids import kmedoids
# from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 获取当前时间并格式化
now = datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
warnings.filterwarnings("ignore")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def ssim_loss(feature_map1, feature_map2):
#     return 1 - F.mse_loss(feature_map1, feature_map2)
huber_loss = nn.SmoothL1Loss()



def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits : 预测类别概率信息，shape=[num_anchors, num_classes]
        box_regression : 预测边目标界框回归信息
        labels : 真实类别信息
        regression_targets : 真实目标边界框信息

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # 计算类别损失信息
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    # 返回标签类别大于0的索引
    # sampled_pos_inds_subset = torch.nonzero(torch.gt(labels, 0)).squeeze(1)
    sampled_pos_inds_subset = torch.where(torch.gt(labels, 0))[0]#哪几个proposal是正样本

    # 返回标签类别大于0位置的类别信息
    labels_pos = labels[sampled_pos_inds_subset]#该返回哪一类的回归参数

    # shape=[num_proposal, num_classes]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)
    # a = box_regression[sampled_pos_inds_subset, labels_pos]

    # 计算边界框损失信息
    box_loss = det_utils.smooth_l1_loss(
        # 获取指定索引proposal的指定类别box信息

        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        size_average=False,
    ) / labels.numel()

    return classification_loss, box_loss, sampled_pos_inds_subset, labels_pos


class RoIHeads(torch.nn.Module):
    __annotations__ = {
        'box_coder': det_utils.BoxCoder,
        'proposal_matcher': det_utils.Matcher,
        'fg_bg_sampler': det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(self,
                 box_roi_pool,   # Multi-scale RoIAlign pooling
                 box_head,       # TwoMLPHead
                 box_predictor,  # FastRCNNPredictor
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,  # default: 0.5, 0.5
                 batch_size_per_image, positive_fraction,  # default: 512, 0.25
                 bbox_reg_weights,  # None
                 # Faster R-CNN inference
                 score_thresh,        # default: 0.05
                 nms_thresh,          # default: 0.5
                 detection_per_img):  # default: 100
        super(RoIHeads, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,  # default: 0.5
            bg_iou_thresh,  # default: 0.5
            allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image,  # default: 512
            positive_fraction)     # default: 0.25

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool    # Multi-scale RoIAlign pooling
        self.box_head = box_head            # TwoMLPHead
        self.box_predictor = box_predictor  # FastRCNNPredictor

        self.score_thresh = score_thresh  # default: 0.05
        self.nms_thresh = nms_thresh      # default: 0.5
        self.detection_per_img = detection_per_img  # default: 100

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        """
        为每个proposal匹配对应的gt_box，并划分到正负样本中
        Args:
            proposals:
            gt_boxes:
            gt_labels:

        Returns:

        """
        matched_idxs = []
        labels = []
        # 遍历每张图像的proposals, gt_boxes, gt_labels信息
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:  # 该张图像中没有gt框，为背景
                # background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                # 计算proposal与每个gt_box的iou重合度
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)

                # 计算proposal与每个gt_box匹配的iou最大值，并记录索引，
                # iou < low_threshold索引值为 -1， low_threshold <= iou < high_threshold索引值为 -2
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                # 限制最小值，防止匹配标签时出现越界的情况
                # 注意-1, -2对应的gt索引会调整到0,获取的标签类别为第0个gt的类别（实际上并不是）,后续会进一步处理
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # 获取proposal匹配到的gt对应标签
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]#正样本（iou达到的要求的）labels_in_image的值为1
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # label background (below the low threshold)
                # 将gt索引为-1的类别设置为0，即背景，负样本
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD  # -1
                labels_in_image[bg_inds] = 0

                # label ignore proposals (between low and high threshold)
                # 将gt索引为-2的类别设置为-1, 即废弃样本
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS  # -2
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels#需要一些“背景”框做负样本

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        # BalancedPositiveNegativeSampler
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        # 遍历每张图片的正负样本索引
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            # 记录所有采集样本索引（包括正样本和负样本）
            # img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        """
        将gt_boxes拼接到proposal后面
        Args:
            proposals: 一个batch中每张图像rpn预测的boxes
            gt_boxes:  一个batch中每张图像对应的真实目标边界框

        Returns:

        """
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        assert targets is not None
        assert all(["boxes" in t for t in targets])
        assert all(["labels" in t for t in targets])

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        划分正负样本，统计对应gt的标签以及边界框回归信息
        list元素个数为batch_size
        Args:
            proposals: rpn预测的boxes
            targets:

        Returns:

        """

        # 检查target数据是否为空
        self.check_targets(targets)
        # 如果不加这句，jit.script会不通过(看不懂)
        assert targets is not None

        dtype = proposals[0].dtype
        device = proposals[0].device

        # 获取标注好的boxes以及labels信息
        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to proposal
        # 将gt_boxes拼接到proposal后面
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        # 为每个proposal匹配对应的gt_box，并划分到正负样本中
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        # 按给定数量和比例采样正负样本
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)

        # 遍历每张图像
        for img_id in range(num_images):
            # 获取每张图像的正负样本索引
            img_sampled_inds = sampled_inds[img_id]
            # 获取对应正负样本的proposals信息
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            # 获取对应正负样本的真实类别信息
            labels[img_id] = labels[img_id][img_sampled_inds]
            # 获取对应正负样本的gt索引信息
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            # 获取对应正负样本的gt box信息
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        # 根据gt和proposal计算边框回归参数（针对gt的）
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, labels, regression_targets

    def postprocess_detections(self,
                               class_logits,    # type: Tensor
                               box_regression,  # type: Tensor
                               proposals,       # type: List[Tensor]
                               image_shapes     # type: List[Tuple[int, int]]
                               ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        对网络的预测数据进行后处理，包括
        （1）根据proposal以及预测的回归参数计算出最终bbox坐标
        （2）对预测类别结果进行softmax处理
        （3）裁剪预测的boxes信息，将越界的坐标调整到图片边界上
        （4）移除所有背景信息
        （5）移除低概率目标
        （6）移除小尺寸目标
        （7）执行nms处理，并按scores进行排序
        （8）根据scores排序返回前topk个目标
        Args:
            class_logits: 网络预测类别概率信息
            box_regression: 网络预测的边界框回归参数
            proposals: rpn输出的proposal
            image_shapes: 打包成batch前每张图像的宽高

        Returns:

        """
        device = class_logits.device
        # 预测目标类别数
        num_classes = class_logits.shape[-1]

        # 获取每张图像的预测bbox数量
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        # 根据proposal以及预测的回归参数计算出最终bbox坐标
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # 对预测类别结果进行softmax处理
        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        # 根据每张图像的预测bbox数量分割结果
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        # 遍历每张图像预测信息
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            # 裁剪预测的boxes信息，将越界的坐标调整到图片边界上
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove prediction with the background label
            # 移除索引为0的所有信息（0代表背景）
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 移除低概率目标，self.scores_thresh=0.05
            # gt: Computes input > other element-wise.
            inds = torch.nonzero(torch.gt(scores, self.score_thresh)).squeeze(1)
            inds = torch.where(torch.gt(scores, self.score_thresh))[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            # 移除小目标
            keep = box_ops.remove_small_boxes(boxes, min_size=1.)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximun suppression, independently done per class
            # 执行nms处理，执行后的结果会按照scores从大到小进行排序返回
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)

            # keep only topk scoring predictions
            # 获取scores排在前topk个预测目标
            keep = keep[:self.detection_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self,
                # i,
                features,       # type: Dict[str, Tensor]
                proposals,      # type: List[Tensor]
                image_shapes,   # type: List[Tuple[int, int]]
                targets=None,    # type: Optional[List[Dict[str, Tensor]]]
                stand_features1 = None,
                tem = None,
                clusk1 = None,

                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """

        # 检查targets的数据类型是否正确
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            # 划分正负样本，统计对应gt的标签以及边界框回归信息
            proposals, labels, regression_targets = self.select_training_samples(proposals, targets)

            # proposals_stand, labels_stand, regression_targets_stand = self.select_training_samples(proposals_stand, targets_stand)
            # print()

            # tb = []
            # for i in targets:
            #     targets_box = i['boxes']
            #     tb.append(targets_box)
            #
            # target_features = self.box_roi_pool(features, tb, image_shapes)
            # g = target_features
            # print("t",target_features)
        else:
            labels = None
            regression_targets = None

        # 将采集样本通过Multi-scale RoIAlign pooling层
        # box_features_shape: [num_proposals, channel, height, width]
        box_features = self.box_roi_pool(features, proposals, image_shapes)

        # box_features_stand = self.box_roi_pool(standard_feature, proposals_stand, image_shapes_stand)
        #
        # tb = []
        # for i in targets:
        #     targets_box = i['boxes']
        # tb.append(targets_box)
        # target_features = self.box_roi_pool(features, tb, image_shapes)
        #

        # 通过roi_pooling后的两层全连接层
        # box_features_shape: [num_proposals, representation_size]
        box_features1 = self.box_head(box_features)
        # box_features_stand1 = self.box_head(box_features_stand)


        # 接着分别预测目标类别和边界框回归参数
        class_logits, box_regression = self.box_predictor(box_features1)

        # class_logits_stand, box_regression_stand = self.box_predictor(box_features_stand1)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None

            # box_features_stand = self.box_roi_pool(standard_feature, proposals_stand, image_shapes_stand)
            # box_features_stand1 = self.box_head(box_features_stand)
            # class_logits_stand, box_regression_stand = self.box_predictor(box_features_stand1)

            loss_classifier, loss_box_reg, proposal_id, labels_pos = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)

            # loss_classifier_stand, loss_box_reg_stand, proposal_id_stand, labels_pos_stand = fastrcnn_loss(
            #     class_logits_stand, box_regression_stand, labels_stand, regression_targets_stand)

            # losses = {
            #     "loss_classifier": loss_classifier,
            #     "loss_box_reg": loss_box_reg
            # }
            # for i in proposals:
            #     a = i[509]
            #     pred_boxes = self.box_coder.decode(box_regression[proposal_id][labels_pos], i[proposal_id])
            #     # print("1")

            #
            # first = []
            # second = []
            # for i in proposal_id:
            #     if i <=512:
            #         first.append(i)
            #     else:
            #         second.append(i)
            # first_num = len(first)
            # second_num = len(second)
            #
            #
            # standard_box = []#tensor
            # for i in range(20):
            #     s = [[0, 0, 100, 100],[0, 0, 100, 100]]
            #     s = np.array(s,dtype='float32')
            #     s = torch.from_numpy(s)
            #     s.to(device)
            #     standard_box.append(s.to(device))

            # standard_box_array = np.array(standard_box)
            # standard_box_tensor = torch.from_numpy(standard_box_array)

            # standard_fea2 = self.box_roi_pool(standard_feature, standard_box, [100,100])
            # standard_fea1 = standard_fea2[0:40:2,:]
            # pred_features_stand = box_features_stand[proposal_id_stand]
            # stand_nun = pred_features_stand.shape[0]


            if stand_features1 != None:

                proposal_id_list = proposal_id.tolist()
                pred_features_stand = stand_features1
                stand_nun = stand_features1.shape[0]


                #
                # pred_boxes_all = self.box_coder.decode(box_regression, proposals)#bs个proposal的框的坐标
                # pred_boxes = pred_boxes_all[proposal_id, labels_pos]#bs个pred的框的坐标
                # pb = []
                #
                # pb.append(pred_boxes[0:first_num:1,:])
                # pb.append(pred_boxes[first_num:first_num + second_num:1,:])
                # pred_features = self.box_roi_pool(features, pb, image_shapes)
                # y = pred_features
                # # print("p",pred_features)

                # pred_features = box_features[proposal_id]
                pred_features = box_features
                pred_num = pred_features.shape[0]

                pred_features = pred_features.reshape(pred_num, -1)
                pred_features.to(device)
                # target_num = target_features.shape[0]
                # target_features = target_features.reshape(target_num, -1)

                # standard_num = standard_fea1.shape[0]
                #
                # stan_features = standard_fea1.reshape(standard_num, -1)

                # start_time = time.perf_counter()
                tpf = []#target和pre总的特征，用来聚类
                sf = []
                tpb = []

                tf = []
                # for i in range(target_num):
                #     tf.append(target_features[i].tolist())
                #     # tpf.append(target_features[i].tolist())
                tf_array = np.array(tf, dtype='float16')
                #
                p1f = []
                p0f = []
                psample = []
                pzheng = []
                # pfu = []
                for i in range(pred_num):
                    # pf.append(pred_features[i].tolist())
                    # tpf.append(pred_features[i].tolist())
                    if i in proposal_id_list:
                        psample.append(pred_features[i])
                        p1f.append(pred_features[i].tolist())
                        tpb.append(0)
                    else:
                        p0f.append(pred_features[i].tolist())

                        tpb.append(-1)
                p1f_array = np.array(p1f, dtype='float64')
                p0f_array = np.array(p0f, dtype='float64')
                if len(p1f_array)==0 or len(p0f_array) == 0:
                    output = 0

                else:
                    for i in range(stand_nun):
                        sf.append(pred_features_stand[i].tolist())
                        tpf.append(pred_features_stand[i].tolist())
                        tpb.append(1)
                    sf_array = np.array(sf, dtype='float64')

                    tpf_array = np.array(tpf, dtype='float16')
                    # end_time = time.perf_counter()
                    #
                    # # 输出运行时间
                    # print(f"代码运行时间1：{end_time - start_time:.6f}秒")

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
                    # k = tf_array.shape[0]#target有几个框
                    # k1 = sf_array.shape[0]
                    # d = pf_array.shape[1]#特征维度
                    # tpf_num = tpf_array.shape[0]

                    # kmeans = KMeans(n_clusters=k, init=tf_array)
                    #对于每个样本（若簇中有标准字库）找到与他同类且最近的标准字库样本

                    #
                    # start_time = time.perf_counter()
                    p1_num = p1f_array.shape[0]#样本
                    p0_num = p0f_array.shape[0]#负样本
                    # k1 = p0_num-100
                    k1 = 100
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

                    #######
                    p0f_array_db = StandardScaler().fit_transform(p0f_array)
                    # from sklearn.neighbors import NearestNeighbors
                    # import matplotlib.pyplot as plt
                    #
                    # # 计算最近邻距离
                    # neighbors = NearestNeighbors(n_neighbors=10)  # n_neighbors=min_samples + 1
                    # neighbors_fit = neighbors.fit(p0f_array_db)
                    # distances, indices = neighbors_fit.kneighbors(p0f_array_db)
                    #
                    # # 将距离按升序排列
                    # distances = np.sort(distances, axis=0)
                    # distances = distances[:, 1]
                    #
                    # # 绘制K-distance图
                    # plt.plot(distances)
                    # plt.title('K-distance Graph')
                    # plt.xlabel('Data Points sorted by distance')
                    # plt.ylabel('Epsilon')
                    # plt.show()
                    # 配置DBSCAN
                    # db = DBSCAN(eps=8, min_samples=5).fit(p0f_array_db)

                    # 获取聚类标签，-1表示噪声点
                    # labels = db.labels_
                    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    #
                    # # 初始化一个列表来保存每个簇的中心点
                    # cluster_centers9 = []
                    #
                    # for i in range(n_clusters):
                    #     # 获取属于当前簇i的所有点的索引
                    #     indices = np.where(labels == i)[0]
                    #
                    #     # 计算这些点的平均值作为簇的中心点
                    #     center = np.mean(p0f_array_db[indices], axis=0)
                    #
                    #     # 将中心点添加到列表中
                    #     cluster_centers9.append(center)
                    #
                    # # 将结果转换为numpy数组方便操作
                    # cluster_centers9 = np.array(cluster_centers9)
                    # cluster_centers1 = cluster_centers9.astype(np.float32)
                    # cluster_centers_tensor1 = torch.from_numpy(cluster_centers1)
                    ########
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
                    # print()
                    # cluster_centers_tensor1.to(device)
                    # end_time = time.perf_counter()
                    #
                    # # 输出运行时间
                    # print(f"代码运行时间2：{end_time - start_time:.6f}秒")

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
                    k2 = 1
                    #
                    # # initial_medoids = kmeans_plusplus_initializer(sf_array, k2).initialize()
                    # # initial_medoids = [np.where(np.all(sf_array == center, axis=1))[0][0] for center in initial_medoids]
                    # #
                    # # kmedoids_instance = kmedoids(sf_array, initial_medoids)  # 假设我们想要将数据分为2个聚类
                    # #
                    # # # 执行聚类
                    # #
                    # # kmedoids_instance.process()
                    # # clusters = kmedoids_instance.get_clusters()
                    # # medoids = kmedoids_instance.get_medoids()
                    #
                    # # start_time = time.perf_counter()
                    kmeans = KMeans(n_clusters=k2)
                    # # mbk = MiniBatchKMeans(
                    # #     n_clusters=k2,  # 要形成的簇的数量
                    # #     init='k-means++',  # 初始化质心的方法：'k-means++' 或 'random'
                    # #     max_iter=100,  # 单次运行中执行的最大迭代次数
                    # #     batch_size=100,  # 每批样本数量
                    # #     verbose=0,  # 是否详细输出日志信息；0为不输出
                    # #     compute_labels=False,  # 是否计算标签
                    # #     random_state=None,  # 随机种子，用于初始化质心
                    # #     tol=0.0,  # 相对于惯性的容忍度，用于声明收敛
                    # #     max_no_improvement=10,  # 如果连续这么多批都没有改善聚类效果，则提前终止算法
                    # #     init_size=None,  # 初始化时使用的样本数；如果为None，则默认为n_clusters * 3
                    # #     n_init=3,  # 使用不同质心种子运行算法的次数
                    # #     reassignment_ratio=0.01  # 控制重新分配的阈值
                    # # )
                    # #
                    # # mbk.fit(sf_array)
                    # # cluster_centers = mbk.cluster_centers_
                    kmeans.fit(sf_array)
                    cluster_centers9 = kmeans.cluster_centers_
                    #######
                    # p1f_array_db = StandardScaler().fit_transform(p1f_array)
                    # db = DBSCAN(eps=12, min_samples=5).fit(p1f_array_db)
                    #
                    # # 获取聚类标签，-1表示噪声点
                    # labels = db.labels_
                    # n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    #
                    # # 初始化一个列表来保存每个簇的中心点
                    # cluster_centers9 = []
                    #
                    # for i in range(n_clusters):
                    #     # 获取属于当前簇i的所有点的索引
                    #     indices = np.where(labels == i)[0]
                    #
                    #     # 计算这些点的平均值作为簇的中心点
                    #     center = np.mean(p0f_array_db[indices], axis=0)
                    #
                    #     # 将中心点添加到列表中
                    #     cluster_centers9.append(center)
                    #
                    # # 将结果转换为numpy数组方便操作
                    # cluster_centers9 = np.array(cluster_centers9)
                    # cluster_centers9 = cluster_centers9.astype(np.float32)
                    # cluster_centers_tensor9 = torch.from_numpy(cluster_centers9)
                    ######
                    c = np.mean(cluster_centers9, axis=0)
                    # c = torch.from_numpy(c)
                    # c0 = c[np.newaxis, :]
                    # cluster_centers9 = c0
                    # print()
                    # end_time = time.perf_counter()
                    # print(f"代码运行时间3：{end_time - start_time:.6f}秒")

                    #
                    # sf_num = sf_array.shape[0]
                    for k in range(p1_num):

                        pzheng.append(c)
                    pzhengcentors = np.array(pzheng, dtype='float32')
                    #pzhengcentors_tensor = torch.stack(pzhengcentors, dim=0)
                    pzhengcentors_tensor = torch.from_numpy(pzhengcentors)

                    # end_time = time.perf_counter()
                    #
                    # # 输出运行时间
                    # print(f"代码运行时间3：{end_time - start_time:.6f}秒")

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
                    # start_time = time.perf_counter()
                    psample_tensor = torch.stack(psample, dim=0)
                    #
                    loss = InfoNCE(temperature=tem, negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                    # batch_size, num_negative, embedding_size = pred_num, k, d
                    query = psample_tensor
                    # query = torch.randn(batch_size, embedding_size)
                    # positive_key = pz_array_tensor
                    positive_key = pzhengcentors_tensor
                    negative_keys = cluster_centers_tensor1
                    output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
                    # output = output*0.01

                    # end_time = time.perf_counter()

                    # 输出运行时间
                    # print(f"代码运行时间4：{end_time - start_time:.6f}秒")
                # print(" ")

                #



#######################
                '''
                kmeans = KMeans(n_clusters=tpf_num-10)

                kmeans.fit(tpf_array)


                cluster_centers = kmeans.cluster_centers_
                cluster_centers_tensor = torch.from_numpy(cluster_centers)
                cluster_centers_tensor.to(device)

                cluster_centers1 = cluster_centers.astype(np.float32)
                cluster_centers_tensor1 = torch.from_numpy(cluster_centers1)
                cluster_centers_tensor1.to(device)


                cluser_labels = kmeans.predict(tpf_array)
                s_cluslabel = []#target的聚类label
                p_cluslabel = []

                #
                # tsne = TSNE(n_components=2, random_state=4)
                # reduced_target = tsne.fit_transform(tpf_array)
                # # plt.scatter(reduced_target[:, 0], reduced_target[:, 1], c=cluser_labels, cmap='rainbow')
                # # plt.title('t-SNEclus Visualization')
                # # folder_path = 'E:/PycharmProjects/faster_rcnn/cluster1'
                # # plt.savefig(os.path.join(folder_path, str(i)+f'simple_plot_{formatted_time}.png'))
                # #
                # plt.scatter(reduced_target[:, 0], reduced_target[:, 1], c=tpb, cmap='viridis')
                # plt.title('t-SNEcla Visualization')
                # folder_path1 = 'E:/PycharmProjects/faster_rcnn/class2'
                # plt.savefig(os.path.join(folder_path1, str(i)+f'simple_plot_{formatted_time}.png'))


                # plt.savefig(E:\PycharmProjects\faster_rcnn\cluster+f'simple_plot_{formatted_time}.png')
                #

                # for i in range(target_num):
                #     t_cluslabel.append(cluser_labels[i])
                # for i in range(target_num,pred_num + target_num):
                #     p_cluslabel.append(cluser_labels[i])

                for i in range(pred_num):
                    p_cluslabel.append(cluser_labels[i])

                for i in range(stand_nun):
                    s_cluslabel.append(cluser_labels[i+pred_num])

                o = Counter(s_cluslabel)
                o1 = dict(o)
                smin_key = min(o1.items(), key=lambda x: x[1])[0]
                smax_key = max(o1.items(), key=lambda x: x[1])[0]
                w = Counter(p_cluslabel)
                w1 = dict(w)
                pmin_key = min(w1.items(), key=lambda x: x[1])[0]
                pmax_key = max(w1.items(), key=lambda x: x[1])[0]

                # 对于每个样本（若簇中有标准字库）找到与他同类且最近的标准字库样本
                pcentors = []
                # pfu = []
                noicentors = []
                noifu = []
                psample = []
                noisample = []

                for i in range(pred_num):
                    if p_cluslabel[i] not in s_cluslabel:#mei有标准字库
                        noicentors.append(cluster_centers[p_cluslabel[i]])
                        noisample.append(pred_features[i])





                for i in range(pred_num):
                    if p_cluslabel[i] in s_cluslabel:#有标准字库
                        k = p_cluslabel[i]
                        indices = [j for j, x in enumerate(s_cluslabel) if x == k]#找到与该样本在同一类的标准字库样本
                        # indices_new = [y + pred_num for y in indices]
                        a = sf_array[indices]
                        b = pf_array[i]
                        b = b.reshape(1, -1)
                        dis = cdist(b, a, 'euclidean')[0]
                        closest_index = np.argmin(dis)#找到在同一类且最近的标准字库样本
                        pcentors.append(a[closest_index])#该样本的正样本
                        psample.append(pred_features[i])
                        # dis = distance.euclidean(pf_array[i], sf_array[indices])
                        # print(distance)
                        # print(" ")
                if len(pcentors) == 0:
                    ppcentor = []
                    ppsample = []
                    pfu = []
                    ppsample.append(cluster_centers[pmax_key])
                    ppcentor.append(cluster_centers[smax_key])
                    pfu.append(cluster_centers[smin_key])
                    ppsample = np.array(ppsample, dtype='float32')
                    ppsample_tensor = torch.from_numpy(ppsample)
                    ppcentor = np.array(ppcentor, dtype='float32')
                    ppcentor_tensor = torch.from_numpy(ppcentor)
                    pfu = np.array(pfu, dtype='float32')
                    pfu_tensor = torch.from_numpy(pfu)
                    loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                    # batch_size, num_negative, embedding_size = pred_num, k, d
                    query = ppsample_tensor
                    # query = torch.randn(batch_size, embedding_size)
                    positive_key = ppcentor_tensor
                    negative_keys = pfu_tensor
                    output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))

                    # p_centors = []
                    # for i in p_cluslabel:
                    #     p_centors.append(cluster_centers[i])
                    # p_centors = np.array(p_centors,dtype='float32')
                    # p_centors_tensor = torch.from_numpy(p_centors)
                    # loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                    # # batch_size, num_negative, embedding_size = pred_num, k, d
                    # query = pred_features
                    # # query = torch.randn(batch_size, embedding_size)
                    # positive_key = p_centors_tensor
                    # negative_keys = cluster_centers_tensor1
                    # output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))

                    # output = 0
                elif len(pcentors) != 0 and len(noicentors)== 0:
                    pcentors = np.array(pcentors, dtype='float32')
                    pcentors_tensor = torch.from_numpy(pcentors)

                    # psample = np.array(psample,dtype='float32')
                    psample_tensor = torch.stack(psample, dim=0)
                    noicentors.append(cluster_centers[smin_key])
                    noicentors = np.array(noicentors, dtype='float32')
                    noicentors_tensor = torch.from_numpy(noicentors)

                    # for i in range(pred_num):
                    #     if p_cluslabel[i] == min_key:  # mei有标准字库
                    #         noicentors.append(cluster_centers[p_cluslabel[i]])
                    #         noisample.append(pred_features[i])
                    # noicentors = np.array(noicentors, dtype='float32')
                    # noicentors_tensor = torch.from_numpy(noicentors)
                    # # noisample = np.array(noisample,dtype='float32')
                    # noisample_tensor = torch.stack(noisample, dim=0)

                    loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                    # batch_size, num_negative, embedding_size = pred_num, k, d
                    query = psample_tensor
                    # query = torch.randn(batch_size, embedding_size)
                    positive_key = pcentors_tensor
                    negative_keys = noicentors_tensor
                    output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))

                    # query1 = noisample_tensor
                    # # query = torch.randn(batch_size, embedding_size)
                    # positive_key1 = noicentors_tensor
                    # negative_keys1 = pcentors_tensor
                    # output1 = loss(query1.to(device), positive_key1.to(device), negative_keys1.to(device))

                    output = output
                else:
                    # pcentors = pcentors.cpu()
                    pcentors = np.array(pcentors, dtype='float32')
                    pcentors_tensor = torch.from_numpy(pcentors)

                    # psample = np.array(psample,dtype='float32')
                    psample_tensor = torch.stack(psample, dim=0)

                    noicentors = np.array(noicentors, dtype='float32')
                    noicentors_tensor = torch.from_numpy(noicentors)
                    # noisample = np.array(noisample,dtype='float32')
                    noisample_tensor = torch.stack(noisample, dim=0)

                    loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                    # batch_size, num_negative, embedding_size = pred_num, k, d
                    query = psample_tensor
                    # query = torch.randn(batch_size, embedding_size)
                    positive_key = pcentors_tensor
                    negative_keys = noicentors_tensor
                    output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))

                    query1 = noisample_tensor
                    # query = torch.randn(batch_size, embedding_size)
                    positive_key1 = noicentors_tensor
                    negative_keys1 = pcentors_tensor
                    output1 = loss(query1.to(device), positive_key1.to(device), negative_keys1.to(device))

                    output = (output + output1)/2
                    '''
##############







                ##
                # p_centors = []
                # for i in p_cluslabel:
                #     p_centors.append(cluster_centers[i])
                # p_centors = np.array(p_centors,dtype='float32')
                # p_centors_tensor = torch.from_numpy(p_centors)
                ##

                # tp_centors = []
                # for i in cluser_labels:
                #     tp_centors.append(cluster_centers[i])
                # tp_centors = np.array(tp_centors, dtype='float16')
                # tp_centors_tensor = torch.from_numpy(tp_centors)


                # def euclidean_distance(x, y):
                #     return np.sqrt(np.sum((x - y) ** 2))


                # dist = 0.0
                # for i in range(pred_num):
                #     i_label = p_cluslabel[i]#该预测图片的聚类类别
                #     if i_label in t_cluslabel:#该类别中有target
                #         dist = dist + euclidean_distance(pf_array[i],cluster_centers[p_cluslabel[i]])#pred与自己类簇中心的距离
                #     else:
                #         dist_case = []
                #         target_clus_set = set(t_cluslabel)
                #         target_clus_list = list(target_clus_set)
                #         for j in target_clus_list:
                #             dist_case.append(euclidean_distance(pf_array[i],cluster_centers[j]))#pred与最近的有target的类簇中心的距离
                #
                #         dist = dist + min(dist_case)

                # dist = 0.0
                #
                # bs = len(image_shapes)
                # tem = 0.5



                # noise = []
                # noise_label = []
                # for i in range(pred_num):
                #     i_label = p_cluslabel[i]  # 该预测图片的聚类类别
                #
                #     if i_label not in t_cluslabel:  # 该类别中没有target
                #         noise.append(i)
                #         noise_label.append(i_label)
                #
                #
                # for i in range(pred_num):
                #     i_label = p_cluslabel[i]  # 该预测图片的聚类类别
                #     if i_label in t_cluslabel:  # 该类别中有target
                #         # dist = dist + ssim_loss(pred_features[i], cluster_centers_tensor[p_cluslabel[i]])  # pred与自己类簇中心的距离
                #         for n in noise_label:
                #             dist = dist + \
                #                    (pred_features[i].to(device),
                #                                      cluster_centers_tensor[n].to(device))
                #         if len(noise) != 0:
                #             dist = dist / len(noise)
                #         # dist = dist + huber_loss(pred_features[i].to(device), cluster_centers_tensor[p_cluslabel[i]].to(device))
                #         # print("")
                #     else:
                #
                #         # dist_case = []
                #         # target_clus_set = set(t_cluslabel)
                #         # target_clus_list = list(target_clus_set)
                #         # for j in target_clus_list:
                #         #     dist_case.append(huber_loss(pred_features[i].to(device), cluster_centers_tensor[j].to(device)).item())  # pred与最近的有target的类簇中心的距离
                #         #
                #         # dist = dist + min(dist_case)
                #         for t in t_cluslabel:
                #             dist = dist + huber_loss(pred_features[i].to(device),
                #                                      cluster_centers_tensor[t].to(device))
                #         dist = dist / len(t_cluslabel)
                #
                #
                # dist = dist / pred_num



                # noise = []
                # noise_label = []
                # for i in range(pred_num):
                #     i_label = p_cluslabel[i]  # 该预测图片的聚类类别
                #
                #     if i_label not in t_cluslabel:  # 该类别中没有target
                #         noise.append(i)
                #         noise_label.append(i_label)


                # pre_loss_sum = 0.0
                #
                # for i in range(pred_num):
                #
                #     # if hasattr(torch.cuda, 'empty_cache'):
                #     #     torch.cuda.empty_cache()
                #     if i in noise:
                #         i_label = p_cluslabel[i]  # 该预测图片的聚类类别
                #         center_s = F.cosine_similarity(cluster_centers_tensor[p_cluslabel[i]].unsqueeze(0).to(device),
                #                                        pred_features[i].unsqueeze(0).to(device)) / tem
                #         e_center_s = torch.exp(center_s)
                #         e_other_s = e_center_s
                #         for j in cluser_labels:
                #             if j not in noise_label:
                #                 e_other_s = e_other_s + torch.exp(
                #                     F.cosine_similarity(cluster_centers_tensor[j].unsqueeze(0).to(device),
                #                                         pred_features[i].unsqueeze(0).to(device)) / tem)
                #
                #         pre_loss_sum = pre_loss_sum + -torch.log(e_center_s / e_other_s)
                #
                #     else:
                #         center_s1 = F.cosine_similarity(cluster_centers_tensor[p_cluslabel[i]].unsqueeze(0).to(device),
                #                                        pred_features[i].unsqueeze(0).to(device)) / tem
                #         e_center_s1 = torch.exp(center_s1)
                #         e_noise_s = e_center_s1
                #         for j in noise_label:
                #             e_noise_s = e_noise_s + torch.exp(
                #                 F.cosine_similarity(cluster_centers_tensor[j].unsqueeze(0).to(device),
                #                                     pred_features[i].unsqueeze(0).to(device)) / tem)
                #         pre_loss_sum = pre_loss_sum + -torch.log(e_center_s1 / e_noise_s)

                    # i_label = p_cluslabel[i]  # 该预测图片的聚类类别
                    # center_s = F.cosine_similarity(cluster_centers_tensor[p_cluslabel[i]].unsqueeze(0).to(device), pred_features[i].unsqueeze(0).to(device)) / tem
                    # e_center_s = torch.exp(center_s)
                    # e_other_s = 0.0

                    # if hasattr(torch.cuda, 'empty_cache'):
                    #     torch.cuda.empty_cache()

                    # for j in cluser_labels:
                    #     e_other_s = e_other_s + torch.exp(
                    #         F.cosine_similarity(cluster_centers_tensor[j].unsqueeze(0).to(device),
                    #                             pred_features[i].unsqueeze(0).to(device)) / tem)
                        # if j != i_label:


                        # e_other_s = e_other_s + torch.exp(F.cosine_similarity(cluster_centers_tensor[j].unsqueeze(0).to(device), pred_features[i].unsqueeze(0).to(device)) / tem)

                    # pre_loss_sum = pre_loss_sum + -torch.log(e_center_s / e_other_s)


                # pre_loss_sum = pre_loss_sum / pred_num





                # print("+++")
                # print(dist)
                # print("+++")


                ##
                # loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value
                # batch_size, num_negative, embedding_size = pred_num, k, d
                # query = pred_features
                # # query = torch.randn(batch_size, embedding_size)
                # positive_key = p_centors_tensor
                # negative_keys = cluster_centers_tensor1
                # output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
                ##





            else:
                output = 0

            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
                "loss_clster":output
            }


        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses
