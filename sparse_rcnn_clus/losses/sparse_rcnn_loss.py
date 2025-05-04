import torch
from losses.commons import IOULoss, BoxSimilarity, focal_loss
from scipy.optimize import linear_sum_assignment
from utils.model_utils import reduce_sum, get_gpu_num_solo
import numpy as np
from sklearn.cluster import KMeans
from info_nce import InfoNCE, info_nce
import heapq
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class BoxCoder(object):
    def __init__(self, weights=None):
        super(BoxCoder, self).__init__()
        if weights is None:
            weights = [0.1, 0.1, 0.2, 0.2]
        self.weights = torch.tensor(data=weights, requires_grad=False)

    def encoder(self, anchors, gt_boxes):
        """
        :param gt_boxes:[box_num, 4]
        :param anchors: [box_num, 4]
        :return:
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = torch.cat([delta_xy, delta_wh], dim=-1) / self.weights

        return delta_targets

    def decoder(self, predicts, anchors):
        """
        :param predicts: [anchor_num, 4] or [bs, anchor_num, 4]
        :param anchors: [anchor_num, 4]
        :return: [anchor_num, 4] (x1,y1,x2,y2)
        """
        if self.weights.device != anchors.device:
            self.weights = self.weights.to(anchors.device)
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.weights
        scale_wh = scale_reg[..., 2:].exp() * anchors_wh
        scale_x1y1 = (anchors_xy + scale_reg[..., :2] * anchors_wh) - 0.5 * scale_wh
        scale_x2y2 = scale_x1y1 + scale_wh
        return torch.cat([scale_x1y1, scale_x2y2], dim=-1)


class HungarianMatcher(object):
    def __init__(self, alpha=0.25, gamma=2.0, cls_cost=1, iou_cost=1, l1_cost=1):
        super(HungarianMatcher, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cls_cost = cls_cost
        self.iou_cost = iou_cost
        self.l1_cost = l1_cost
        self.similarity = BoxSimilarity(iou_type="giou")
    def find_smallest_values_and_indices(self, lst, n):
        # 使用enumerate将列表转换为(key, value)对，其中key为索引
        indexed_list = list(enumerate(lst))

        # 使用heapq.nsmallest找到前n个最小值的(key, value)对
        smallest_items = heapq.nsmallest(n, indexed_list, key=lambda x: x[1])

        # 提取最小值和对应的索引
        # smallest_values = [item[1] for item in smallest_items]
        indices = [item[0] for item in smallest_items]

        return indices

    @torch.no_grad()
    def __call__(self, predicts_cls, predicts_box, gt_boxes, shape_norm, key=0, epoch=-1):
        """
        :param predicts_box: [bs,proposal,80]
        :param predicts_cls: [bs,proposal,4]
        :param gt_boxes: [(num,5)](label_idx,x1,y1,x2,y2)
        :return:
        """
        bs, num_queries = predicts_cls.shape[:2]
        predicts_cls = predicts_cls.view(-1, predicts_cls.size(-1)).sigmoid()
        predicts_box = predicts_box.view(-1, predicts_box.size(-1))
        combine_gts = torch.cat(gt_boxes)
        gt_num = [len(i) for i in gt_boxes]
        positive_loss = -self.alpha * ((1 - predicts_cls) ** self.gamma) * predicts_cls.log()
        negative_loss = - (1 - self.alpha) * (predicts_cls ** self.gamma) * ((1 - predicts_cls).log())
        cls_cost = positive_loss[:, combine_gts[:, 0].long()] - negative_loss[:, combine_gts[:, 0].long()]

        pred_norm = predicts_box / shape_norm[None, :]
        target_norm = combine_gts[:, 1:] / shape_norm[None, :]
        l1_cost = torch.cdist(pred_norm, target_norm, p=1)

        pred_expand = predicts_box[:, None, :].repeat(1, len(combine_gts), 1).view(-1, 4)
        target_expand = combine_gts[:, 1:][None, :, :].repeat(len(predicts_box), 1, 1).view(-1, 4)
        iou_cost = -self.similarity(pred_expand, target_expand).view(len(predicts_cls), -1)
        cost = self.iou_cost * iou_cost + self.cls_cost * cls_cost + self.l1_cost * l1_cost
        cost = cost.view(bs, num_queries, -1).cpu()
        ret = list()
        for i, item in enumerate(cost.split(gt_num, -1)):
            if item.shape[-1] == 0:
                continue
            indices = linear_sum_assignment(item[i])
            ret.append((i, indices[0].tolist(), indices[1].tolist()))
        # print()
        if key == 1 and epoch != -1:
            sizes = [num_queries] * bs  # 每个子张量的大小为 64
            iou_cost_split = torch.split(iou_cost, split_size_or_sections=sizes, dim=0)  # 4个
            indies_all_batch = []

           

            # 检查拆分后的张量
            for i in range(bs):  # batchsize
                indices_all = []
                giou = iou_cost_split[i]
                tensors_split = torch.split(giou, 1, dim=0)
                min_values = []
                if tensors_split[0].numel() > 0:
                    for t in tensors_split:
                        min_values.append(t.squeeze().min())  # 匹配得最好的框
                        # 在每个与之最匹配的框中找正样本（>0.7转换后应该是<-0.7）和负样本(<0.3转换后应该是>-0.3)
                    # print(min_values)
                    average = torch.stack(min_values).mean().item()
                    indices_fu = [i for i, tensor in enumerate(min_values) if tensor >= average]
                    indices_zheng = [i for i, tensor in enumerate(min_values) if tensor < average]
                    # indies, smallkey = self.find_smallest_values_and_indices(min_values, 10)
                    indices_all.append(indices_fu)
                    indices_all.append(indices_zheng)
                    indies_all_batch.append(indices_all)

                # 对每个拆分后的张量求最小值,得到最小值列表list（200）

            # print("")
            return ret, indies_all_batch
        else:
            return ret
class SparseRCNNLoss(object):
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 iou_weights=2.0,
                 cls_weights=2.0,
                 l1_weights=5.0,
                 iou_type="giou",
                 iou_cost=1.0,
                 cls_cost=1.0,
                 l1_cost=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.iou_weights = iou_weights
        self.cls_weights = cls_weights
        self.l1_weights = l1_weights
        self.iou_loss = IOULoss(iou_type=iou_type)
        self.matcher = HungarianMatcher(
            iou_cost=iou_cost,
            cls_cost=cls_cost,
            l1_cost=l1_cost
        )

    def __call__(self, cls_predicts, cls_predicts_stand, reg_predicts, reg_predicts_stand, targets, shape, roi_fea, roi_fea_stand, shape_stand, epoch=None):
        #
        # hs, ws = shape_stand
        # shape_norms = torch.tensor([ws, hs, ws, hs], device=cls_predicts.device)
        # pos_nums = len(targets['target_stand'])
        # gt_boxess = targets['target_stand'].split(targets['batch_len_stand'])
        # if cls_predicts_stand.dtype == torch.float16:
        #     cls_predicts_stand = cls_predicts_stand.float()
        # all_imme_idxs = list()
        # all_batch_idxs = list()
        # all_proposal_idxs = list()
        # all_cls_label_idxs = list()
        # all_box_targetss = list()
        # for imme_idxs, batch_cls_predicts, batch_reg_predicts in zip(range(len(cls_predicts_stand)), cls_predicts_stand, reg_predicts_stand):
        #     matchess = self.matcher(batch_cls_predicts.detach(), batch_reg_predicts.detach(), gt_boxess, shape_norms)
        #
        #     match_cls_bidxs = sum([[i] * len(j) for i, j, _ in matchess], [])
        #     match_proposal_idxs = sum([j for _, j, _ in matchess], [])
        #     #
        #
        #     #
        #     match_cls_label_idxs = torch.cat([gt_boxess[i][:, 0][k].long() for i, _, k in matchess])
        #     match_boxs = torch.cat([gt_boxess[i][:, 1:][k] for i, _, k in matchess])
        #
        #     all_imme_idxs.append([imme_idxs] * len(match_cls_bidxs))
        #     all_batch_idxs.append(match_cls_bidxs)
        #     all_proposal_idxs.append(match_proposal_idxs)
        #     all_cls_label_idxs.append(match_cls_label_idxs)
        #     all_box_targetss.append(match_boxs)
        #
        # all_imme_idxs = sum(all_imme_idxs, [])
        # all_batch_idxs = sum(all_batch_idxs, [])
        # all_proposal_idxs = sum(all_proposal_idxs, [])
        # all_cls_label_idxs = torch.cat(all_cls_label_idxs)
        # all_box_targetss = torch.cat(all_box_targetss)
        # cls_targetss = torch.zeros_like(cls_predicts_stand)
        # cls_targetss[all_imme_idxs, all_batch_idxs, all_proposal_idxs, all_cls_label_idxs] = 1.0
        # box_pred = reg_predicts_stand[all_imme_idxs, all_batch_idxs, all_proposal_idxs]  # reg_predicts(6,2,128,4)
        # batch_proposal_idx = all_proposal_idxs
        # roi_cs = roi_fea_stand.view(6, 50, 64, 64, 3, 3)#(head的个数,batchsize,proposal个数,in_channel,poolingsize)
        # roi_chooses = roi_cs[all_imme_idxs, all_batch_idxs, all_proposal_idxs]
        # f_stand = roi_chooses.reshape(roi_chooses.shape[0], -1)
        #
        h, w = shape
        shape_norm = torch.tensor([w, h, w, h], device=cls_predicts.device)
        pos_num = len(targets['target'])
        gt_boxes = targets['target'].split(targets['batch_len'])
        if cls_predicts.dtype == torch.float16:
            cls_predicts = cls_predicts.float()
        all_imme_idx = list()
        all_batch_idx = list()
        all_proposal_idx = list()
        all_cls_label_idx = list()
        all_box_targets = list()
        all_imme_idx_neg = list()


        #
        # all_neg_idx = list()
        all_imme_idx_iou = list()
        all_batch_idx_iou = list()
        all_proposal_idx_iou = list()

        all_imme_idx_iou_f = list()
        all_batch_idx_iou_f = list()
        all_proposal_idx_iou_f = list()
        neg_batch = []
        neg_idx = []
        #

        for imme_idx, batch_cls_predict, batch_reg_predict in zip(range(len(cls_predicts)), cls_predicts, reg_predicts):
            matches, indies_all = self.matcher(batch_cls_predict.detach(), batch_reg_predict.detach(), gt_boxes, shape_norm, 1, epoch)

            indies_all_match_z = []
            indies_all_match_f = []
            for k in range(len(indies_all)):
                l = []
                l_z = []
                l.append(k)
                l_z.append(k)
                l_z.append(indies_all[k][1])
                l.append(indies_all[k][0])
                indies_all_match_f.append(l)
                indies_all_match_z.append(l_z)
            match_cls_bidx = sum([[i] * len(j) for i, j, _ in matches], [])
            match_proposal_idx = sum([j for _, j, _ in matches], [])

            match_proposal_idx_i = sum([j for _, j in indies_all_match_z], [])
            match_cls_bidx_i = sum([[i] * len(j) for i, j in indies_all_match_z], [])

            match_proposal_idx_i_f = sum([j for _, j in indies_all_match_f], [])
            match_cls_bidx_i_f = sum([[i] * len(j) for i, j in indies_all_match_f], [])
            #
            # neg_idx = []
            # neg_batch = []


            # num_neg = 0
            # for j in range(len(indies_all_match)):
            #     # num_neg = 0
            #     batchidx = indies_all_match[j][1]
            #     for i in range(64):  # num_proposals的值
            #         if i not in batchidx:
            #             num_neg = num_neg + 1
            #             neg_idx.append(i)#等于all_proposal_idx
            #             neg_batch.append(indies_all_match[j][0])#等于all_batch_idx
            #
            # all_imme_idx_neg.append([imme_idx] * num_neg)



            # neg_idx =
            #
            match_cls_label_idx = torch.cat([gt_boxes[i][:, 0][k].long() for i, _, k in matches])
            match_box = torch.cat([gt_boxes[i][:, 1:][k] for i, _, k in matches])


            all_imme_idx.append([imme_idx] * len(match_cls_bidx))
            all_batch_idx.append(match_cls_bidx)
            all_proposal_idx.append(match_proposal_idx)
            all_cls_label_idx.append(match_cls_label_idx)
            all_box_targets.append(match_box)

            all_proposal_idx_iou.append(match_proposal_idx_i)
            all_batch_idx_iou.append(match_cls_bidx_i)
            all_imme_idx_iou.append([imme_idx] * len(match_cls_bidx_i))

            all_proposal_idx_iou_f.append(match_proposal_idx_i_f)
            all_batch_idx_iou_f.append(match_cls_bidx_i_f)
            all_imme_idx_iou_f.append([imme_idx] * len(match_cls_bidx_i_f))

        all_imme_idx = sum(all_imme_idx, [])
        #
        all_imme_idx_neg = sum(all_imme_idx_iou_f, [])
        neg_batch = sum(all_batch_idx_iou_f, [])
        neg_idx = sum(all_proposal_idx_iou_f, [])
        #
        all_batch_idx = sum(all_batch_idx, [])
        #
        all_imme_idx_iou = sum(all_imme_idx_iou, [])
        all_proposal_idx_iou = sum(all_proposal_idx_iou, [])
        all_batch_idx_iou = sum(all_batch_idx_iou, [])
        # all_batch_idx = sum(all_batch, [])
        #
        all_proposal_idx = sum(all_proposal_idx, [])
        all_cls_label_idx = torch.cat(all_cls_label_idx)
        all_box_targets = torch.cat(all_box_targets)
        cls_targets = torch.zeros_like(cls_predicts)
        cls_targets[all_imme_idx, all_batch_idx, all_proposal_idx, all_cls_label_idx] = 1.0
        #
        box_pred = reg_predicts[all_imme_idx, all_batch_idx, all_proposal_idx]#reg_predicts(6,2,128,4)
        batch_proposal_idx = all_proposal_idx
        # box_preds = reg_predicts[all_imme_idx_iou, all_batch_idx_iou, all_proposal_idx_iou]  # reg_predicts(6,2,128,4)
        # batch_proposal_idxs = all_proposal_idx_iou


        roi_fea_c = roi_fea.view(6, -1, 128, 128, 7, 7)
        roi_fea_choose = roi_fea_c[all_imme_idx_iou, all_batch_idx_iou, all_proposal_idx_iou]
        f = roi_fea_choose.reshape(roi_fea_choose.shape[0], -1)

        #
        roi_fea_neg_choose = roi_fea_c[all_imme_idx_neg, neg_batch, neg_idx]
        f_neg = roi_fea_neg_choose.reshape(roi_fea_neg_choose.shape[0], -1)
        #
        hs, ws = shape_stand
        shape_norms = torch.tensor([ws, hs, ws, hs], device=cls_predicts.device)
        pos_nums = len(targets['target_stand'])
        gt_boxess = targets['target_stand'].split(targets['batch_len_stand'])
        if cls_predicts_stand.dtype == torch.float16:
            cls_predicts_stand = cls_predicts_stand.float()
        all_imme_idxs = list()
        all_batch_idxs = list()
        all_proposal_idxs = list()
        all_cls_label_idxs = list()
        all_box_targetss = list()
        for imme_idxs, batch_cls_predicts, batch_reg_predicts in zip(range(len(cls_predicts_stand)), cls_predicts_stand,
                                                                     reg_predicts_stand):
            matchess = self.matcher(batch_cls_predicts.detach(), batch_reg_predicts.detach(), gt_boxess, shape_norms)

            match_cls_bidxs = sum([[i] * len(j) for i, j, _ in matchess], [])
            match_proposal_idxs = sum([j for _, j, _ in matchess], [])
            #

            #
            match_cls_label_idxs = torch.cat([gt_boxess[i][:, 0][k].long() for i, _, k in matchess])
            match_boxs = torch.cat([gt_boxess[i][:, 1:][k] for i, _, k in matchess])

            all_imme_idxs.append([imme_idxs] * len(match_cls_bidxs))
            all_batch_idxs.append(match_cls_bidxs)
            all_proposal_idxs.append(match_proposal_idxs)
            all_cls_label_idxs.append(match_cls_label_idxs)
            all_box_targetss.append(match_boxs)

        all_imme_idxs = sum(all_imme_idxs, [])
        all_batch_idxs = sum(all_batch_idxs, [])
        all_proposal_idxs = sum(all_proposal_idxs, [])
        all_cls_label_idxs = torch.cat(all_cls_label_idxs)
        all_box_targetss = torch.cat(all_box_targetss)
        cls_targetss = torch.zeros_like(cls_predicts_stand)
        cls_targetss[all_imme_idxs, all_batch_idxs, all_proposal_idxs, all_cls_label_idxs] = 1.0
        # box_pred = reg_predicts_stand[all_imme_idxs, all_batch_idxs, all_proposal_idxs]  # reg_predicts(6,2,128,4)
        batch_proposal_idx = all_proposal_idxs
        roi_cs = roi_fea_stand.view(6, -1, 128, 128, 7, 7)  # (head的个数,batchsize,proposal个数,in_channel,poolingsize)
        roi_chooses = roi_cs[all_imme_idxs, all_batch_idxs, all_proposal_idxs]
        f_stand = roi_chooses.reshape(roi_chooses.shape[0], -1)
        #
        #

        # for i in range(len(all_batch_idx)):
        #     if all_batch_idx[i] == 1:
        #         batch_proposal_idx[i] = all_proposal_idx[i] + 64
        # roi_fea_choose = roi_fea[all_imme_idx, batch_proposal_idx]
        #


        #
        #
        device = cls_predicts.device
        f_num = f.shape[0]
        neg_num = f_neg.shape[0]
        p1f = []
        p0f = []
        psample = []
        pzheng = []
        pfu = []
        sf = []
        for i in range(f_num):
            psample.append(f[i])
            p1f.append(f[i].tolist())
        for i in range(neg_num):
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
        sf_num = f_stand.shape[0]

        for i in range(sf_num):
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
        k1 = 6
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

        ###########
        p0f_array_db = StandardScaler().fit_transform(p0f_array)
        db = DBSCAN(eps=5, min_samples=2).fit(p0f_array_db)

        # 获取聚类标签，-1表示噪声点
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # 初始化一个列表来保存每个簇的中心点
        cluster_centers9 = []

        for i in range(n_clusters):
            # 获取属于当前簇i的所有点的索引
            indices = np.where(labels == i)[0]

            # 计算这些点的平均值作为簇的中心点
            center = np.mean(p0f_array_db[indices], axis=0)

            # 将中心点添加到列表中
            cluster_centers9.append(center)

        # 将结果转换为numpy数组方便操作
        cluster_centers9 = np.array(cluster_centers9)
        cluster_centers1 = cluster_centers9.astype(np.float32)
        cluster_centers_tensor1 = torch.from_numpy(cluster_centers1)
        ########
        # kmeans = KMeans(n_clusters=k1)
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

        # kmeans.fit(p0f_array)
        #
        # # cluster_centers = mbk.cluster_centers_
        # cluster_centers = kmeans.cluster_centers_
        # # cluster_centers_tensor = torch.from_numpy(cluster_centers)
        # # cluster_centers_tensor.to(device)
        #
        # cluster_centers1 = cluster_centers.astype(np.float32)
        # cluster_centers_tensor1 = torch.from_numpy(cluster_centers1)
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
        loss = InfoNCE(temperature=0.05, negative_mode='unpaired')  # negative_mode='unpaired' is the default value
        # batch_size, num_negative, embedding_size = pred_num, k, d
        query = psample_tensor
        # query = torch.randn(batch_size, embedding_size)
        # positive_key = pz_array_tensor
        positive_key = pzhengcentors_tensor
        negative_keys = cluster_centers_tensor1
        output = loss(query.to(device), positive_key.to(device), negative_keys.to(device))
        output = output*0.01
        outlossall = output
#

        # pf = []
        # tpf = []
        # num_boxes = f.shape[0]
        # for q in range(num_boxes):
        #     pf.append(f[q].tolist())
        #     tpf.append(f[q].tolist())
        # pf_array = np.array(pf, dtype='float16')
        # sf = []
        # s_num = f_stand.shape[0]
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
        # cluster_centers_tensor.to(cls_predicts.device)
        #
        # cluster_centers1 = cluster_centers.astype(np.float32)
        # cluster_centers_tensor1 = torch.from_numpy(cluster_centers1)
        # cluster_centers_tensor1.to(cls_predicts.device)
        #
        # cluser_labels = kmeans.predict(tpf_array)
        # s_cluslabel = []  # target的聚类label
        # p_cluslabel = []
        #
        # for ia in range(num_boxes):
        #     p_cluslabel.append(cluser_labels[ia])
        #
        # device = cls_predicts.device
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
        # output_loss = loss(query.to(cls_predicts.device), positive_key.to(cls_predicts.device), negative_keys.to(cls_predicts.device))
        # # outlossall.append(output_loss)
        # outlossall = outlossall + output_loss
        # outlossall = outlossall * 2
        #
        cls_loss = self.cls_weights * focal_loss(cls_predicts.sigmoid(), cls_targets).sum()
        box_loss = self.iou_weights * self.iou_loss(box_pred, all_box_targets).sum()
        l1_loss = self.l1_weights * torch.nn.functional.l1_loss(box_pred / shape_norm[None, :],
                                                                all_box_targets / shape_norm[None, :],
                                                                reduction="none").sum()
        pos_num = reduce_sum(torch.tensor(pos_num, device=cls_predicts.device)).item() / get_gpu_num_solo()

        return cls_loss / pos_num, box_loss / pos_num, l1_loss / pos_num, pos_num, outlossall