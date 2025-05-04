import os
import yaml
import torch
import torch.distributed as dist
from tqdm import tqdm

import cv2 as cv
import time
import datetime
from torch import nn
import numpy as np
import json
from torch.cuda import amp
from torch.utils.data.distributed import DistributedSampler
from datasets.coco import COCODataSets
from nets.sparse_rcnn import SparseRCNN
from torch.utils.data.dataloader import DataLoader
from utils.model_utils import rand_seed, ModelEMA, AverageLogger, reduce_sum
from metrics.map import coco_map
from utils.optims_utils import IterWarmUpMultiStepDecay, split_optimizer_v2
from utils.augmentations import RandScaleToMax
from datasets.coco import coco_ids, rgb_mean, rgb_std
from eval import coco_eval

rand_seed(1024)




class DDPMixSolver(object):
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as rf:
            self.cfg = yaml.safe_load(rf)
        self.data_cfg = self.cfg['data']
        self.model_cfg = self.cfg['model']
        self.optim_cfg = self.cfg['optim']
        self.val_cfg = self.cfg['val']
        print(self.data_cfg)
        print(self.model_cfg)
        print(self.optim_cfg)
        print(self.val_cfg)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.cfg['gpus']
        self.gpu_num = len(self.cfg['gpus'].split(','))
        dist.init_process_group(backend='NCCL')


        self.tdata = COCODataSets(img_root=self.data_cfg['train_img_root'],
                                  annotation_path=self.data_cfg['train_annotation_path'],
                                  max_thresh=self.data_cfg['max_thresh'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=True,
                                  remove_blank=self.data_cfg['remove_blank']
                                  )
        self.tloader = DataLoader(dataset=self.tdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.tdata.collect_fn,
                                  sampler=DistributedSampler(dataset=self.tdata, shuffle=True))
        self.vdata = COCODataSets(img_root=self.data_cfg['val_img_root'],
                                  annotation_path=self.data_cfg['val_annotation_path'],
                                  max_thresh=self.data_cfg['max_thresh'],
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=False,
                                  remove_blank=False
                                  )
        self.vloader = DataLoader(dataset=self.vdata,
                                  batch_size=self.data_cfg['batch_size'],
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.vdata.collect_fn,
                                  sampler=DistributedSampler(dataset=self.vdata, shuffle=False))

        self.sdata = COCODataSets(img_root=self.data_cfg['stand_img_root'],
                                  annotation_path=self.data_cfg['stand_annotation_path'],
                                  max_thresh=100,
                                  debug=self.data_cfg['debug'],
                                  use_crowd=self.data_cfg['use_crowd'],
                                  augments=False,
                                  remove_blank=False
                                  )

        self.sloader = DataLoader(dataset=self.sdata,
                                  batch_size=10,
                                  num_workers=self.data_cfg['num_workers'],
                                  collate_fn=self.vdata.collect_fn,
                                  sampler=DistributedSampler(dataset=self.sdata, shuffle=False))


        print("train_data: ", len(self.tdata), " | ",
              "val_data: ", len(self.vdata), " | ",
              "satnd_data:", len(self.sdata), " | ",
              "empty_data: ", self.tdata.empty_images_len)
        print("train_iter: ", len(self.tloader), " | ",
              "val_iter: ", len(self.vloader))
        model = SparseRCNN(**self.model_cfg)
        # modelval = SparseRCNNval(**self.model_cfg)
        self.best_map = 0.
        optimizer = split_optimizer_v2(model, self.optim_cfg)
        # optimizerval = split_optimizer_v2(modelval, self.optim_cfg)
        local_rank = dist.get_rank()
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        model.to(self.device)
        # modelval.to(self.device)
        if self.optim_cfg['sync_bn']:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            # modelval = nn.SyncBatchNorm.convert_sync_batchnorm(modelval)
        self.model = nn.parallel.distributed.DistributedDataParallel(model,
                                                                     device_ids=[local_rank],
                                                                     find_unused_parameters=True,
                                                                     output_device=local_rank)
        # self.modelval = nn.parallel.distributed.DistributedDataParallel(modelval,
        #                                                              device_ids=[local_rank],
        #                                                              output_device=local_rank)
        self.scaler = amp.GradScaler(enabled=True) if self.optim_cfg['amp'] else None
        self.optimizer = optimizer
        self.ema = ModelEMA(self.model)
        self.lr_adjuster = IterWarmUpMultiStepDecay(init_lr=self.optim_cfg['lr'],
                                                    milestones=self.optim_cfg['milestones'],
                                                    warm_up_iter=self.optim_cfg['warm_up_iter'],
                                                    iter_per_epoch=len(self.tloader),
                                                    epochs=self.optim_cfg['epochs'],
                                                    alpha=self.optim_cfg['alpha'],
                                                    warm_up_factor=self.optim_cfg['warm_up_factor']
                                                    )
        self.cls_loss_logger = AverageLogger()
        self.l1_loss_logger = AverageLogger()
        self.iou_loss_logger = AverageLogger()
        self.match_num_logger = AverageLogger()
        self.loss_logger = AverageLogger()
        # if self.local_rank == 0:
        #     print(self.model)

        '''
        checkpoint_path = 'E:\\sparse_rcnnv1-master\\sparse_rcnnv1-master\\checkpoints\\checkpoint_epoch_10.pth'

        checkpoint_path = '/home/taoye/code/sparse_rcnnv1-master/checkpoints/checkpoint_epoch_11.pth'

        # 加载检查点
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 恢复优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #
        # # 恢复其他相关信息（如epoch、损失等）
        # start_epoch = checkpoint['epoch'] + 1
        # last_loss = checkpoint['loss']
        # last_cls_loss = checkpoint['cls_loss']
        # last_l1_loss = checkpoint['l1_loss']
        # last_iou_loss = checkpoint['iou_loss']
        # last_match_num = checkpoint['match_num']
        # last_lr = checkpoint['lr']

        # 如果有EMA，恢复EMA状态
        if 'ema_state_dict' in checkpoint and hasattr(model, 'ema'):
            self.model.ema.load_state_dict(checkpoint['ema_state_dict'])
        '''

    def train(self, epoch):
        self.loss_logger.reset()
        self.cls_loss_logger.reset()
        self.l1_loss_logger.reset()
        self.iou_loss_logger.reset()
        self.match_num_logger.reset()
        self.model.train()
        if self.local_rank == 0:#True
            pbar = tqdm(self.tloader)

        else:
            pbar = self.tloader
        sbar = tqdm(self.sloader)
        for i, (img_tensor_stand, targets_tensor_stand, batch_len_stand) in enumerate(sbar):
            img_tensor_stand = img_tensor_stand.to(self.device)
            targets_tensor_stand = targets_tensor_stand.to(self.device)
            break

        self.img_tensor_stand = img_tensor_stand
        for i, (img_tensor, targets_tensor, batch_len) in enumerate(pbar):
            _, _, h, w = img_tensor.shape
            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                targets_tensor = targets_tensor.to(self.device)
            self.optimizer.zero_grad()
            if self.scaler is not None:
                with amp.autocast(enabled=True):
                    # out1 = self.model(img_tensor_stand, targets={"target": targets_tensor_stand, "batch_len":batch_len_stand})
                    out = self.model(img_tensor,img_tensor_stand,
                                     targets={"target": targets_tensor, "batch_len": batch_len, "target_stand": targets_tensor_stand, "batch_len_stand": batch_len_stand}, epoch=epoch)
                    cls_loss = out['cls_loss']
                    l1_loss = out['l1_loss']
                    iou_loss = out['iou_loss']
                    match_num = out['match_num']
                    loss = cls_loss + l1_loss + iou_loss
                    self.scaler.scale(loss).backward()
                    self.lr_adjuster(self.optimizer, i, epoch)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:#执行这个
                out = self.model(img_tensor, img_tensor_stand,
                                 targets={"target": targets_tensor, "batch_len": batch_len,
                                          "target_stand": targets_tensor_stand, "batch_len_stand": batch_len_stand}, epoch=epoch)
                cls_loss = out['cls_loss']
                l1_loss = out['l1_loss']
                iou_loss = out['iou_loss']
                match_num = out['match_num']
                clus_loss = out['clus_loss']
                loss = cls_loss + l1_loss + iou_loss + clus_loss
                loss.backward()
                self.lr_adjuster(self.optimizer, i, epoch)
                self.optimizer.step()
            self.ema.update(self.model)
            lr = self.optimizer.param_groups[0]['lr']
            self.loss_logger.update(loss.item())
            self.iou_loss_logger.update(iou_loss.item())
            self.l1_loss_logger.update(l1_loss.item())
            self.cls_loss_logger.update(cls_loss.item())
            self.match_num_logger.update(match_num)
            str_template = \
                "epoch:{:2d}|match_num:{:0>4d}|size:{:3d}|loss:{:6.4f}|cls:{:6.4f}|l1:{:6.4f}|iou:{:6.4f}|lr:{:8.6f}"
            if self.local_rank == 0:
                # torch.save(self.model.state_dict(), "/home/taoye/code/sparse_rcnnv1-master/xxsmall0.015 0.1  kmeans tar1/sparse_rcnn-model-{}.pth.tar".format(epoch), _use_new_zipfile_serialization=False)
                #######
                # torch.save(self.model.state_dict(), "/home/taoye/code/sparse_rcnnv1-master/xxsmall0.015 0.1  kmeans10 swint0.05/resNetFpn-model-{}.pth".format(epoch))
                pbar.set_description(str_template.format(
                    epoch + 1,
                    int(match_num),
                    h,
                    self.loss_logger.avg(),
                    self.cls_loss_logger.avg(),
                    self.l1_loss_logger.avg(),
                    self.iou_loss_logger.avg(),
                    lr)
                )
                

        '''
        checkpoint_dir = 'xxsmall0.015 0.1  kmeans'
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        '''
                #

        self.ema.update_attr(self.model)

        
        # if args.amp:
        #     save_files["scaler"] = scaler.state_dict()
        
        # torch.save(self.model.state_dict(), "/home/taoye/code/sparse_rcnnv1-master/solver/saveiou/resNetFpn-model-{}.pth".format(epoch))
        # torch.save(self.model.state_dict(), "/home/taoye/code/sparse_rcnnv1-master/xxsmall0.015 0.1  kmeans tar/sparse_rcnn-model-{}.pth.tar".format(epoch), _use_new_zipfile_serialization=False)

        loss_avg = reduce_sum(torch.tensor(self.loss_logger.avg(), device=self.device)) / self.gpu_num
        iou_loss_avg = reduce_sum(torch.tensor(self.iou_loss_logger.avg(), device=self.device)).item() / self.gpu_num
        l1_loss_avg = reduce_sum(torch.tensor(self.l1_loss_logger.avg(), device=self.device)).item() / self.gpu_num
        cls_loss_avg = reduce_sum(torch.tensor(self.cls_loss_logger.avg(), device=self.device)).item() / self.gpu_num
        match_num_sum = reduce_sum(torch.tensor(self.match_num_logger.sum(), device=self.device)).item() / self.gpu_num
        if self.local_rank == 0:
            final_template = "epoch:{:2d}|match_num:{:d}|loss:{:6.4f}|cls:{:6.4f}|l1:{:6.4f}|iou:{:6.4f}"
            print(final_template.format(
                epoch + 1,
                int(match_num_sum),
                loss_avg,
                cls_loss_avg,
                l1_loss_avg,
                iou_loss_avg
            ))
    @torch.no_grad()
    def eeval(self):
        from pycocotools.coco import COCO
        device = self.device
        self.model.eval()
        self.ema.ema.eval()
        with open("config/sparse_rcnn.yaml", 'r') as rf:
            cfg = yaml.safe_load(rf)
        data_cfg = cfg['data']
        basic_transform = RandScaleToMax(max_threshes=[data_cfg['max_thresh']])
        coco = COCO(data_cfg['val_annotation_path'])
        coco_predict_list = list()
        time_logger = AverageLogger()
        pbar = tqdm(coco.imgs.keys())
        for img_id in pbar:
            file_name = coco.imgs[img_id]['file_name']
            img_path = os.path.join(data_cfg['val_img_root'], file_name)
            img = cv.imread(img_path)
            h, w, _ = img.shape
            img, ratio, (left, top) = basic_transform.make_border(img,
                                                                  max_thresh=data_cfg['max_thresh'],
                                                                  border_val=(103, 116, 123))
            img_inp = (img[:, :, ::-1] / 255.0 - np.array(rgb_mean)) / np.array(rgb_std)
            img_inp = torch.from_numpy(img_inp).unsqueeze(0).permute(0, 3, 1, 2).contiguous().float().to(device)
            tic = time.time()
            predict = self.ema.ema(img_inp, self.img_tensor_stand)["predicts"][0]
            duration = time.time() - tic
            time_logger.update(duration)
            pbar.set_description("fps:{:4.2f}".format(1 / time_logger.avg()))
            if predict is None:
                continue
            predict[:, [0, 2]] = ((predict[:, [0, 2]] - left) / ratio).clamp(min=0, max=w)
            predict[:, [1, 3]] = ((predict[:, [1, 3]] - top) / ratio).clamp(min=0, max=h)
            box = predict.cpu().numpy()
            coco_box = box[:, :4]
            coco_box[:, 2:] = coco_box[:, 2:] - coco_box[:, :2]
            for p, b in zip(box.tolist(), coco_box.tolist()):
                coco_predict_list.append({'image_id': img_id,
                                          'category_id': coco_ids[int(p[5])],
                                          'bbox': [round(x, 3) for x in b],
                                          'score': round(p[4], 5)})
        if self.local_rank != 0:
            return
        with open("predicts.json", 'w') as file:
            json.dump(coco_predict_list, file)
        coco_eval(anno_path=data_cfg['val_annotation_path'], pred_path="predicts.json")

    @torch.no_grad()
    def val(self, epoch):
        # sbar = tqdm(self.sloader)
        # for i, (img_tensor_stand, targets_tensor_stand, batch_len_stand) in enumerate(sbar):
        #     img_tensor_stand = img_tensor_stand.to(self.device)
        #     targets_tensor_stand = targets_tensor_stand.to(self.device)
        #     break
        predict_list = list()
        target_list = list()
        self.model.eval()
        self.ema.ema.eval()
        if self.local_rank == 0:
            pbar = tqdm(self.vloader)
        else:
            pbar = self.vloader
        for img_tensor, targets_tensor, batch_len in pbar:
            img_tensor = img_tensor.to(self.device)
            targets_tensor = targets_tensor.to(self.device)
            predicts = self.ema.ema(img_tensor, self.img_tensor_stand)['predicts']
            for pred, target in zip(predicts, targets_tensor.split(batch_len)):
                predict_list.append(pred)
                target_list.append(target)
        mp, mr, map50, mean_ap = coco_map(predict_list, target_list)
        mp = reduce_sum(torch.tensor(mp, device=self.device)) / self.gpu_num
        mr = reduce_sum(torch.tensor(mr, device=self.device)) / self.gpu_num
        map50 = reduce_sum(torch.tensor(map50, device=self.device)) / self.gpu_num
        mean_ap = reduce_sum(torch.tensor(mean_ap, device=self.device)) / self.gpu_num
        print("---",map50,"---")

        if self.local_rank == 0:
            print("*" * 20, "eval start", "*" * 20)
            print("epoch: {:2d}|mp:{:6.4f}|mr:{:6.4f}|map50:{:6.4f}|map:{:6.4f}"
                  .format(epoch + 1,
                          mp * 100,
                          mr * 100,
                          map50 * 100,
                          mean_ap * 100))
            print("*" * 20, "eval end", "*" * 20)
        last_weight_path = os.path.join(self.val_cfg['weight_path'],
                                        "{:s}_{:s}_last.pth"
                                        .format(self.cfg['model_name'],
                                                self.model_cfg['backbone']))
        best_map_weight_path = os.path.join(self.val_cfg['weight_path'],
                                            "{:s}_{:s}_best_map.pth"
                                            .format(self.cfg['model_name'],
                                                    self.model_cfg['backbone']))
        model_static = self.model.module.state_dict()
        cpkt = {
            "model": model_static,
            "map": mean_ap * 100,
            "epoch": epoch,
            "ema": self.ema.ema.state_dict()
        }
        if self.local_rank != 0:
            return
        torch.save(cpkt, last_weight_path)
        if map50 > self.best_map:
            torch.save(cpkt, best_map_weight_path)
            self.best_map = mean_ap

    def run(self):
        start_time = time.time()
        for epoch in range(self.optim_cfg['epochs']):
            self.train(epoch)
            if (epoch + 1) % self.val_cfg['interval'] == 0:
                self.eeval()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
