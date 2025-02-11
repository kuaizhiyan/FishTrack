# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData

try:
    import motmetrics
    from motmetrics.lap import linear_sum_assignment
except ImportError:
    motmetrics = None
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_overlaps, bbox_xyxy_to_cxcyah
from mmdet.utils import OptConfigType
from ..utils import imrenormalize
from .base_tracker import BaseTracker
from .sort_tracker import SORTTracker


@MODELS.register_module()
class MMSORTTracker(SORTTracker):
    def __init__(self,
                 motion: Optional[dict] = None,
                 obj_score_thr: float = 0.3,
                 reid: dict = dict(
                     num_samples=10,
                     img_scale=(256, 128),
                     img_norm_cfg=None,
                     match_score_thr=2.0),
                 match_iou_thr: float = 0.7,
                 num_tentatives: int = 3,
                 text_factor: float = 0.0001,
                 **kwargs):
        if motmetrics is None:
            raise RuntimeError('motmetrics is not installed,\
                 please install it by: pip install motmetrics')
        
        # ✅ 在 kwargs 里移除 text_factor，防止传递给 SORTTracker
        kwargs.pop('text_factor', None)
            
        # ✅ 这里显式传递参数给 SORTTracker
        super().__init__(
            motion=motion,
            obj_score_thr=obj_score_thr,
            reid=reid,
            match_iou_thr=match_iou_thr,
            num_tentatives=num_tentatives,
            **kwargs
        )

        self.text_factor = text_factor  # ✅ 这样才不会丢失参数
    

    def compute_text_embedding_distance(self,
                                    track_labels: torch.Tensor, 
                                    labels: torch.Tensor, 
                                    label_embedding_dict: dict) -> torch.Tensor:
        """
        计算当前帧目标标签 `track_labels` 和上一帧目标标签 `labels` 之间的文本嵌入余弦距离，并整体 +1。

        Args:
            labels (torch.Tensor): 当前帧目标的 label，形状为 (N,)。
            track_labels (torch.Tensor): 上一帧目标的 label，形状为 (M, )。
            label_embedding_dict (dict): label 到 text embedding 的映射，{int: torch.Tensor}，每个 embedding 形状为 (D,)。

        Returns:
            torch.Tensor: 计算得到的余弦距离矩阵，形状为 (N, M)，整体加 1 使其 > 0。
        """
        if track_labels.numel() == 0 or labels.numel() == 0:
            return torch.empty((len(track_labels), len(labels)))
        # 获取上一帧的 text embedding
        track_embeddings = torch.stack([label_embedding_dict[int(label)] for label in track_labels], dim=0)  # (N, D)
        # 获取当前帧的 text embedding
        label_embeddings = torch.stack([label_embedding_dict[int(label)] for label in labels], dim=0)  # (M, D)
        
        # 归一化 embedding，计算余弦相似度
        track_embeddings = F.normalize(track_embeddings, p=2, dim=-1)  # (N, D)
        label_embeddings = F.normalize(label_embeddings, p=2, dim=-1)  # (M, D)

        # 计算余弦距离
        cosine_distance = 1 - torch.mm(track_embeddings, label_embeddings.T)  # (N, M)

        # 整体加 1，确保值 > 0
        return cosine_distance + 1


    
    def track(self,
              model: torch.nn.Module,                       # 只用到， model.reid
              img: Tensor,                                  #  'The img must be 5D Tensor (N, T, C, H, W).'  # [1,1,3,640,1088]，只是tensor 之后的
              data_sample: DetDataSample,                   # ok 
              data_preprocessor: OptConfigType = None,      # 只使用到归一化参数，需要修改
              rescale: bool = False,
              label_embedding_dict: dict = None,
              **kwargs) -> InstanceData:
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                SORT method.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.
            data_preprocessor (dict or ConfigDict, optional): The pre-process
               config of :class:`TrackDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """
        metainfo = data_sample.metainfo
        bboxes = data_sample.pred_instances.bboxes      # data_sample 保存了目标检测结果
        labels = data_sample.pred_instances.labels
        scores = data_sample.pred_instances.scores

        frame_id = metainfo.get('frame_id', -1)
        if frame_id == 0:
            self.reset()
        if not hasattr(self, 'kf'):
            self.kf = self.motion

        if self.with_reid:
            if self.reid.get('img_norm_cfg', False):
                img_norm_cfg = dict(
                    mean=data_preprocessor['mean'],
                    std=data_preprocessor['std'],
                    to_bgr=data_preprocessor['rgb_to_bgr'])
                reid_img = imrenormalize(img, img_norm_cfg,
                                         self.reid['img_norm_cfg'])     # 再一次 norm
            else:
                reid_img = img.clone()

        valid_inds = scores > self.obj_score_thr    # 筛选出大于阈值的 box ，关键在于开放集的 labels 处如何处理
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]                 # tensor([100])
        scores = scores[valid_inds]

        # tracker 初始化
        if self.empty or bboxes.size(0) == 0:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long).to(bboxes.device)     # 沿着全局 track id 生成新的 id
            self.num_tracks += num_new_tracks
            if self.with_reid:
                crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),      # e.g. [100,3,256,128]
                                       rescale)
                if crops.size(0) > 0:
                    embeds = model.reid(crops, mode='tensor')
                else:
                    embeds = crops.new_zeros((0, model.reid.head.out_channels))     # 生成 0 向量
        else:
            ids = torch.full((bboxes.size(0), ), -1,
                             dtype=torch.long).to(bboxes.device)            # 保存 id 分配结果

            # motion
            self.tracks, costs = self.motion.track(self.tracks,
                                                   bbox_xyxy_to_cxcyah(bboxes))     # 对已有 box 预测位置

            active_ids = self.confirmed_ids
            if self.with_reid:
                crops = self.crop_imgs(reid_img, metainfo, bboxes.clone(),
                                       rescale)
                embeds = model.reid(crops, mode='tensor')       # 计算新的  reid emb 为什么带有梯度? deepsort 是没有的， DeepSORT.training=False

                # reid
                if len(active_ids) > 0:
                    track_embeds = self.get(
                        'embeds',
                        active_ids,
                        self.reid.get('num_samples', None),
                        behavior='mean')
                    reid_dists = torch.cdist(track_embeds, embeds)  # 新旧 emb 计算 cosine 距离

                    # support multi-class association
                    track_labels = torch.tensor([
                        self.tracks[id]['labels'][-1] for id in active_ids
                    ]).to(bboxes.device)
                    cate_match = labels[None, :] == track_labels[:, None]
                    cate_cost = (1 - cate_match.int()) * 1e6            # 支撑多类别匹配，所以分类损失是 0
                    
                    # 计算标签距离
                    label_dists = 0
                    if label_embedding_dict is not None:
                        label_dists = self.compute_text_embedding_distance(track_labels=track_labels,
                                                                           labels=labels,
                                                                           label_embedding_dict=label_embedding_dict
                                                                            )
                        
                    reid_dists = (reid_dists + cate_cost + label_dists*self.text_factor).cpu().numpy()

                    valid_inds = [list(self.ids).index(_) for _ in active_ids]
                    reid_dists[~np.isfinite(costs[valid_inds, :])] = np.nan

                    row, col = linear_sum_assignment(reid_dists)
                    for r, c in zip(row, col):   # hungiran match
                        dist = reid_dists[r, c]
                        if not np.isfinite(dist):
                            continue
                        if dist <= self.reid['match_score_thr']:
                            ids[c] = active_ids[r]  # 匹配成功，赋值 ids
                            
            # cascade match ，reid 匹配失败（没有在 ids 中），则采用 iou 距离二次匹配
            active_ids = [
                id for id in self.ids if id not in ids
                and self.tracks[id].frame_ids[-1] == frame_id - 1
            ]
            if len(active_ids) > 0:
                active_dets = torch.nonzero(ids == -1).squeeze(1)
                track_bboxes = self.get('bboxes', active_ids)
                ious = bbox_overlaps(track_bboxes, bboxes[active_dets])         # 计算 iou

                # support multi-class association   
                track_labels = torch.tensor([
                    self.tracks[id]['labels'][-1] for id in active_ids
                ]).to(bboxes.device)
                cate_match = labels[None, active_dets] == track_labels[:, None]     # 为了满足多标签匹配，进行升维
                cate_cost = (1 - cate_match.int()) * 1e6        # 分类 cost，都是 0，忽略了类别影响 [100,100]

                label_dists = 0
                if label_embedding_dict is not None:
                    label_dists = self.compute_text_embedding_distance(track_labels=track_labels,
                                                                        labels=labels[active_dets],
                                                                        label_embedding_dict=label_embedding_dict
                                                                        )
                dists = (1 - ious + cate_cost + label_dists*self.text_factor).cpu().numpy()    # (100,100)

                row, col = linear_sum_assignment(dists)         # (100,) (100,)
                for r, c in zip(row, col):
                    dist = dists[r, c]
                    if dist < 1 - self.match_iou_thr:
                        ids[active_dets[c]] = active_ids[r]     # 更新 id 分配结果. 根据 r,c 到 activate_id 取出对应的 id

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            embeds=embeds if self.with_reid else None,
            frame_ids=frame_id)

        # update pred_track_instances
        pred_track_instances = InstanceData()
        pred_track_instances.bboxes = bboxes
        pred_track_instances.labels = labels
        pred_track_instances.scores = scores
        pred_track_instances.instances_id = ids

        return pred_track_instances     # 返回跟踪成功的 预测结果
