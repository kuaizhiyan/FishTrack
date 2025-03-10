import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image

# mmdet
try:
    import mmcv
    import mmdet
    from mmdet.apis import inference_detector, init_detector
    from mmdet.models.trackers import ByteTracker
    from mmdet.models.trackers import SORTTracker
    from mmdet.structures import DetDataSample
    from mmdet.visualization.local_visualizer import TrackLocalVisualizer
    from mmengine.config import Config
    from mmengine.structures import InstanceData
    from mmdet.registry import MODELS
except ImportError:
    mmdet = None

# groudingdino
# import mmdet.datasets.transforms as T

import groundingdino.datasets.transforms as T
sort_transform = T.Compose([
    T.ToTensor(),
])

import sys

img_scale=(640,640)
# detector_transform=T.Compose([        # yolox config 复制过来的
#     T.ToTensor(),
#     T.Resize(img_scale,keep_ratio=True),
#     T.Pad(pad_to_square=True,pad_val=dict(img=(114.0, 114.0, 114.0)))
# ])


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def parse_args():
    parser = argparse.ArgumentParser('Open Tracking Demo', add_help=True)
    parser.add_argument('inputs', type=str, help='path to video or image dirs')
    parser.add_argument('det_config', type=str, help='path to det config file')     # Important!  目标检测器的配置
    parser.add_argument('det_weight', type=str, help='path to det weight file')

    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--out-dir',
        '-o',
        type=str,
        default='outputs',
        help='output directory')
    parser.add_argument(
        '--box-thr', '-b', type=float, default=0.05, help='box threshold')
    parser.add_argument(
        '--det-device',
        '-d',
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument('--cpu-off-load', '-c', action='store_true')
    parser.add_argument('--tracker-path', help='deep sort config file')

    # track params
    # you can modify tracker score to fit your task
    # use glip, in bdd demo: use init
    # init_track_thr 0.65 and obj_score_thrs_high 0.6
    parser.add_argument(
        '--init_track_thr', type=float, default=0.45, help='init track')
    parser.add_argument(
        '--obj_score_thrs_high',
        type=float,
        default=0.4,
        help='first association threshold')
    parser.add_argument(
        '--obj_score_thrs_low',
        type=float,
        default=0.1,
        help='second association threshold')
    parser.add_argument(
        '--num_frames_retain',
        type=int,
        default=30,
        help='remove lost tracklet more than num frames')

    # video params
    parser.add_argument('--fps', type=int, default=30, help='video fps')
    parser.add_argument(
        '--out', type=str, default='demo.mp4', help='output video name')
    return parser.parse_args()


# 创建 目标检测器
def build_detector_model(args):
    config = Config.fromfile(args.det_config)
    if 'init_cfg' in config.model.backbone: # 清除 backbone 的权重
        config.model.backbone.init_cfg = None
    detecter = init_detector(
        config, args.det_weight, device='cpu', cfg_options={},palette='coco')
    return detecter


def run_detector(model, image_new, args, label_name=None):
    """
    image_new: PIL.Image.Image image (1280,720)
    label_name: list(8):
    """
    
    if args.cpu_off_load:
        model = model.to(args.det_device)

    if 'GroundingDINO' in args.det_config:      # Disable

        image, _ = grounding_dino_transform(image_new, None)  # 3, h, w (1280,720,3)-> [3,750,1333]  resize+norm
       
        image = image.to(next(model.parameters()).device)
        with torch.no_grad():
            outputs = model(image[None], captions=[args.text_prompt])       # pred_logits:[1,900,256],pred_boxes:[1,900,4]

        logits = outputs['pred_logits'].cpu().sigmoid()[0]  # (nq, 256) 降了一维        logits 是什么东西？
        boxes = outputs['pred_boxes'].cpu()[0]  # (nq, 4)

        logits = convert_grounding_to_od_logits(
            logits, len(label_name),
            positive_map_label_to_token)  # [N, num_classes] 【900，8】

        # filter output
        logits_filt = logits.clone()        # 【900，8】
        boxes_filt = boxes.clone()          # 【9，4】
        filt_mask = logits_filt.max(dim=1)[0] > args.box_thr        # 0.05
        logits_filt = logits_filt[filt_mask]  # num_filt, 256  [112,8]
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4      [112,4]
        # 将概率最大的类别取出
        scores, pred_phrase_idx = logits_filt.max(1)  # 112， 112，
        # 将 box 转换到原图尺寸
        size = image_new.size
        boxes_filt = boxes_filt * torch.tensor(
            [size[0], size[1], size[0], size[1]]).repeat(len(boxes_filt), 1)
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]
        # 构造标准输出
        pred_instances = InstanceData()
        pred_instances.bboxes = boxes_filt      # 原图尺寸 box
        pred_instances.labels = pred_phrase_idx # 类别标签
        pred_instances.scores = scores          # 概率
    else:
        # transform
        # image, _ = detector_transform(image_new, None)
        
        result = inference_detector(model,image_new)
        
        # inference_detector 已经转换为原尺寸了 ps:[[ 221.7274,  723.7135, 1284.3472, 1069.9478]
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.box_thr]
        # # 构造标准输出
        # pred_instances = InstanceData()
        # pred_instances.bboxes = boxes_filt      # 原图尺寸 box
        # pred_instances.labels = pred_phrase_idx # 类别标签
        # pred_instances.scores = scores          # 概率

    if args.cpu_off_load:
        model = model.to('cpu')

    return pred_instances


def main():
    if mmdet is None:
        raise RuntimeError('mmdet is not installed,\
                 please install it follow README')
    args = parse_args()
    
    if args.cpu_off_load is True:
        if 'cpu' in args.det_device and 'cpu ' in args.sam_device:
            raise RuntimeError(
                'args.cpu_off_load is an invalid parameter due to '
                'detection and mask model IS on the cpu.')

 # define output
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # define input
    if osp.isdir(args.inputs):
        imgs = sorted(
            filter(lambda x: x.endswith(IMG_EXTENSIONS),
                   os.listdir(args.inputs)),
            key=lambda x: x.split('.')[0])
        in_video = False
    else:
        imgs = []
        cap = cv2.VideoCapture(args.inputs)
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            imgs.append(frame)
        in_video = True

    # define fs
    fps = args.fps
    if args.show:
        if fps is None and in_video:
            fps = video_fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)
        
    # visulization
    label_name = 'fish'
    visualizer = TrackLocalVisualizer()
    visualizer.dataset_meta = {'classes': label_name}

    # det model
    det_model = build_detector_model(args)

    # deepsort model
    cfg = Config.fromfile(args.tracker_path)
    sort_model = MODELS.build(cfg.model)
    sort_model.eval()
    tracker = sort_model.tracker
    
    if not args.cpu_off_load:
        det_model = det_model.to(args.det_device)
    
    # 保存跟踪结果
    track_results = []

    for frame_id, img in enumerate(imgs):
        save_path = os.path.join(args.out_dir, f'{frame_id:06d}.jpg')

        if isinstance(img, str):
            image_path = osp.join(args.inputs, img)                 # 
            image_new = cv2.imread(image_path)
            image_copy = Image.open(image_path).convert('RGB') 
        # print('image_new type:',type(image_new))
        # print('image_copy type:',type(image_copy))


        pred_instances = run_detector(det_model, image_new, args, label_name) # {boxes[325,4]原图尺寸,labels[325],scores[325]} ,给出的还是对应的下标，只不过标签可以手动输入
        # print('image_new type:',type(image_new))
        
        # track input
        img_data_sample = DetDataSample()                           # 构造 DetDataSample()
        img_data_sample.pred_instances = pred_instances
        # print('image_copy type:',type(image_copy))
        img_data_sample.set_metainfo(dict(frame_id=frame_id,img_shape=(image_copy.height,image_copy.width)))       # 所以，只要使用检测器给出检测结果，送到 Tracker 里面就好了

        # track
        img_track,_ = sort_transform(image_new, None)
        img_track = img_track.unsqueeze(0)                                #  'The img must be 5D Tensor (N, T, C, H, W).'  # [1,1,3,640,1088]，只是tensor 之后的
        data_preprocessor=dict(
                                type='TrackDataPreprocessor',
                                mean=[123.675, 116.28, 103.53],
                                std=[58.395, 57.12, 57.375],
                                bgr_to_rgb=True,
                                pad_size_divisor=32)
        with torch.no_grad():
            pred_track_instances = tracker.track(
                                                model=sort_model,
                                                img=img_track,
                                                data_sample=img_data_sample,
                                                data_preprocessor=data_preprocessor,
                                                # rescale=False
                                                )
        img_data_sample.pred_track_instances = pred_track_instances

        vis_image = image_new[..., ::-1]

        visualizer.add_datasample(
            'mot',
            vis_image,
            data_sample=img_data_sample,        # pred_track_instance
            show=True,
            # args.show,
            draw_gt=False,
            out_file=save_path,
            wait_time=float(1 / int(fps)) if fps else 0,
            pred_score_thr=0.0,
            step=frame_id)

        # 本地写入逻辑      
        instances_id = pred_track_instances.instances_id.cpu().numpy()  # [n,1]
        labels = pred_track_instances.labels.cpu().numpy()  # [n,1]
        bboxes = pred_track_instances.bboxes.cpu().numpy()  # [n,4]
        scores = pred_track_instances.scores.cpu().numpy()  # [n,1]

        # 遍历当前帧的所有目标，存入 track_results
        for obj_id, label, bbox, conf in zip(instances_id, labels, bboxes, scores):
            x1, y1, x2, y2 = bbox  # 解包 bbox
            w = x2 - x1
            h = y2 - y1
            # conf = 1.0  # 如果有置信度信息，可以替换此值
            track_results.append(f"{frame_id+1},{obj_id+1},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{conf:.3f},-1,-1,-1") # 注意 frame / id 是否从 1 开始 


    mmcv.frames2video(args.out_dir, args.out, fps=fps, fourcc='mp4v')

    # 一次性写入文件
    with open(os.path.join(args.out_dir,"tracking_results.txt"), "w") as f:
        f.write("\n".join(track_results))


if __name__ == '__main__':
    main()