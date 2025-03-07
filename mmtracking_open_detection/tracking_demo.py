# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image

# sentence-transformers
from sentence_transformers import SentenceTransformer

# groudingdino
try:
    import groundingdino
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import get_tokenlizer
    from groundingdino.util.utils import clean_state_dict
    grounding_dino_transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
except ImportError:
    groundingdino = None

# mmdet
try:
    import mmcv
    import mmdet
    from mmdet.apis import inference_detector, init_detector
    from mmdet.models.trackers import ByteTracker
    from mmdet.structures import DetDataSample
    from mmdet.visualization.local_visualizer import TrackLocalVisualizer
    from mmengine.config import Config
    from mmengine.structures import InstanceData
except ImportError:
    mmdet = None

try:
    import segment_anything
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    segment_anything = None

import sys

sys.path.append('../')

from mmtracking_open_detection.utils import apply_exif_orientation  # noqa

# GLIP inflect
try:
    import maskrcnn_benchmark

    from mmtracking_open_detection.predictor_glip import GLIPDemo
except ImportError:
    maskrcnn_benchmark = None

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def parse_args():
    parser = argparse.ArgumentParser('Open Tracking Demo', add_help=True)
    parser.add_argument('inputs', type=str, help='path to video or image dirs')
    parser.add_argument('det_config', type=str, help='path to det config file')     # Important!  目标检测器的配置
    parser.add_argument('det_weight', type=str, help='path to det weight file')
    parser.add_argument(
        '--sam-type',
        type=str,
        default='vit_h',
        choices=['vit_h', 'vit_l', 'vit_b'],
        help='sam type')
    parser.add_argument(
        '--sam-weight',
        type=str,
        default='../models/sam_vit_h_4b8939.pth',
        help='path to checkpoint file')
    parser.add_argument('--text-prompt', '-t', type=str, help='text prompt')
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
    parser.add_argument(
        '--sam-device',
        '-s',
        default='cuda:0',
        help='Device used for inference')
    parser.add_argument('--cpu-off-load', '-c', action='store_true')
    parser.add_argument('--mots', action='store_true')

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

# 创建 grounding dino 目标检测器
def __build_grounding_dino_model(args):
    gdino_args = Config.fromfile(args.det_config)
    model = build_model(gdino_args)
    checkpoint = torch.load(args.det_weight, map_location='cpu')
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    return model


def __build_glip_model(args):
    assert maskrcnn_benchmark is not None
    from maskrcnn_benchmark.config import cfg
    cfg.merge_from_file(args.det_config)
    cfg.merge_from_list(['MODEL.WEIGHT', args.det_weight])
    cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
    model = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=args.box_thr,
        show_mask_heatmaps=False)
    return model


def build_detecter(args):
    if 'GroundingDINO' in args.det_config:
        detecter = __build_grounding_dino_model(args)
    elif 'glip' in args.det_config:
        detecter = __build_glip_model(args)
    else:
        config = Config.fromfile(args.det_config)
        if 'init_cfg' in config.model.backbone:
            config.model.backbone.init_cfg = None
        detecter = init_detector(
            config, args.det_weight, device='cpu', cfg_options={})
    return detecter


def create_positive_dict(tokenized, tokens_positive, labels):
    """construct a dictionary such that positive_map[i] = j,
    if token i is mapped to j label"""

    positive_map_label_to_token = {}

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)          # 将原始输入字符位置，映射到 token 中的位置
            end_pos = tokenized.char_to_token(end - 1)

            assert beg_pos is not None and end_pos is not None
            positive_map_label_to_token[labels[j]] = []
            for i in range(beg_pos, end_pos + 1):
                positive_map_label_to_token[labels[j]].append(i)        # 这一个标签的每一个字符，都映射到 token 

    return positive_map_label_to_token


def convert_grounding_to_od_logits(logits,              # 【900，256】 900 是 proposal 的个数
                                   num_classes,
                                   positive_map,        # 每个 label 对应的 token
                                   score_agg='MEAN'):
    """
    logits: (num_query, max_seq_len)
    num_classes: 80 for COCO
    """
    assert logits.ndim == 2
    assert positive_map is not None
    scores = torch.zeros(logits.shape[0], num_classes).to(logits.device)    # [900,8]
    # 256 -> 80, average for each class
    # score aggregation method
    if score_agg == 'MEAN':  # True
        for label_j in positive_map:
            scores[:, label_j] = logits[:,
                                        torch.LongTensor(positive_map[label_j]
                                                         )].mean(-1)        # [900,1]->[900,]
    else:
        raise NotImplementedError
    return scores

#   Important!!!    执行开放目标检测器的过程 
def run_detector(model, image_new, args, label_name=None):
    """
    image_new: PIL.Image.Image image (1280,720)
    label_name: list(8):
    """
    
    if args.cpu_off_load:
        if 'glip' in args.det_config:
            model.model = model.model.to(args.det_device)
            model.device = args.det_device
        else:
            model = model.to(args.det_device)

    if 'GroundingDINO' in args.det_config:

        image, _ = grounding_dino_transform(image_new, None)  # 3, h, w (1280,720)-> [3,750,1333]  resize+norm
        tokens_positive = []        # list(8):[(s_i,e_i)],... 切分输入 prompt 中的每个类别 起止下标
        start_i = 0
        separation_tokens = ' . '
        for _index, label in enumerate(label_name):
            end_i = start_i + len(label)
            tokens_positive.append([(start_i, end_i)])
            if _index != len(label_name) - 1:
                start_i = end_i + len(separation_tokens)
        # 创建 tokenizer
        tokenizer = get_tokenlizer.get_tokenlizer('bert-base-uncased')
        tokenized = tokenizer(
            args.text_prompt, padding='longest', return_tensors='pt')   # tokenize text prompt
        # 将原始text下标映射到 token 下标  list(8):[1],[..],  每个类别对应到 token 中的下标
        positive_map_label_to_token = create_positive_dict(
            tokenized, tokens_positive, list(range(len(label_name))))

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

    elif 'glip' in args.det_config:
        top_predictions = model.inference(
            image_new, args.text_prompt, use_other_text=False)

        pred_instances = InstanceData()
        pred_instances.bboxes = top_predictions.bbox
        pred_instances.labels = top_predictions.get_field('labels') - 1
        pred_instances.scores = top_predictions.get_field('scores')

    else:
        result = inference_detector(model, image_new)
        pred_instances = result.pred_instances[
            result.pred_instances.scores > args.box_thr]

    if args.cpu_off_load:
        if 'glip' in args.det_config:
            model.model = model.model.to('cpu')
            model.device = 'cpu'
        else:
            model = model.to('cpu')

    return pred_instances


def main():
    if mmdet is None:
        raise RuntimeError('mmdet is not installed,\
                 please install it follow README')
    args = parse_args()

    if 'glip' in args.det_config:
        if maskrcnn_benchmark is None:
            raise RuntimeError('GLIP model is not installed,\
                 please install it follow README')
    elif 'GroundingDINO' in args.det_config:
        if groundingdino is None:
            raise RuntimeError('GroundingDINO model is not installed,\
                 please install it follow README')
    elif args.mots:
        if segment_anything is None:
            raise RuntimeError('SAM model is not installed,\
                 please install it follow README')

    if args.cpu_off_load is True:
        if 'cpu' in args.det_device and 'cpu ' in args.sam_device:
            raise RuntimeError(
                'args.cpu_off_load is an invalid parameter due to '
                'detection and mask model IS on the cpu.')

    if 'GroundingDINO' in args.det_config or 'glip' in args.det_config or \
       'Detic' in args.det_config:
        assert args.text_prompt

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

    # text_prompt
    text_prompt = args.text_prompt
    text_prompt = text_prompt.lower()
    text_prompt = text_prompt.strip()
    if not text_prompt.endswith('.'):
        text_prompt = text_prompt + '.'
    args.text_prompt = text_prompt
    

    # visulization
    visualizer = TrackLocalVisualizer()
    visualizer.dataset_meta = {'classes': label_name}

    # det model
    det_model = build_detecter(args)

    # sam model
    if args.mots:
        build_sam = sam_model_registry[args.sam_type]
        sam_model = SamPredictor(build_sam(checkpoint=args.sam_weight))
        if not args.cpu_off_load:
            sam_model.mode = sam_model.model.to(args.sam_device)

    # tracker  创建跟踪器
    tracker = ByteTracker(
        motion=dict(type='KalmanFilter'),
        obj_score_thrs=dict(
            high=args.obj_score_thrs_high, low=args.obj_score_thrs_low),
        init_track_thr=args.init_track_thr,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=args.num_frames_retain)

    if not args.cpu_off_load:
        if 'glip' in args.det_config:
            det_model.model = det_model.model.to(args.det_device)
            det_model.device = args.det_device
        else:
            det_model = det_model.to(args.det_device)

    if 'Detic' in args.det_config:
        from projects.Detic.detic.utils import (get_text_embeddings,
                                                reset_cls_layer_weight)
        det_model.dataset_meta['classes'] = label_name
        embedding = get_text_embeddings(custom_vocabulary=custom_vocabulary)
        reset_cls_layer_weight(det_model, embedding)

    # 保存跟踪结果
    track_results = []

    for frame_id, img in enumerate(imgs):
        save_path = os.path.join(args.out_dir, f'{frame_id:06d}.jpg')

        if isinstance(img, str):
            image_path = osp.join(args.inputs, img)                 # 
            if 'GroundingDINO' in args.det_config:
                image_new = Image.open(image_path).convert('RGB')       # （1280，720）
                image_new = apply_exif_orientation(image_new)           # PIL.Image (1280,720)
            else:
                image_new = cv2.imread(image_path)
        else:
            if 'GroundingDINO' in args.det_config:
                image_new = Image.fromarray(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                image_new = apply_exif_orientation(image_new)
            else:
                image_new = img

        pred_instances = run_detector(det_model, image_new, args, label_name) # {boxes[325,4]原图尺寸,labels[325],scores[325]} ,给出的还是对应的下标，只不过标签可以手动输入

        # track input
        img_data_sample = DetDataSample()                           # 构造 DetDataSample()
        img_data_sample.pred_instances = pred_instances
        img_data_sample.set_metainfo(dict(frame_id=frame_id))       # 所以，只要使用检测器给出检测结果，送到 Tracker 里面就好了

        # track
        pred_track_instances = tracker.track(img_data_sample) # boxes:[11,4] instance_id:[11](track_id),labels[11], scores[11],
        img_data_sample.pred_track_instances = pred_track_instances

        if 'GroundingDINO' in args.det_config:
            vis_image = np.asarray(image_new)
        else:
            vis_image = image_new[..., ::-1]

        if args.mots:       # False
            if args.cpu_off_load:
                sam_model.mode = sam_model.model.to(args.sam_device)
            if isinstance(img, str):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            sam_model.set_image(image)

            transformed_boxes = sam_model.transform.apply_boxes_torch(
                pred_track_instances.bboxes, image.shape[:2])
            transformed_boxes = transformed_boxes.to(sam_model.model.device)

            masks, _, _ = sam_model.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False)
            pred_track_instances.masks = masks.squeeze().cpu().numpy()

            if args.cpu_off_load:
                sam_model.model = sam_model.model.to('cpu')

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
            track_results.append(f"{frame_id},{obj_id},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},{conf:.3f},-1,-1,-1")


    mmcv.frames2video(args.out_dir, args.out, fps=fps, fourcc='mp4v')

    # 一次性写入文件
    with open(os.path.join(args.out_dir,"tracking_results.txt"), "w") as f:
        f.write("\n".join(track_results))


if __name__ == '__main__':
    main()
