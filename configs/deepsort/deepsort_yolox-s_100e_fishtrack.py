_base_ = [
    '../yolox/yolox_s.py',
    '../_base_/datasets/fish_track.py', '../_base_/default_runtime.py'
]

''''
修改字典类型，可以直接修改下级字段
对于非字典类型的字段，例如整数，字符串，列表等，重新定义即可完全覆盖，
'''

# dataset paht
dataset_type='MOTChallengeDataset'
data_root='data/fish_track'

# hooks
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    visualization=dict(type='TrackVisualizationHook', draw=False))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='TrackLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]


# detector = Config.fromfile('/home/kzy/project/PartDecoder/mmdetection/configs/yolox/yolox_s_8xb8-weizhoudao.py').model
detector = _base_.model
detector.pop('data_preprocessor')
detector.bbox_head.update(dict(num_classes=1))
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    '/home/kzy/project/PartDecoder/mmdetection/work_dirs/yolox_s_8xb8-weizhoudao/best_coco_bbox_mAP_epoch_95.pth'
    )
del _base_.model


model = dict(
    type='DeepSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    detector=detector,
    reid=dict(
        type='BaseReID',
        data_preprocessor=dict(type='mmpretrain.ClsDataPreprocessor'),
        backbone=dict(
            type='mmpretrain.ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss_cls=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
            loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
        )),
    tracker=dict(
        type='SORTTracker',
        motion=dict(type='KalmanFilter', center_only=False),
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))

train_dataloader = None
train_pipeline = None
train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
            dict(type='Resize', scale=_base_.img_scale, keep_ratio=True),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadTrackAnnotations'),
        ]),
    dict(type='PackTrackInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    # Now we support two ways to test, image_based and video_based
    # if you want to use video_based sampling, you can use as follows
    # sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    sampler=dict(type='TrackImgSampler'),  # image-based sampling
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/train_cocoformat.json',
        data_prefix=dict(img_path='train'),
        test_mode=True,
        pipeline=test_pipeline))


test_evaluator = dict(
    type='MOTChallengeMetric',
    postprocess_tracklet_cfg=[
        dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
    ],
    format_only=True,
    outfile_prefix='./work_dirs/deepsort_yolox-s_fishtrack_test1')