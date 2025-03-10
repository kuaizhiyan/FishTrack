_base_ = [
    '../yolox/yolox_s.py',
    '../_base_/datasets/fish_track.py', '../_base_/default_runtime.py'
]

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

detector = _base_.model
detector.pop('data_preprocessor')
detector.bbox_head.num_classes=1
# detector.bbox_head.update(dict(num_classes=1))
detector['init_cfg'] = dict(
    type='Pretrained',
    checkpoint=  # noqa: E251
    'models/best_fishtrackcoco_bbox_mAP_epoch_30.pth')  # noqa: E501
del _base_.model


model = dict(
    type='DeepSORT',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        rgb_to_bgr=False,
        pad_size_divisor=32),
    detector=detector,
    tracker=dict(
        type='SORTTracker',
        motion=dict(type='KalmanFilter', center_only=False),
        obj_score_thr=0.5,
        match_iou_thr=0.5,
        reid=None))

train_dataloader = None

del _base_.test_pipeline
img_scale=(640,640)
backend_args=None
test_pipeline = [
    dict(
        type='TransformBroadcaster',
        transforms=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=img_scale, keep_ratio=True),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='LoadTrackAnnotations')
        ]),
    dict(type='PackTrackInputs')
]

train_cfg = None
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

test_evaluator = dict(
    type='MOTChallengeMetric',
    postprocess_tracklet_cfg=[
        dict(type='InterpolateTracklets', min_num_frames=5, max_num_frames=20)
    ],
    format_only=False,
    outfile_prefix='./work_dirs/sort_yolox_s_fish_track_3100100')