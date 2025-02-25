backend_args = None
data_root = 'data/fish_track'
dataset_type = 'ReIDDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='mmpretrain.ResNet'),
    data_preprocessor=dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        to_rgb=True,
        type='ReIDDataPreprocessor'),
    head=dict(
        act_cfg=dict(type='ReLU'),
        fc_channels=1024,
        in_channels=2048,
        loss_cls=dict(loss_weight=1.0, type='mmpretrain.CrossEntropyLoss'),
        loss_triplet=dict(loss_weight=1.0, margin=0.3, type='TripletLoss'),
        norm_cfg=dict(type='BN1d'),
        num_classes=380,
        num_fcs=1,
        out_channels=128,
        type='LinearReIDHead'),
    neck=dict(kernel_size=(
        4,
        8,
    ), stride=1, type='GlobalAveragePooling'),
    type='BaseReID')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=6,
        gamma=0.1,
        milestones=[
            5,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='reid/meta/val_10.txt',
        data_prefix=dict(img_path='reid/imgs'),
        data_root='data/fish_track',
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                256,
                128,
            ), type='Resize'),
            dict(type='PackReIDInputs'),
        ],
        triplet_sampler=None,
        type='ReIDDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    metric=[
        'mAP',
        'CMC',
    ], type='ReIDMetrics')
test_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        256,
        128,
    ), type='Resize'),
    dict(type='PackReIDInputs'),
]
train_cfg = dict(max_epochs=6, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='reid/meta/train_90.txt',
        data_prefix=dict(img_path='reid/imgs'),
        data_root='data/fish_track',
        pipeline=[
            dict(
                share_random_params=False,
                transforms=[
                    dict(
                        backend_args=None,
                        to_float32=True,
                        type='LoadImageFromFile'),
                    dict(
                        clip_object_border=False,
                        keep_ratio=False,
                        scale=(
                            256,
                            128,
                        ),
                        type='Resize'),
                    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
                ],
                type='TransformBroadcaster'),
            dict(
                meta_keys=(
                    'flip',
                    'flip_direction',
                ), type='PackReIDInputs'),
        ],
        triplet_sampler=dict(ins_per_id=4, num_ids=8),
        type='ReIDDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        share_random_params=False,
        transforms=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(
                clip_object_border=False,
                keep_ratio=False,
                scale=(
                    256,
                    128,
                ),
                type='Resize'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='GridMask'),
        ],
        type='TransformBroadcaster'),
    dict(meta_keys=(
        'flip',
        'flip_direction',
    ), type='PackReIDInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='reid/meta/val_10.txt',
        data_prefix=dict(img_path='reid/imgs'),
        data_root='data/fish_track',
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                256,
                128,
            ), type='Resize'),
            dict(type='PackReIDInputs'),
        ],
        triplet_sampler=None,
        type='ReIDDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metric=[
        'mAP',
        'CMC',
    ], type='ReIDMetrics')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs/reid_r50_fishreid_dataaug'
