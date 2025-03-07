_base_ = [
    '../_base_/datasets/fish_track_reid.py', '../_base_/default_runtime.py'
]
model = dict(
    type='BaseReID',
    data_preprocessor=dict(
        type='ReIDDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    backbone=dict(
        type='ResNeSt',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        norm_eval=False,
        # pretrained='torchvision://resnet50'
        # init_cfg=dict(
        #     type='Pretrained',
        # )
        ),
    neck=dict(type='GlobalAveragePooling', kernel_size=(4, 8), stride=1),
    head=dict(
        type='LinearReIDHead',
        num_fcs=1,
        in_channels=2048,
        fc_channels=1024,
        out_channels=128,
        num_classes=81,        # train cls < 80
        loss_cls=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0),
        loss_triplet=dict(type='TripletLoss', margin=0.3, loss_weight=1.0),
        norm_cfg=dict(type='BN1d'),
        act_cfg=dict(type='ReLU')),
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint=  # noqa: E251
    #     '/home/kzy/project/PartDecoder/mmdetection/work_dirs/reid_r50_fishreid_dataaug/gridmask.pth'  # noqa: E501
    # )
    )

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=None,
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=6,
        by_epoch=True,
        milestones=[5],
        gamma=0.1)
]

# train_dataloader = dict(
#     sampler=dict(type='InfiniteSampler'),
#     sampler=dict(type='DefaultSampler'),
#     dataset=dict(
#         triplet_sampler=dict(num_ids=32, ins_per_id=4),
# ))

# train, val, test setting
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)
log_processor = dict(by_epoch=False)
# train_cfg = dict(
#     type='IterBasedTrainLoop',
#     max_iters=140000,
#     val_interval=2000,
# )
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

del _base_.train_pipeline

backend_args = None
train_pipeline = [
    dict(
        type='TransformBroadcaster',
        share_random_params=False,
        transforms=[
            dict(
                type='LoadImageFromFile',
                backend_args=backend_args,
                to_float32=True),
            dict(
                type='Resize',
                scale=(256, 128),
                keep_ratio=False,
                clip_object_border=False),
        ]),
    dict(type='PackReIDInputs', meta_keys=('flip', 'flip_direction'))
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,save_best='auto'),
)
