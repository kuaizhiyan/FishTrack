_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/weizhoudao_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# _base_.optim_wrapper.type = 'AmpOptimWrapper'
_base_.model.roi_head.bbox_head.num_classes=31

# optimizer
_base_.optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00001, momentum=0.9, weight_decay=0.0005))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
_base_.auto_scale_lr = dict(enable=True, base_batch_size=64)
_base_.default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="loss",
        patience=5,
        min_delta=0.005),
    )