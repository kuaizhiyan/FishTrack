
model = dict(
    type='BaseReID',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        norm_eval=False,
        # pretrained='torchvision://resnet50'
        plugins = [
            dict(
                position='after_conv3',
                 stages=(False, True, True, True),
                cfg = dict(type='CBAMBlock',reduction=16,kernel_size=7)
                # cfg = dict(type='BAMBlock', reduction=16, dia_val=1)
                # cfg = dict(type='SEAttention', reduction=8)
                # cfg = dict(type='ECAAttention', kernel_size=3),
                # cfg = dict(type='MPE_ds',groups=4, reduction=8, use_fc=True, global_method="avg_max")
                # cfg = dict(type='APA',
                #            use_channel_att=True,
                #            groups=16,
                #            use_global_spatial_att=True
                #            )
                # cfg = dict(type='NonLocal2d')
                
                #cfg = dict(type='ShuffleAttention', G=8)
                #cfg = dict(type='SpatialGroupEnhance', groups=8)
                #cfg = dict(type='DoubleAttention')
                #cfg = dict(type='SequentialPolarizedSelfAttention')
                #cfg = dict(type='CoTAttention', kernel_size=3)
                #cfg = dict(type='TripletAttention')
                #cfg = dict(type='CoordAtt', reduction=32)
                #cfg = dict(type='ParNetAttention')
            )
        ],
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
        num_classes=380,
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
