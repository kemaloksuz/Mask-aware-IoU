_base_ = 'yolact_r50_4x8_coco_scale550_ATSSwmaIoU.py'

model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(_delete_=True,
            type='FPN_CARAFE',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            start_level=1,
            order=('conv', 'norm', 'act'),
            upsample_cfg=dict(
                type='carafe',
                up_kernel=5,
                up_group=1,
                encoder_kernel=3,
                encoder_dilation=1,
                compressed_channels=64)),
    bbox_head=dict(anchor_generator=dict(scales_per_octave=2)))

# optimizer
optimizer = dict(type='SGD', lr=8e-3, momentum=0.9, weight_decay=5e-4)
# learning policy
lr_config = dict(_delete_=True,
    policy='CosineAnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=0.)
