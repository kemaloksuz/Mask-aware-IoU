_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/minicoco500_detection_augm.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(step=[75, 95])
total_epochs = 100
