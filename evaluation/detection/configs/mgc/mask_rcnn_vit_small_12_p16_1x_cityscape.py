# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Hyperparameters modifed from
https://github.com/SwinTransformer/Swin-Transformer-Object-Detection
"""

_base_ = [
    '../_base_/models/mask_rcnn_vit_small.py',
    '../_base_/datasets/cityscapes_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained ='' , 
    backbone=dict(
        type='ViT',
        patch_size=16,
        # img_size=256,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        out_indices=[3, 5, 7, 11],
        abs = True,
        pqcl = False,
        num_classes=8,
    ),
    neck=dict(in_channels=[384, 384, 384, 384]),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_head=dict(num_classes=8,),
        mask_head=dict(num_classes=8,),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32])
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
data = dict(samples_per_gpu=1)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'pos_embed': dict(decay_mult=0.),
                                                 'cls_token': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(warmup_iters=500*2,step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
