SWIN_CONFIGS = '../../Video-Swin-Transformer/configs/'

_base_ = [
    SWIN_CONFIGS + '_base_/models/swin/contra_swin_tiny.py', SWIN_CONFIGS + '_base_/default_runtime.py'
]

WORK_DIR_PATH = '/vision/group/ntu-rgbd/zane/work_dirs/'


model=dict(
    backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1,), 
    cls_head=None,
    #cls_head=dict( # added for debugging
    #    type='I3DHead',
    #    in_channels=768,
    #    num_classes=6000,
    ##    spatial_type='avg',
     #   dropout_ratio=0.5),
    train_cfg=dict(feature_extraction=True, aux_info=['pos_imgs', 'neg_imgs']),
    contra_loss=dict(type='TripletLoss'),
)

load_from ='/vision/group/ntu-rgbd/zane/work_dirs/50_few_shot_nturgbd_swin_tiny_rgb.py/latest.pth'

# dataset settings
dataset_type = 'RawframeTripletDatasetPNG'
DATASET_PATH = '/vision/group/ntu-rgbd/'
ann_file_train = DATASET_PATH + 'few_shot_rgb_unlabeled_ann.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=24, frame_interval=2, num_clips=1), # Reduce clip length to lower memory (was 32 before)
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'), # Makes training harder
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5), # Makes training harder
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

data = dict(
    videos_per_gpu=1,
    num_gpus=1,
    workers_per_gpu=1, # Changed from 4
    
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline),
)

# optimizer
# Modify training schedule???? changed lr from 1e-3 to 1e-4
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 50

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = WORK_DIR_PATH + 'contrastive_swin_tiny_rgb.py'
find_unused_parameters = False

# REMOVED: Not using mixed-precision training right now
# do not use mmdet version fp16
#fp16 = None
#optimizer_config = dict(
#    type="DistOptimizerHook",
#    update_interval=4,
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1,
#    use_fp16=True,
#)