SWIN_CONFIGS = '../Video-Swin-Transformer/configs/'

_base_ = [
    SWIN_CONFIGS + '_base_/models/swin/swin_tiny.py', SWIN_CONFIGS + '_base_/default_runtime.py'
]

WORK_DIR_PATH = '/vision/group/ntu-rgbd/zane/work_dirs/'


model=dict(
    backbone=dict(patch_size=(2,4,4), drop_path_rate=0.1), 
    cls_head=dict(
        type='I3DHead',
        in_channels=768,
        num_classes=60,
        spatial_type='avg',
        dropout_ratio=0.5),
    test_cfg=dict(average_clips='prob'))

load_from ='../models/pre-trained/swin/swin_tiny_patch244_window877_kinetics400_1k.pth'

# dataset settings
dataset_type = 'RawframeDatasetPNG'
DATASET_PATH = '/vision/group/ntu-rgbd/'
ann_file_train = DATASET_PATH + '50_few_shot_rgb_train_ann.txt'
ann_file_val = DATASET_PATH + '50_few_shot_rgb_val_ann.txt'
ann_file_test = DATASET_PATH + '50_few_shot_rgb_val_ann.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
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

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=2, # was 4 improves performance but takes much longer
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    #dict(type='ThreeCrop', crop_size=224), # Improves performance but takes much longer 
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2, # Changed from 4
    val_dataloader=dict(
        videos_per_gpu=2,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=4,
        workers_per_gpu=2
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        pipeline=test_pipeline))

evaluation = dict(
    interval=2, metrics=['top_k_accuracy', 'mean_class_accuracy'], 
    save_best='top_k_accuracy')

# optimizer
# Modify training schedule???? changed lr from 1e-3 to 1e-4
optimizer = dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))

# Modify workflow to calculate validation loss every 2 epochs
workflow = [('train', 1), ('val', 1)]

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 10

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = WORK_DIR_PATH + '50_few_shot_nturgbd_swin_tiny_rgb.py'
find_unused_parameters = False

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=4,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)