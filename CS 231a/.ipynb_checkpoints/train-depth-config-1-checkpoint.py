SWIN_CONFIGS = '../Video-Swin-Transformer/configs/'

_base_ = [
    SWIN_CONFIGS + '_base_/models/swin/swin_tiny_depth.py', SWIN_CONFIGS + '_base_/default_runtime.py'
] # Changed from swin_tiny_depth for debugging

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

# dataset settings
dataset_type = 'RawframeDepthDatasetPNG' # Change to VideoDataset when debugging model
DATASET_PATH = '/vision/group/ntu-rgbd/'
ann_file_train = '/sailhome/zanedurante/CS 231a/depth_1_train_ann.txt'
ann_file_val = '/sailhome/zanedurante/CS 231a/depth_1_train_ann.txt'
ann_file_test = DATASET_PATH + 'depth_test_ann.txt'

# Added for debugging
load_from = '../models/pre-trained/swin/swin_tiny_patch244_window877_kinetics400_1k.pth'


img_norm_cfg = dict(
    mean=[114.495], std=[57.63], to_bgr=False) # means of RGB
train_pipeline = [
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    #dict(type='Resize', scale=(-1, 256)), See if model can overfit the data
    #dict(type='RandomResizedCrop'), # See if model can overfit the data
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    
    #dict(type='Flip', flip_ratio=0.5), # See if model can overfit the data
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CopyDims', num_times=2), # Used for debugging (expand depth dimension to be 1 --> 3 channels) 
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
        num_clips=4,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=1, # Changed from 4
    val_dataloader=dict(
        videos_per_gpu=2,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
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
    interval=2, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
# Modify training schedule? Same as before... (changed 1e-4 --> 1e-3 --> 1e-5 --> 1e-4) AROUND THE WORLD  
optimizer = dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))

# Don't calculate validation loss for this experiment
workflow = [('train', 1)]

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 1000

# runtime settings
checkpoint_config = dict(interval=1000) # Never save since this is a test anyway
work_dir = WORK_DIR_PATH + 'small_depth_nturgbd_swin_tiny_patch244_window877_0-1Norm_4_exs.py'
find_unused_parameters = False

# Logging details
log_config = dict(  # Config to register logger hook
    interval=4,  # Interval to print the log
    hooks=[  # Hooks to be implemented during training
        dict(type='TextLoggerHook'),  # The logger used to record the training process
        # dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
    ])

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
