# model settings
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    type='OrientedRepPointsDetector',
    # pretrained='torchvision://resnet50',
    pretrained = "F:/360downloads\OrientedRepPoints-main\work_dirs\orientedreppoints_r50_demo/resnet50-19c8e357.pth",
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
    ),
    neck=
        dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
            # 问号??  为什么不用start_level=0,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=norm_cfg
        ),
    bbox_head=dict(
        type='OrientedRepPointsHead',
        num_classes=16,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        # gradient_mul=0.3,
        gradient_mul=0.0,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=2,
        norm_cfg=norm_cfg,
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        # loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=9.0),
        # loss_rbox_init=dict(type='GIoULoss', loss_weight=0.375),
        loss_rbox_init=dict(type='GIoULoss', loss_weight=1.0),
        # loss_rbox_refine=dict(type='GIoULoss', loss_weight=1.0),
        loss_rbox_refine=dict(type='GIoULoss', loss_weight=0.375),
        # loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
        loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
        # loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
        loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
        top_ratio=0.4,
        # is_division=True,
        # is_division=False,
        # is_division_pts=False, # 此为common1x1卷积
        # is_division_pts=True,# 此为分类自适应卷积
        # my_pts_mode="com3",  # "pts_up","pts_down","com1","com3","demo"
        my_pts_mode = "ide",  # "pts_up","pts_down","com1","com3","demo"
        # my_pts_mode="com1",  # "pts_up","pts_down","com1","com3","demo"
    ))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='PointAssigner', scale=4, pos_num=4),
        # assigner=dict(type='PointAssigner', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(
            type='MaxIoUAssigner', #pre-assign
            pos_iou_thr=0.1,
            neg_iou_thr=0.1,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    # score_thr=0.05, # 如果没有大于这个阈值的det时会发生严重的错误
    score_thr=0.05, # 如果没有大于这个阈值的det时会发生严重的错误
    nms=dict(type='rnms', iou_thr=0.4),
    # nms=dict(type='rnms', iou_thr=0.9),
    max_per_img=2000)

# dataset settings
dataset_type = 'DotaDataset'
# data_root = 'data/dota_1024/'
# data_root = 'data/dota_512_train_val/'
data_root = 'data/dota_512_first100_train_val/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CorrectBox', correct_rbbox=True, refine_rbbox=True),
    dict(type='RotateResize',
        # img_scale=[(666, 512), (666, 512)],
        # img_scale=[(888, 666), (888, 788)],
        # img_scale=[(512, 448), (512, 576)],
        img_scale=[(512, 512), (512, 512)],
        # img_scale=[(1333, 768), (1333, 1280)],
        keep_ratio=True,
        multiscale_mode='range',
        clamp_rbbox=False),
    dict(type='RotateRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_scale = (1024, 1024),
        img_scale = (512, 512),
        # img_scale=(1333, 1024),
        # img_scale=(1204, 1024),

        flip=False,
        transforms=[
            dict(type='RotateResize', keep_ratio=True),
            dict(type='RotateRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'trainval_split/trainval_dota.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points(3.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points(one_img_full_dets.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points(val.json',
        ann_file=data_root + 'trainval_split/trainval.json',
        img_prefix=data_root + 'trainval_split/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'test_split/test_dota.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points.json',  #这是part1; val需要ann
        # ann_file=data_root + 'trainval_split/trainval_coco_8points(one_img_full_dets.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points(P0887.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points(val.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points(3.json',
        ann_file=data_root + 'trainval_split/trainval.json',
        img_prefix=data_root + 'trainval_split/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'test_split/test_dota.json',
        ann_file=data_root + 'test_split/test_coco_8points.json',
        # ann_file=data_root + 'trainval_split/trainval_coco_8points(one_img_full_dets.json',
        img_prefix=data_root + 'test_split/images/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
# optimizer
# optimizer = dict(type='SGD', lr=0.0008, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.0008, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    # warmup_iters=50,
    warmup_ratio=1.0 / 3,
    # step = [24, 32, 38],
    step=[8, 16, 20]
)
checkpoint_config = dict(interval=1)
# checkpoint_config = dict(interval=-1)
# yapf:disable
log_config = dict(
    interval=100,
    # interval=9,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
evaluate_config = dict(
iou_thr=0.5,
# iou_thr=0.3,
)
# yapf:enable
# runtime settings
total_epochs = 20
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = 'work_dirs/orientedreppoints_r50_demo/'
# work_dir = 'work_dirs/orp_r50_division_pts'
# work_dir = 'work_dirs/orp_r50_division_common_conv'
# work_dir = 'work_dirs/orp_r50_division_common_conv5'
# work_dir = 'work_dirs/orp_r50_division_common_conv1'
# work_dir = 'work_dirs/orp_r50_division_ide_nolimit'
# work_dir = 'work_dirs/orp_r50_part1(val_first100_ide'
work_dir = 'work_dirs/orp_r50_part1(val_first100_norefine'
# work_dir = 'work_dirs/orp_r50_division_common_conv0'

# load_from = None
load_from = 'work_dirs/orp_r50_part1(val_first100_norefine/latest.pth'
# load_from = 'work_dirs/orp_r50_part1(val_first100_norefine/epoch_3.pth'
# load_from = 'work_dirs/orp_r50_part1(val_first100_ide/epoch_8.pth'
# load_from = 'work_dirs/orp_r50_what_nolimit/epoch_120.pth'
# load_from = 'work_dirs/orp_r50_division_ide_nolimit/epoch_80(ide_same_cfg0.979.pth'
# load_from = 'work_dirs/orp_r50_what_nolimit/epoch_90(ide_0_9887.pth'
# load_from = 'work_dirs/orp_r50_what_nolimit/epoch_80(ide_0.9778.pth'
# load_from = 'work_dirs/orp_r50_division_common_conv/latest.pth'
# load_from = 'work_dirs/orp_r50_division_common_conv0/epoch_80.pth'
# load_from = 'work_dirs/orp_r50_division_common_conv5/epoch_80.pth'
# load_from = 'work_dirs/orp_r50_division_common_conv1/epoch_80.pth'
# load_from = 'work_dirs/orp_r50_division_common_conv1/epoch_100.pth'
# load_from = 'work_dirs/orp_r50_division_common_conv/epoch_80(temp.pth'

resume_from = None
# resume_from = 'work_dirs/orp_r50_part1(val_first100_norefine/epoch_1.pth'

# resume_from = 'work_dirs/orientedreppoints_r50_demo/epoch_2.pth'
# resume_from = 'work_dirs/orp_r50_division_common_conv/epoch_130.pth'
# resume_from = 'work_dirs/orp_r50_division_common_conv1/epoch_70.pth'


workflow = [('train', 1)]
