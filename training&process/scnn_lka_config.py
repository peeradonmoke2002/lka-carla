# pytorch-auto-drive config for SCNN on the LKA CARLA dataset
#
# Usage (from ~/pytorch-auto-drive/):
#   cp /home/peeradon/lka-carla-yolo/training/scnn_lka_config.py \
#      configs/lane_detection/scnn/resnet18_lka.py
#
#   python main_landet.py --config configs/lane_detection/scnn/resnet18_lka.py
#   python main_landet.py --config configs/lane_detection/scnn/resnet18_lka.py --test
#
# Classes:  0=background  1=left_marking  2=right_edge

from importmagician import import_from

with import_from('./'):
    from configs.lane_detection.common.optims.sgd02 import optimizer
    from configs.lane_detection.common.optims.ep36_poly_warmup200 import lr_scheduler

# ── Dataset ────────────────────────────────────────────────────────────────────
LKA_ROOT = '/home/peeradon/lka-carla-yolo/lka.scnn'

dataset = dict(
    name='LkaAsSegmentation',   # register this class in utils/datasets/__init__.py
    root=LKA_ROOT,
    num_classes=3,
    train_split='list/train_gt.txt',
    val_split='list/val_gt.txt',
    img_dir='images',
    mask_dir='laneseg_label_w16',
)

train_augmentation = dict(
    name='train_level0_288',   # resize to 288×800 + random flip + colour jitter
)
test_augmentation = dict(
    name='test_288',
)

# ── Loss ───────────────────────────────────────────────────────────────────────
loss = dict(
    name='segloss_3class',   # cross-entropy on 3 classes + BCE lane-existence loss
    weight=[1.0, 4.0, 4.0],  # upweight lane classes (minority pixels)
)

# ── Model ──────────────────────────────────────────────────────────────────────
model = dict(
    name='standard_segmentation_model',
    backbone_cfg=dict(
        name='predefined_resnet_backbone',
        backbone_name='resnet18',
        return_layer='layer4',
        pretrained=True,
        replace_stride_with_dilation=[False, True, True],
    ),
    reducer_cfg=dict(name='RESAReducer', in_channels=512, reduce=128),
    spatial_conv_cfg=dict(name='SpatialConv', num_channels=128),
    classifier_cfg=dict(
        name='DeepLabV1Head',
        in_channels=128,
        num_classes=3,
        dilation=1,
    ),
    lane_classifier_cfg=dict(
        name='SimpleLaneExist',
        num_output=2,          # 2 lanes: left_marking, right_edge
        flattened_size=2700,   # num_classes * (288/8/2) * (800/8/2) = 3 * 18 * 50
    ),
)

# ── Training ───────────────────────────────────────────────────────────────────
train = dict(
    exp_name='resnet18_scnn_lka',
    workers=4,
    batch_size=8,
    checkpoint=None,
    world_size=0,
    dist_url='env://',
    device='cuda',
    val_num_steps=0,
    save_dir='./checkpoints',
    input_size=(360, 1000),
    original_size=(900, 1600),
    num_classes=3,
    num_epochs=100,
    collate_fn=None,
    seg=True,
)

# ── Testing ────────────────────────────────────────────────────────────────────
test = dict(
    exp_name='resnet18_scnn_lka',
    workers=4,
    batch_size=16,
    checkpoint='./checkpoints/resnet18_scnn_lka/model.pt',
    device='cuda',
    save_dir='./checkpoints',
    seg=True,
    gap=20,
    ppl=18,
    thresh=0.3,
    collate_fn=None,
    input_size=(360, 1000),
    original_size=(900, 1600),
    max_lane=2,
    dataset_name='lka',
)
