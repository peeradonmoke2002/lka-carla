#!/usr/bin/env python3
"""Quick SCNN inference on a single image for slide visualization."""
import sys
import numpy as np
import cv2
import torch

PAD = '/home/peeradon/lka-carla-yolo/pytorch-auto-drive'
sys.path.insert(0, PAD)

from utils.models.builder import MODELS  # noqa: E402

WEIGHTS   = f'{PAD}/checkpoints/resnet18_scnn_lka/model.pt'
IMG_PATH  = '/home/peeradon/lka-carla-yolo/Images/sample_rgb.jpg'
OUT_PATH  = '/home/peeradon/lka-carla-yolo/Images/scnn_result.jpg'
IN_W, IN_H = 800, 288
THRESH     = 0.3
NUM_CLASSES = 3
Y_REF_RATIO = 0.85

# ── Load model ──────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cfg = dict(
    name='standard_segmentation_model',
    backbone_cfg=dict(name='predefined_resnet_backbone', backbone_name='resnet18',
                      return_layer='layer4', pretrained=False,
                      replace_stride_with_dilation=[False, True, True]),
    reducer_cfg=dict(name='RESAReducer', in_channels=512, reduce=128),
    spatial_conv_cfg=dict(name='SpatialConv', num_channels=128),
    classifier_cfg=dict(name='DeepLabV1Head', in_channels=128, num_classes=NUM_CLASSES, dilation=1),
    lane_classifier_cfg=dict(name='SimpleLaneExist', num_output=NUM_CLASSES - 1,
                             flattened_size=(NUM_CLASSES * (IN_H // 16) * (IN_W // 16))),
)
net = MODELS.from_dict(model_cfg).to(device)
ckpt = torch.load(WEIGHTS, map_location=device)
net.load_state_dict(ckpt.get('model', ckpt))
net.eval()
print(f'Model loaded on {device}')

# ── Preprocess ──────────────────────────────────────────────────────────────
img_bgr = cv2.imread(IMG_PATH)
h, w = img_bgr.shape[:2]
rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
resized = cv2.resize(rgb, (IN_W, IN_H)).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], np.float32)
std  = np.array([0.229, 0.224, 0.225], np.float32)
x = torch.from_numpy(((resized - mean) / std).transpose(2, 0, 1)).unsqueeze(0).to(device)

# ── Inference ────────────────────────────────────────────────────────────────
with torch.no_grad():
    result = net(x)
    prob = torch.softmax(result['out'], dim=1)[0].cpu().numpy()  # (C, H, W)

left_prob  = cv2.resize(prob[1], (w, h))
right_prob = cv2.resize(prob[2], (w, h))

# ── Fit lanes ────────────────────────────────────────────────────────────────
def fit_lane(pmap, x_min=0, x_max=None):
    if x_max is None:
        x_max = pmap.shape[1]
    ys, xs = np.where(pmap > THRESH)
    mask = (xs >= x_min) & (xs < x_max)
    ys, xs = ys[mask], xs[mask]
    if len(xs) < 10:
        return None
    weights = pmap[ys, xs]
    try:
        return np.polyfit(ys, xs, 1, w=weights), int(ys.min()), int(ys.max())
    except np.linalg.LinAlgError:
        return None

y_ref    = int(h * Y_REF_RATIO)
img_cx   = w // 2
left_fit  = fit_lane(left_prob,  x_min=0,      x_max=w * 3 // 4)
right_fit = fit_lane(right_prob, x_min=w // 4, x_max=w)

if left_fit  and np.polyval(left_fit[0],  y_ref) > img_cx: left_fit  = None
if right_fit and np.polyval(right_fit[0], y_ref) < img_cx: right_fit = None

lx = float(np.polyval(left_fit[0],  y_ref)) if left_fit  else None
rx = float(np.polyval(right_fit[0], y_ref)) if right_fit else None

# ── Raw prob map on top of original image ────────────────────────────────────
overlay = np.zeros_like(img_bgr)
overlay[:, :, 1] = (left_prob  * 255).clip(0, 255).astype(np.uint8)  # green
overlay[:, :, 0] = (right_prob * 255).clip(0, 255).astype(np.uint8)  # blue
vis = cv2.addWeighted(overlay, 0.6, img_bgr, 1.0, 0)

cv2.imwrite(OUT_PATH, vis)
print(f'Saved: {OUT_PATH}')
print(f'lx={lx:.1f}  rx={rx:.1f}  center={((lx+rx)/2/w):.3f}' if lx and rx else 'Detection incomplete')
