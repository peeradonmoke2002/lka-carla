# SCNN Training Setup & Compatibility Fixes

pytorch-auto-drive was written for older Python/PyTorch/Pillow versions.
After cloning it, apply the fixes below before training.

**Environment tested:** Python 3.10, PyTorch 2.x, CUDA 12.4, Pillow 10+, mmcv 2.x

---

## 1. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -U openmim && mim install mmcv
pip install "numpy<2.0"
pip install tensorboard --upgrade
pip install -r pytorch-auto-drive/requirements.txt
```

---

## 2. Create missing config files

### `configs/lane_detection/common/optims/ep36_poly_warmup200.py`
```python
lr_scheduler = dict(
    name='poly_scheduler_with_warmup',
    epochs=36,
    power=0.9,
    warmup_steps=200
)
```

### `configs/lane_detection/common/optims/segloss_3class.py`
```python
loss = dict(
    name='LaneLoss',
    existence_weight=0.1,
    ignore_index=255,
    weight=[0.4, 1, 1]
)
```

---

## 3. Fix Python 3.10 SyntaxWarning

**File:** `utils/models/segmentation/segmentation.py` line 45

```python
# Before (broken — compares string literal, not variable)
assert 'aux_classifier_cfg' is not None

# After
assert aux_classifier_cfg is not None
```

---

## 4. Fix mmcv 2.x import breaks

mmcv 2.x removed/moved several APIs used in pytorch-auto-drive.
These modules are NOT used by SCNN — wrap their imports with try/except so startup doesn't crash.

**File:** `utils/models/common_models/plugins/feature_flip_fusion.py`
```python
# Before
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d

# After
try:
    from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
except ImportError:
    ModulatedDeformConv2d = object
    modulated_deform_conv2d = None
```

**File:** `utils/datasets/video.py`
```python
# Before
from mmcv import VideoReader

# After
try:
    from mmcv import VideoReader
except ImportError:
    VideoReader = None
```

**File:** `utils/frames_to_video.py`
```python
# Before
from mmcv.utils import check_file_exist, track_progress

# After
try:
    from mmcv.utils import check_file_exist, track_progress
except ImportError:
    from mmengine.utils import track_progress
    def check_file_exist(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(f'{filename} does not exist')
```

---

## 5. Fix Pillow 10+ removed constants

`PIL.Image.LINEAR` was removed in Pillow 10. Replace all occurrences in `utils/transforms/transforms.py`:

```bash
sed -i 's/Image\.LINEAR/Image.BILINEAR/g' utils/transforms/transforms.py
```

Also add `InterpolationMode` import (needed for type reference, harmless):
```bash
sed -i 's/^from PIL import Image$/from PIL import Image\nfrom torchvision.transforms import InterpolationMode/' utils/transforms/transforms.py
```

---

## 6. Fix torchvision rotate/affine API

The local `utils/transforms/functional.py` uses `resample=` parameter (not `interpolation=`).
After step 5, `F.rotate` and `F.affine` calls in `transforms.py` must use `resample=Image.BILINEAR`, not `interpolation=`.

Check lines calling `F.rotate` and `F.affine` — they should look like:
```python
# Correct
image = F.rotate(image, angle, resample=Image.BILINEAR, ...)
image = F.affine(image, *ret, resample=Image.BILINEAR, ...)

# Wrong (causes TypeError)
image = F.rotate(image, angle, interpolation=..., ...)
```

---

## 7. Create output directory

```bash
mkdir -p pytorch-auto-drive/output
```

---

## 8. Register LkaAsSegmentation dataset

See step 3 in the main [README.md](../README.md) for the full class code to add to
`pytorch-auto-drive/utils/datasets/lane_as_segmentation.py` and `__init__.py`.
