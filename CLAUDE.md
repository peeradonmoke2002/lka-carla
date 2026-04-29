# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# lka-carla-yolo

Vision-Based Lane Keeping Assist System with Weather-Robust Lane Detection using YOLOv8 and Adaptive Preprocessing on CARLA Simulator.

## Project Overview

This project implements a Lane Keeping Assist (LKA) system that:
1. Collects lane detection dataset automatically from CARLA using semantic segmentation camera
2. Trains YOLO-seg on CARLA-generated data
3. Applies adaptive image preprocessing to improve detection under adverse weather (rain, fog, night)
4. Controls the vehicle using a confidence-weighted PID controller — steering gain is modulated by YOLO detection confidence
5. Evaluates the full closed-loop pipeline across 4 weather presets in CARLA

The key contribution is the explicit link between **vision quality → control behavior**: when YOLO confidence is low (bad weather), the controller reduces steering aggressiveness to prevent erratic behavior.

---

## Workspace Layout

```
lka-carla-yolo/
├── lka_ws/                        # ROS2 workspace
│   └── src/
│       ├── carla_ros/             # Submodule: carla_ros_bridge (branch: lka)
│       ├── lka_dataset_collection/ # Phase 1: dataset collection node
│       ├── lka_perception/        # Phase 3–4: preprocessing + YOLO inference
│       └── lka_control/           # Phase 4: confidence-weighted PID controller
└── training/                      # Phase 2: YOLOv8 training scripts
```

### Submodule

`lka_ws/src/carla_ros` is a Git submodule tracking branch `lka` of `https://github.com/peeradonmoke2002/carla_ros.git`.

To initialize after cloning:
```bash
git submodule update --init --recursive
```

---

## Phases

### Phase 1 — Dataset Collection (`lka_ws/src/lka_dataset_collection/`)
- Spawn vehicle in CARLA with autopilot
- Attach RGB camera + semantic segmentation camera (synchronized)
- Extract road marking pixels (semantic class 6 = RoadLine in CARLA)
- Convert mask contours to YOLO polygon format and save `.txt` labels
- Target: ~1000 images across multiple CARLA maps and weather conditions
- Output: `dataset/images/` + `dataset/labels/` + `data.yaml`

### Phase 2 — Model Training (`training/`)
- Fine-tune `yolov8n-seg.pt` on collected dataset
- Validate mIoU on held-out scenes
- Export best model weights to `training/weights/best.pt`

### Phase 3 — Preprocessing Module (`lka_ws/src/lka_perception/`)
- ROS2 node subscribes to `/carla/ego_vehicle/rgb_front/image`
- Applies weather-adaptive preprocessing before passing to YOLO:
  - **Fog**: CLAHE (contrast limited adaptive histogram equalization)
  - **Rain/Night**: Gamma correction
  - **Faded markings**: Morphological edge enhancement
- Publishes enhanced image to `/lka/enhanced_image`

### Phase 4 — Detection + Control (`lka_ws/src/lka_perception/` + `lka_ws/src/lka_control/`)
- Detection node runs YOLOv8-seg on enhanced image
- Extracts lane center point (pixel x-coordinate) and per-frame confidence score
- Publishes to `/lka/lane_center` and `/lka/detection_confidence`
- Controller node subscribes to both topics
- Computes lateral error = (lane_center_x - image_width/2)
- Steering = Kp * confidence * lateral_error  ← confidence-weighted gain
- Publishes `CarlaEgoVehicleControl` to `/carla/ego_vehicle/vehicle_control_cmd`

### Phase 5 — Evaluation (`evaluation/`)
- Test across 4 CARLA weather presets: Clear, Rainy, Foggy, Night
- Vision metrics: mean IoU, mean confidence, detection rate (% valid frames)
- Control metrics: mean lateral error (m), lane departure count
- Compare: fixed-gain PID vs confidence-weighted PID

---

## ROS2 Topics

| Topic | Type | Description |
|---|---|---|
| `/carla/ego_vehicle/rgb_front/image` | `sensor_msgs/Image` | Raw RGB from CARLA |
| `/lka/enhanced_image` | `sensor_msgs/Image` | After adaptive preprocessing |
| `/lka/lane_center` | `std_msgs/Float32` | Normalized lane center x (0.0–1.0) |
| `/lka/detection_confidence` | `std_msgs/Float32` | YOLO confidence score (0.0–1.0) |
| `/carla/ego_vehicle/vehicle_control_cmd` | `carla_msgs/CarlaEgoVehicleControl` | Throttle + steer command |

---

## Key Implementation Notes

### Confidence-Weighted PID
```python
# steering gain scales with detection confidence
# low confidence → conservative steering (avoid erratic behavior in bad weather)
steering = Kp * confidence * lateral_error + Ki * integral + Kd * derivative
steering = max(-1.0, min(1.0, steering))  # clamp to [-1, 1]
```

### Adaptive Preprocessing Selection
```python
# weather_mode set via ROS2 parameter or CARLA weather API
if weather_mode == "fog":
    img = apply_clahe(img, clip_limit=3.0, tile_size=(8, 8))
elif weather_mode in ["rain", "night"]:
    img = apply_gamma(img, gamma=1.5)
elif weather_mode == "clear":
    img = img  # pass through
```

### CARLA Semantic Class for Lane
```python
# CARLA semantic segmentation: channel R = class ID
# Class 6 = RoadLine (lane markings)
class_map = semantic_array[:, :, 2]  # Red channel
lane_mask = (class_map == 6).astype(np.uint8) * 255
```

### TF Publishing (Disabled)
TF publishing from `carla_ros_bridge` is intentionally disabled by removing the `sensor.pseudo.tf` entry from `carla_spawn_objects/config/objects.json` in the `lka` branch of the submodule. This avoids TF conflicts with our own frame management. There is no launch parameter for this — it must be controlled via the objects config.

---

## Evaluation Metrics

| Layer | Metric | Description |
|---|---|---|
| Vision | mIoU | Mean Intersection over Union of lane mask |
| Vision | Mean confidence | Average YOLO score per weather condition |
| Vision | Detection rate | % frames with valid lane detection |
| Control | Mean lateral error | Average distance from lane center (meters) |
| Control | Lane departure count | # frames vehicle exceeds lane boundary |

---

## Weather Presets (CARLA)

```python
WEATHER_PRESETS = {
    "clear":  carla.WeatherParameters.ClearNoon,
    "rain":   carla.WeatherParameters.HardRainNoon,
    "fog":    carla.WeatherParameters(fog_density=80.0, fog_distance=10.0),
    "night":  carla.WeatherParameters.ClearSunset,  # low light
}
```

---

## References

- CLRNet (CVPR 2022): https://arxiv.org/abs/2203.10350
- CARLANE Benchmark (NeurIPS 2022): https://arxiv.org/abs/2206.08083
- Urban Traffic Dataset from CARLA + YOLOv8 (MDPI 2023): https://doi.org/10.3390/data9010004
- CarFree: Automatic Dataset Generation from CARLA (MDPI 2022): https://doi.org/10.3390/app12010281
- Surendra et al., Lane Detection using CARLA (IJRITCC 2023): https://doi.org/10.17762/ijritcc.v11i10.8891
