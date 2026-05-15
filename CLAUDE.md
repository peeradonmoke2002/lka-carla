# CLAUDE.md

## lka-carla

Vision-Based Lane Keeping Assist (LKA) using multiple detection methods on CARLA Simulator. KMUTT FIBO — Machine Vision course, student ID 67340700403.

## Goal

Compare lane detection methods (YOLO, Pure Vision/OpenCV, SCNN) → find ego-lane center → Pure Pursuit controller → test across 4 weather conditions (Clear, Rain, Fog, Night).

## Progress

| Component | Status |
|---|---|
| Dataset collection (2000 img, clear weather, low graphic) | ✅ Done |
| YOLO training (`yolo26l-seg.pt`, RTX 5070 Ti) | ✅ Done |
| Pure Vision detection (OpenCV — HSV yellow + gray Canny + Hough) | ✅ Done |
| SCNN detection (ResNet18 + SpatialConv, pytorch-auto-drive) | ✅ Done |
| GT node (signed CTE from Town01 OpenDrive map) | ✅ Done |
| Find center point of ego lane (all nodes publish `/lka/lane_center`) | ✅ Done |
| Pure Pursuit controller | ✅ Done |
| Automated evaluation script (`run_trials.py`, 3 methods × 4 weathers) | ✅ Done |
| Closed-loop evaluation runs | ✅ Done |

## Workspace

```
lka-carla-yolo/
├── models/
│   └── best_vision.pt          # yolo26l-seg weights
├── Images/                     # 4 reference images (clear/fog/night/rain)
├── bags/closed_loop/           # recorded trial bags + calibration_log.csv
├── pytorch-auto-drive/         # SCNN backbone (ResNet18 + SpatialConv)
│   └── checkpoints/resnet18_scnn_lka/model.pt
├── lka.yolo26/                 # YOLO dataset (train/valid splits, YOLO format)
├── lka.scnn/                   # SCNN dataset (images/, laneseg_label_w16/, list/)
├── analysis/                   # Post-run analysis scripts + results
│   ├── eval_perception.py
│   ├── eval_controller.py
│   ├── plot_perception.py
│   └── results/
├── training&process/           # Phase 2: training configs + setup guides
│   ├── scnn_lka_config.py      # pytorch-auto-drive config for SCNN
│   ├── SCNN_SETUP.md           # compat fixes for Python 3.10 / PyTorch 2.x
│   ├── train.ipynb / test.ipynb
│   └── yolo2scnn.py            # dataset format converter
├── lka_ws/src/
│   ├── carla_ros/              # submodule: github.com/peeradonmoke2002/carla_ros (branch: lka)
│   ├── lka_bringup/            # CARLA bridge + vehicle spawn + run_trials.py
│   ├── lka_dataset_collection/ # Phase 1: dataset collection + roi.yaml
│   ├── lka_msgs/               # Custom message interfaces (LaneCenter.msg)
│   ├── lka_perception/         # Phase 3: YOLO + Pure Vision + SCNN + GT nodes
│   └── lka_control/            # Phase 4: Pure Pursuit controller
└── REPORT.md                   # project report
```

```bash
git submodule update --init --recursive
```

## System Flow

```
Front CAM → [YOLO | Pure Vision] → /lka/lane_center → Pure Pursuit → CARLA vehicle control
                ↑                                            ↑
    /carla/weather_control                        /carla/ego_vehicle/odometry
    (live weather adaptive)
```

## Launch Commands

```bash
# Build all
cd lka_ws && colcon build --symlink-install && source install/setup.bash

# Single perception node (unified launcher — remaps to /lka/lane_center)
ros2 launch lka_bringup perception_only.launch.py method:=yolo
ros2 launch lka_bringup perception_only.launch.py method:=pure_vision
ros2 launch lka_bringup perception_only.launch.py method:=scnn

# Individual perception nodes (direct, no remap)
ros2 launch lka_perception pure_vision.launch.py
ros2 launch lka_perception yolo.launch.py
ros2 launch lka_perception scnn.launch.py
ros2 launch lka_perception gt.launch.py

# Pure Pursuit controller
ros2 launch lka_control lka_controller.launch.py

# Automated closed-loop evaluation (3 methods × 4 weathers, 3 repeats = 36 trials)
# Prerequisites: CARLA running + bridge spawned (bring_up_carla.launch.py)
python3 lka_ws/src/lka_bringup/scripts/run_trials.py
python3 lka_ws/src/lka_bringup/scripts/run_trials.py --methods yolo scnn --weathers clear rain
python3 lka_ws/src/lka_bringup/scripts/run_trials.py --dry-run   # print plan only
```

## Key Facts

- **Perception methods**: YOLO (seg), Pure Vision (HSV+Canny+Hough), SCNN (ResNet18+SpatialConv)
- **Detection classes**: `left_marking` (HSV yellow), `right_edge` (gray Canny / YOLO / SCNN)
- **YOLO model**: `yolo26l-seg.pt` — weights at `/home/peeradon/lka-carla-yolo/models/best_vision.pt`
- **YOLO result**: confidence 0.85–0.99 across all 4 weather conditions
- **SCNN model**: ResNet18 backbone — weights at `/home/peeradon/lka-carla-yolo/models/scnn.pt` (copied from `pytorch-auto-drive/checkpoints/resnet18_scnn_lka/model.pt`)
- **SCNN**: weather-agnostic; NUM_CLASSES=3 (bg, left_marking, right_edge); input 800×288
- **GT node**: loads Town01 OpenDrive offline, publishes signed CTE (m) from odometry → `/lka/gt/cross_track_m`
- **Controller**: Pure Pursuit (Autoware-based), `wheel_base=3.0046m`, `ld_k=2.4`; accepts `bias_offset` param
- **Dataset**: 2000 images, CARLA semantic seg class 6 (RoadLine) + sidewalk, clear weather only
- **TF publishing**: disabled — removed `sensor.pseudo.tf` from objects config in submodule
- **ROI polygon**: `[(102,892),(764,457),(825,456),(1465,894)]` — image 1600×900
- **Per-method topics**: `/lka/yolo/lane_center`, `/lka/pure_vision/lane_center`, `/lka/scnn/lane_center` (remapped to `/lka/lane_center` via `perception_only.launch.py`)

## ROS2 Topics

| Topic | Type | Note |
|---|---|---|
| `/carla/ego_vehicle/CAM_FRONT/image` | `sensor_msgs/Image` | camera input |
| `/carla/weather_control` | `carla_msgs/CarlaWeatherParameters` | live weather (YOLO + Pure Vision subscribe; SCNN ignores) |
| `/carla/ego_vehicle/odometry` | `nav_msgs/Odometry` | speed for Pure Pursuit + GT node |
| `/lka/lane_center` | `lka_msgs/LaneCenter` | active perception node (remapped by `perception_only.launch.py`) |
| `/lka/yolo/lane_center` | `lka_msgs/LaneCenter` | YOLO node direct output |
| `/lka/pure_vision/lane_center` | `lka_msgs/LaneCenter` | Pure Vision node direct output |
| `/lka/scnn/lane_center` | `lka_msgs/LaneCenter` | SCNN node direct output |
| `/lka/gt/cross_track_m` | `std_msgs/Float64` | signed CTE (m) from GT node; positive = ego right of centre |
| `/lka/pure_vision_image` | `sensor_msgs/Image` | Pure Vision debug view |
| `/lka/enhanced_image` | `sensor_msgs/Image` | YOLO debug view |
| `/lka/scnn_image` | `sensor_msgs/Image` | SCNN debug view |
| `/lka/controller/state` | `std_msgs/String` | controller state (`goal_reached`, etc.) |
| `/carla/ego_vehicle/vehicle_control_cmd` | `carla_msgs/CarlaEgoVehicleControl` | controller output |

### LaneCenter.msg fields
```
std_msgs/Header header
float32 center      # normalized ego-lane center x [0.0, 1.0]; -1.0 = no detection
float32 confidence  # [0.0, 1.0]; 0.0 = N/A (Pure Vision / SCNN)
bool    detected    # true when lane is actively detected / tracked
float32 lx          # left lane x at y_ref (pixels); -1.0 = not detected
float32 rx          # right lane x at y_ref (pixels); -1.0 = not detected
```

## Pure Vision HSV Thresholds (left marking)

| Weather | HSV_LO | HSV_HI | Gray Canny (right) |
|---|---|---|---|
| clear | [10, 30, 250] | [40, 120, 255] | (30, 90) |
| fog   | [10,  5, 180] | [40, 120, 255] | (20, 60) |
| night | [10,150,  30] | [40, 255, 255] | (20, 60) |
| rain  | [15, 25, 150] | [35, 255, 255] | (20, 60) |

## Pure Pursuit Parameters

| Parameter | Value | Note |
|---|---|---|
| `wheel_base` | 3.0046 m | Tesla (CARLA default) |
| `lane_width` | 3.5 m | standard lane |
| `min_lookahead` | 3.0 m | at low speed |
| `max_lookahead` | 10.0 m | at high speed |
| `ld_velocity_ratio` | 2.4 | from Autoware |
| `max_steer_rad` | 1.2217 rad | ~70° |
| `throttle` | 0.3 | constant |

## Training Hyperparameters

### YOLO (`yolo26l-seg.pt`, Segmentation)
| Parameter | Value | Note |
|---|---|---|
| epochs | 100 | — |
| imgsz | 640 | input image size |
| batch | 16 | images per step |
| patience | 30 | early stop threshold |
| dataset | `lka.yolo26/` | train/valid YOLO format |

### SCNN (ResNet18 + SpatialConv, pytorch-auto-drive)
| Parameter | Value | Note |
|---|---|---|
| epochs | 100 | `num_epochs` in config |
| train input_size | (360, 1000) | H×W, from `scnn_lka_config.py` |
| inference input | (288, 800) | H×W, from `scnn_node.py` params |
| batch_size | 8 | from `scnn_lka_config.py` (slide showed 32 — code is authoritative) |
| optimizer | SGD | lr=0.02, momentum=0.9 (sgd02 config) |
| pretrained | True | ResNet-18 from ImageNet |
| loss weights | [1.0, 4.0, 4.0] | [bg, left_marking, right_edge] |
| dataset | `lka.scnn/` | images/, laneseg_label_w16/, list/ |
| config | `training&process/scnn_lka_config.py` | copy to pytorch-auto-drive/configs/... |

## SCNN Parameters

| Parameter | Value | Note |
|---|---|---|
| `input_w` | 800 | model input width |
| `input_h` | 288 | model input height |
| `y_ref_ratio` | 0.85 | row fraction for lane x measurement |
| `prob_thresh` | 0.3 | min softmax prob to count pixel |
| `lane_width_px` | 760 | fallback px width when one side missing |
| `lane_width_m` | 4.0 | Town01 driving lane width |
| `enable_hysteresis` | True (trials) | stability filter (confirm_frames=3, lost_frames=5) |

## Automated Evaluation (`run_trials.py`)

**Location**: `lka_ws/src/lka_bringup/scripts/run_trials.py`

**Trials**: 3 methods × 4 weathers × 3 repeats = 36 runs

**Per-trial sequence** (steps 1–8):
1. Respawn — teleport ego to `(317.099, -195.158, 2.2)`, settle 7 s
2. Perception — launch node via `perception_only.launch.py`, wait `detected=True`
3. Calibrate bias — collect 5 s of center samples while stationary; pass `bias_offset` to controller
4. Weather — publish preset
5. Record — `ros2 bag record` → `bags/closed_loop/<method>_<weather>_rep<N>_<ts>/`
6. Controller — launch Pure Pursuit with calibrated `bias_offset`
7. Drive — wait for `controller_state=goal_reached` or 120 s timeout
8. Stop — kill controller → recorder → perception; verify topic goes quiet

**Calibration log**: `bags/closed_loop/calibration_log.csv`

**Bag topics recorded**: `/lka/lane_center`, `/lka/gt/cross_track_m`, `/carla/ego_vehicle/odometry`, `/carla/ego_vehicle/vehicle_control_cmd`, `/carla/weather_control`

## Center Point Formula

All three methods use the same formula:

1. Fit `x = a·y + b` through lane pixels (weighted least squares per side)
2. Evaluate at `y_ref` row: `lx = a_L·y_ref + b_L`, `rx = a_R·y_ref + b_R`
3. `center = (lx + rx) / (2 × W)  ∈ [0, 1]` → published as `LaneCenter.center`

`y_ref = y_ref_ratio × H` (YOLO/PV use 0.9; SCNN uses 0.85)

## Pure Pursuit Formulas

```
Step 1: lateral error    e = (center_norm − 0.5) × W_lane
Step 2: curvature        κ = 2e / l_d²
Step 3: steering angle   δ = arctan(L · κ)
Step 4: normalized cmd   δ_norm = clip(δ / δ_max, −1, 1)
```

`l_d = k · v`, clamped to `[min_lookahead, max_lookahead]`

## Hysteresis Filter

Applied in all perception nodes (enabled via `perception_only.launch.py`).

**Why needed**: SCNN prob below threshold / YOLO low confidence / Pure Vision Hough can't fit → frame drops → unstable lane center → steering jitter.

**Logic**:
- Requires **3 consecutive good frames** (`confirm_frames=3`) to enter TRACKING
- Holds last valid center for up to **5 consecutive bad frames** (`lost_frames=5`) before reverting to SEARCHING
- A "good frame" = detected + center jump < `jump_thresh` (0.12 normalized)

## Experiment 1 — Perception Performance Results

**Setup**: CARLA Town01, Tesla Model 3, 1600×900 front camera, 60 s/weather, vehicle stationary.
**Ground truth**: CARLA Waypoint API (normalized true center). All methods: **100% detection rate**.

**Metrics** (confirmed from `analysis/results/perception/metrics.csv`):
- `RMSE (px) = err_mean × 1600`
- `Jitter σ (px) = center_diff_std × 1600` (std of frame-to-frame center jump; note: slide reported max jitter which is larger)

| Weather | YOLO RMSE | YOLO Jitter σ | PV RMSE | PV Jitter σ | SCNN RMSE | SCNN Jitter σ |
|---------|-----------|---------------|---------|-------------|-----------|---------------|
| Clear   | 16.8 px   | 0.00 px       | 10.7 px | 1.44 px     | 17.9 px   | 0.00 px       |
| Rain    | 16.6 px   | 0.16 px       | 14.2 px | 3.04 px     | 17.1 px   | 0.32 px       |
| Fog     | 18.4 px   | 0.16 px       | 17.8 px | 2.72 px     | 18.6 px   | 0.00 px       |
| Night   | 16.8 px   | 0.16 px       | 11.0 px | 11.20 px    | 17.9 px   | 0.32 px       |

**Analysis script**: `analysis/eval_perception.py`, `analysis/plot_perception.py`

## Experiment 2 — Controller Performance Results

**Setup**: CARLA Town01, Tesla Model 3, Pure Pursuit (throttle=0.3), hysteresis ON, 3 repeats/condition.
**Ground truth**: CARLA Waypoint API (real-world CTE from GT node). All methods: **0% off-lane rate**.

**Metrics** (confirmed from `analysis/results/controller/controller_metrics_summary.csv`):
- `CTE RMSE (cm) = cte_rmse_mean × 100`
- `Max Steer Jitter (%) = steer_jitter_mean × 100`

| Weather | YOLO CTE | YOLO Steer J | PV CTE  | PV Steer J | SCNN CTE | SCNN Steer J |
|---------|----------|--------------|---------|------------|----------|--------------|
| Clear   | 1.30 cm  | 1.41%        | 3.41 cm | 0.74%      | 2.44 cm  | 0.52%        |
| Rain    | 2.19 cm  | 1.09%        | 3.01 cm | 2.07%      | 1.63 cm  | 0.58%        |
| Fog     | 1.90 cm  | 0.83%        | 1.72 cm | 1.52%      | 1.69 cm  | 0.26%        |
| Night   | 1.97 cm  | 0.79%        | 3.97 cm | 4.81%      | 1.37 cm  | 0.59%        |

**Analysis script**: `analysis/eval_controller.py`

## Weather Presets (Town01, confirmed)

| Field | Clear | Rain | Fog | Night |
|---|---|---|---|---|
| cloudiness | 0.0 | 60.0 | 80.0 | 0.0 |
| precipitation | 0.0 | 40.0 | 0.0 | 0.0 |
| precipitation_deposits | 0.0 | 40.0 | 0.0 | 0.0 |
| wind_intensity | 0.0 | 30.0 | 0.0 | 0.0 |
| sun_azimuth_angle | 0.0 | 275.0 | 0.0 | 0.0 |
| sun_altitude_angle | 75.0 | 20.0 | 45.0 | -90.0 |
| fog_density | 0.0 | 5.0 | 80.0 | 0.0 |
| fog_distance | 0.0 | 0.75 | 10.0 | 0.0 |
| wetness | 0.0 | 80.0 | 0.0 | 0.0 |

```bash
# Clear
ros2 topic pub --once /carla/weather_control carla_msgs/msg/CarlaWeatherParameters \
  "{cloudiness: 0.0, precipitation: 0.0, precipitation_deposits: 0.0, wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: 75.0, fog_density: 0.0, fog_distance: 0.0, wetness: 0.0}"

# Rain
ros2 topic pub --once /carla/weather_control carla_msgs/msg/CarlaWeatherParameters \
  "{cloudiness: 60.0, precipitation: 40.0, precipitation_deposits: 40.0, wind_intensity: 30.0, sun_azimuth_angle: 275.0, sun_altitude_angle: 20.0, fog_density: 5.0, fog_distance: 0.75, wetness: 80.0}"

# Fog
ros2 topic pub --once /carla/weather_control carla_msgs/msg/CarlaWeatherParameters \
  "{cloudiness: 80.0, precipitation: 0.0, precipitation_deposits: 0.0, wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: 45.0, fog_density: 80.0, fog_distance: 10.0, wetness: 0.0}"

# Night
ros2 topic pub --once /carla/weather_control carla_msgs/msg/CarlaWeatherParameters \
  "{cloudiness: 0.0, precipitation: 0.0, precipitation_deposits: 0.0, wind_intensity: 0.0, sun_azimuth_angle: 0.0, sun_altitude_angle: -90.0, fog_density: 0.0, fog_distance: 0.0, wetness: 0.0}"
```

### Weather classification thresholds (both perception nodes)
```python
fog_density > 40   → fog
precipitation > 30 → rain
sun_altitude < 0   → night
else               → clear
```

# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
