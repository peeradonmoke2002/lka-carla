# CLAUDE.md

## lka-carla

Vision-Based Lane Keeping Assist (LKA) using multiple detection methods on CARLA Simulator. KMUTT FIBO — Machine Vision course, student ID 67340700403.

## Goal

Compare lane detection methods (YOLO, Pure Vision/OpenCV) → find ego-lane center → Pure Pursuit controller → test across 4 weather conditions (Clear, Rain, Fog, Night).

## Progress

| Component | Status |
|---|---|
| Dataset collection (2000 img, clear weather, low graphic) | ✅ Done |
| YOLO training (`yolo26l-seg.pt`, RTX 5070 Ti) | ✅ Done |
| Pure Vision detection (OpenCV — HSV yellow + gray Canny + Hough) | ✅ Done |
| Find center point of ego lane (both nodes publish `/lka/lane_center`) | ✅ Done |
| Pure Pursuit controller | ✅ Done |
| Evaluation across 4 weather conditions | 🔲 TODO |

## Workspace

```
lka-carla-yolo/
├── models/
│   └── best_vision.pt          # yolo26l-seg weights
├── Images/                     # 4 reference images (clear/fog/night/rain)
├── lka_ws/src/
│   ├── carla_ros/              # submodule: github.com/peeradonmoke2002/carla_ros (branch: lka)
│   ├── lka_bringup/            # CARLA bridge + vehicle spawn launch files
│   ├── lka_dataset_collection/ # Phase 1: dataset collection + roi.yaml
│   ├── lka_msgs/               # Custom message interfaces (LaneCenter.msg)
│   ├── lka_perception/         # Phase 3: YOLO node + Pure Vision node
│   └── lka_control/            # Phase 4: Pure Pursuit controller
└── training/                   # Phase 2: YOLO training scripts & weights
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

# Pure Vision node only
ros2 launch lka_perception pure_vision.launch.py

# YOLO node only
ros2 launch lka_perception yolo.launch.py

# Pure Pursuit controller
ros2 launch lka_control lka_controller.launch.py
```

## Key Facts

- **Detection classes**: `left_marking` (HSV yellow), `right_edge` (gray Canny / YOLO)
- **YOLO model**: `yolo26l-seg.pt` — weights at `/home/peeradon/lka-carla-yolo/models/best_vision.pt`
- **YOLO result**: confidence 0.85–0.99 across all 4 weather conditions
- **Controller**: Pure Pursuit (Autoware-based), `wheel_base=3.0046m`, `ld_k=2.4`
- **Dataset**: 2000 images, CARLA semantic seg class 6 (RoadLine) + sidewalk, clear weather only
- **TF publishing**: disabled — removed `sensor.pseudo.tf` from objects config in submodule
- **ROI polygon**: `[(102,892),(764,457),(825,456),(1465,894)]` — image 1600×900

## ROS2 Topics

| Topic | Type | Note |
|---|---|---|
| `/carla/ego_vehicle/CAM_FRONT/image` | `sensor_msgs/Image` | camera input |
| `/carla/weather_control` | `carla_msgs/CarlaWeatherParameters` | live weather (both nodes subscribe) |
| `/carla/ego_vehicle/odometry` | `nav_msgs/Odometry` | speed for Pure Pursuit |
| `/lka/lane_center` | `lka_msgs/LaneCenter` (center, confidence, detected, header) | from either perception node |
| `/lka/pure_vision_image` | `sensor_msgs/Image` | Pure Vision debug view |
| `/lka/enhanced_image` | `sensor_msgs/Image` | YOLO debug view |
| `/carla/ego_vehicle/vehicle_control_cmd` | `carla_msgs/CarlaEgoVehicleControl` | controller output |

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
