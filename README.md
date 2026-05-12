# lka-carla
 




## How to Train SCNN

> For compatibility fixes needed after cloning pytorch-auto-drive, see [training/SCNN_SETUP.md](training/SCNN_SETUP.md).

1. Git clone the SCNN repository

```bash
cd lka-carla
git clone https://github.com/voldemortX/pytorch-auto-drive.git
```

2. Convert YOLO dataset to SCNN format

```bash
python3 training/yolo2scnn.py
# Output: lka.scnn/
```

3. Register `LkaAsSegmentation` dataset class in pytorch-auto-drive

In `pytorch-auto-drive/utils/datasets/lane_as_segmentation.py` — add before the LLAMAS class:

```python
@DATASETS.register()
class LkaAsSegmentation(_StandardLaneDetectionDataset):
    colors = [[0, 0, 0], [255, 255, 0], [128, 128, 128]]

    def init_dataset(self, root):
        self.image_dir = root
        self.mask_dir = root
        self.splits_dir = os.path.join(root, 'list')
        self.output_prefix = './output'
        self.output_suffix = '.lines.txt'
        self.image_suffix = ''
        if not os.path.exists(self.output_prefix):
            os.makedirs(self.output_prefix)

    def _init_all(self):
        split_map = {'train': 'train_gt.txt', 'val': 'val_gt.txt'}
        split_f = os.path.join(self.splits_dir, split_map.get(self.image_set, self.image_set + '.txt'))
        with open(split_f, 'r') as f:
            contents = [x.strip() for x in f.readlines() if x.strip()]
        parts_list = [x.split() for x in contents]
        self.images = [os.path.join(self.image_dir, p[0]) for p in parts_list]
        self.masks  = [os.path.join(self.image_dir, p[1]) for p in parts_list]
        if self.test == 0:
            self.lane_existences = [[int(p[2]), int(p[3])] for p in parts_list]
```

In `pytorch-auto-drive/utils/datasets/__init__.py` — update the import line:

```python
from .lane_as_segmentation import TuSimpleAsSegmentation, CULaneAsSegmentation, LLAMAS_AsSegmentation, LkaAsSegmentation
```

4. Copy Config SCNN file to the SCNN repository

```bash
cp training/scnn_lka_config.py \
   pytorch-auto-drive/configs/lane_detection/scnn/resnet18_lka.py
```

5. Start training

```bash
cd pytorch-auto-drive
python main_landet.py --config configs/lane_detection/scnn/resnet18_lka.py --train
```

6. Test after training

```bash
python main_landet.py --config configs/lane_detection/scnn/resnet18_lka.py --test
```


## How to run
1. Start CARLA simulator

```bash
cd carla
./CarlaUE4.sh -prefernvidia -RenderOffScreen
```

2. Start CarLA ROS bridge

```bash
cd lka-carla-yolo/lka_ws
source install/setup.bash && ros2 launch lka_bringup bring_up_carla.launch.py
```

3. Start Gt node (for evaluation)

```bash
source install/setup.bash &&
ros2 launch lka_perception gt.launch.py
```

4. Start perception nodes

```bash
# Pure Vision node only
source install/setup.bash &&
ros2 launch lka_perception pure_vision.launch.py
# YOLO node only
source install/setup.bash &&
ros2 launch lka_perception yolo.launch.py
# SCNN node only
source install/setup.bash &&
ros2 launch lka_perception scnn.launch.py   
# All perception nodes
source install/setup.bash &&
ros2 launch lka_perception run_all_perception.launch.py
```

5. Start Save bag node (for evaluation)

```bash
source install/setup.bash && ros2 launch lka_bringup record_bag.launch.py
```


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