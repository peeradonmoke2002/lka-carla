# lka-carla-yolo
 


## How to run
1. Start CARLA simulator

```bash
cd carla
./CarlaUE4.sh -prefernvidia -RenderOffScreen
```

2. Start CarLA ROS bridge

```bash
cd lka-carla-yolo/lka_ws
source install/setup.bash
ros2 launch lka_bringup bring_up_carla.launch.py
```


3. Start perception nodes

```bash
# Pure Vision node only
source install/setup.bash &&
ros2 launch lka_perception pure_vision.launch.py
# YOLO node only
source install/setup.bash &&
ros2 launch lka_perception yolo.launch.py
```