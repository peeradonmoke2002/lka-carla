# lka-carla
 

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