cmake_minimum_required(VERSION 3.5)
project(lka_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/lane_detection_node.py
  scripts/pure_vision_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
