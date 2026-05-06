cmake_minimum_required(VERSION 3.5)
project(lka_dataset_collection)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/dataset_collector_node.py
  scripts/save_sem_sample.py
  scripts/set_roi.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
