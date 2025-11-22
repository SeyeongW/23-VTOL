cmake_minimum_required(VERSION 3.8)
project(vtol_mission)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(vtol_mission src/vtol_mission.cpp)

ament_target_dependencies(vtol_mission
  rclcpp
  mavros_msgs
  geometry_msgs
  std_msgs
)

install(TARGETS
  vtol_mission
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()