cmake_minimum_required(VERSION 3.8)
project(vtol_mission)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(basic_takeoff_land_node src/basic_takeoff_land_node.cpp)
ament_target_dependencies(basic_takeoff_land_node
  rclcpp
  mavros_msgs
  geometry_msgs
  std_msgs
)

install(TARGETS
  basic_takeoff_land_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
