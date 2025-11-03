cmake_minimum_required(VERSION 3.8)
project(auto_mission)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)

# executable
add_executable(waypoint_node src/waypoint_node.cpp)
ament_target_dependencies(waypoint_node
  rclcpp
  px4_msgs
)

# install
install(TARGETS
  waypoint_node
  DESTINATION lib/${PROJECT_NAME}
)

# launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
