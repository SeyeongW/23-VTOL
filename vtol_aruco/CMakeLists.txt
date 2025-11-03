cmake_minimum_required(VERSION 3.8)
project(vtol_aruco)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)  # must have opencv_contrib (aruco)

add_executable(aruco_detector_node src/aruco_detector_node.cpp)
ament_target_dependencies(aruco_detector_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
)

target_link_libraries(aruco_detector_node
  ${OpenCV_LIBS}
)

install(TARGETS
  aruco_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
