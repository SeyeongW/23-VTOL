# 23-VTOL

#Terminal command
#ros2
ros2 run <package> <node>
colcon build install 

#MAVLINK
ros2 run mavros mavros_node --ros-args \
  -p fcu_url:=udp://:14557@127.0.0.1:14540 \
  -p tgt_system:=1

#PX4_sitl
make px4_sitl gazebo_standard_vtol
