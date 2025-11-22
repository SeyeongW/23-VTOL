FROM ros:humble-ros-base-jammy
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-rosdep \
    python3-colcon-common-extensions \
    git \
    ros-humble-mavros \
    ros-humble-mavros-msgs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep update
WORKDIR /ros2_ws

COPY ./vtol_mission /ros2_ws/src/vtol_mission

SHELL ["/bin/bash", "-c"]

RUN source /opt/ros/humble/setup.bash && \
    rosdep install -i --from-path src --rosdistro humble -y

RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --packages-select vtol_mission

CMD ["/bin/bash", "-c", "source /ros2_ws/install/setup.bash && ros2 launch vtol_mission takeoff.launch.py"]