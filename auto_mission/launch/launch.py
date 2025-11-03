from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='auto_mission',
            executable='waypoint_node',
            name='waypoint_node',
            output='screen'
        )
    ])
