# launch/aruco_detector.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vtol_aruco',
            executable='aruco_detector_node',
            name='aruco_detector_node',
            output='screen',
            parameters=[{
                'image_topic': '/camera/image_raw',
                'dictionary': 'DICT_4X4_50'
            }]
        )
    ])
