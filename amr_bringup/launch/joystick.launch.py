from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='joy', executable='joy_node', name='joy_node'),
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            parameters=[{
                'require_enable_button': False,
                'axis_linear.x': 1,
                'axis_angular.yaw': 3,
                'enable_turbo_button': 5,
                'scale_linear.x': 0.1,
                'scale_linear_turbo.x': 0.2,
                'scale_angular.yaw': 0.3,
                'scale_angular_turbo.yaw': 0.6,
                'publish_stamped_twist': False,  # Must match amr_base use_stamped_cmd_vel.
            }],
        ),
    ])
