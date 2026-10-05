from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the launch description."""
    survey_config = LaunchConfiguration('survey_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    rviz_config = PathJoinSubstitution([
        FindPackageShare('global_planner'),
        'config/rviz/',
        'view.rviz',
    ])
    default_survey_config = PathJoinSubstitution([
        FindPackageShare('global_planner'),
        'config',
        'ros.yaml',
    ])
    shutdown_timeouts = {
        'sigterm_timeout': '2.0',
        'sigkill_timeout': '2.0',
    }
    rviz_condition = IfCondition(
        PythonExpression([
            "'",
            use_rviz,
            "'.lower() == 'true'",
        ])
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'survey_config',
            default_value=default_survey_config,
            description=(
                'YAML file with survey_waypoint_node planner parameters.'
            ),
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description=(
                'Use the /clock simulation time source for all survey nodes.'
            ),
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Open RViz2 with the survey waypoint configuration.',
        ),
        Node(
            package='global_planner',
            executable='lawn_mower_node',
            name='lawn_mower_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                survey_config,
                {'use_sim_time': use_sim_time},
            ],
            **shutdown_timeouts,
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='survey_waypoints_rviz',
            output='screen',
            condition=rviz_condition,
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
            **shutdown_timeouts,
        ),
    ])
