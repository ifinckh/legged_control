from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    terrain = LaunchConfiguration("terrain")

    world = PathJoinSubstitution([
        FindPackageShare("forma_simulation"),
        "worlds",
        ["forest_", terrain, ".world"],
    ])

    declare_terrain = DeclareLaunchArgument(
        "terrain",
        default_value="terrain_01",
        description="Terrain to load. Expected values: terrain_01, terrain_02, terrain_03.",
    )

    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world],
        output="screen",
    )

    # Start RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
    )

    return LaunchDescription([
        declare_terrain,
        gazebo,
        rviz,
    ])
