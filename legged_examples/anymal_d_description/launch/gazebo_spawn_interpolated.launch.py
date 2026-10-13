from bisect import bisect_right
import csv
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def heightmap_height(csv_path, x, y):
    """Bilinearly interpolate the world-frame heightmap CSV at world x,y."""
    x_values = set()
    y_values = set()
    row_count = 0
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            x_values.add(float(row["x"]))
            y_values.add(float(row["y"]))
            row_count += 1

    xs, ys = sorted(x_values), sorted(y_values)
    if len(xs) < 2 or len(ys) < 2 or len(xs) * len(ys) != row_count:
        raise ValueError(f"Heightmap CSV is not a complete rectangular grid: {csv_path}")
    if x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1]:
        raise ValueError(
            f"Spawn x,y ({x}, {y}) lies outside the heightmap extent "
            f"x=[{xs[0]}, {xs[-1]}], y=[{ys[0]}, {ys[-1]}]"
        )

    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    heights = [[None] * len(xs) for _ in ys]
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            ix = x_index[float(row["x"])]
            iy = y_index[float(row["y"])]
            heights[iy][ix] = float(row["z"])

    ix = min(bisect_right(xs, x) - 1, len(xs) - 2)
    iy = min(bisect_right(ys, y) - 1, len(ys) - 2)
    dx = (x - xs[ix]) / (xs[ix + 1] - xs[ix])
    dy = (y - ys[iy]) / (ys[iy + 1] - ys[iy])
    z00 = heights[iy][ix]
    z10 = heights[iy][ix + 1]
    z01 = heights[iy + 1][ix]
    z11 = heights[iy + 1][ix + 1]
    if None in (z00, z10, z01, z11):
        raise ValueError(f"Heightmap CSV has a missing grid sample: {csv_path}")

    return (
        z00 * (1.0 - dx) * (1.0 - dy)
        + z10 * dx * (1.0 - dy)
        + z01 * (1.0 - dx) * dy
        + z11 * dx * dy
    )


def launch_simulation(context):
    terrain = LaunchConfiguration("terrain").perform(context)
    spawn_x = float(LaunchConfiguration("spawn_x").perform(context))
    spawn_y = float(LaunchConfiguration("spawn_y").perform(context))
    clearance = float(LaunchConfiguration("spawn_clearance").perform(context))
    robot_type = LaunchConfiguration("robot_type")
    description_file = LaunchConfiguration("description_file")

    package_share = Path(get_package_share_directory("legged_gazebo"))
    world_path = package_share / "worlds" / f"forest_{terrain}.world"
    heightmap_csv = package_share / "worlds" / f"forest_{terrain}_heightmap.csv"
    if not world_path.is_file():
        raise FileNotFoundError(f"Forest world not found: {world_path}")
    if not heightmap_csv.is_file():
        raise FileNotFoundError(f"Heightmap CSV not found: {heightmap_csv}")

    ground_z = heightmap_height(heightmap_csv, spawn_x, spawn_y)
    spawn_z = ground_z + clearance
    print(
        f"Terrain {terrain}: ground z at ({spawn_x:.3f}, {spawn_y:.3f}) "
        f"is {ground_z:.3f} m; spawning at z={spawn_z:.3f} m "
        f"with {clearance:.3f} m clearance."
    )

    return [
        ExecuteProcess(
            cmd=["ros2", "run", "legged_common", "generate_urdf.sh", description_file, robot_type],
            output="screen",),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": ParameterValue(Command(["xacro ", description_file]), value_type=str),
                "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),}],),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"])),
            launch_arguments={
                "gz_args": f"-r {world_path}",
                "on_exit_shutdown": "true",
            }.items(),),
        Node(
            package="ros_gz_sim",
            executable="create",
            name="spawn_urdf",
            output="screen",
            arguments=[
                "-x", str(spawn_x), "-y", str(spawn_y), "-z", str(spawn_z),
                "-topic", "robot_description", "-name", robot_type,],),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="clock_bridge",
            output="screen",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot_type", default_value="anymal", description="Robot type"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("description_file", default_value=PathJoinSubstitution([FindPackageShare("anymal_d_description"), "urdf", "anymal.xacro"]), description="Robot xacro file",),
        DeclareLaunchArgument("terrain", default_value="terrain_01", description="Terrain name, e.g. terrain_01"),
        DeclareLaunchArgument("spawn_x", default_value="0.0", description="Spawn x coordinate in world frame (m)"),
        DeclareLaunchArgument("spawn_y", default_value="0.0", description="Spawn y coordinate in world frame (m)"),
        DeclareLaunchArgument("spawn_clearance", default_value="1.5", description="Robot origin height above the interpolated terrain (m)",),
        OpaqueFunction(function=launch_simulation),
    ])
