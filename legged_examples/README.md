# Legged examples: ANYmal D

This guide covers launching the ANYmal D Gazebo simulation from
`anymal_d_description`.

## Build and source

From the repository root, build the package and its workspace dependencies if
they have not been built yet:

```bash
source /opt/ros/jazzy/setup.bash
cd ros2_ws
colcon build --symlink-install --packages-up-to anymal_d_description
source install/setup.bash
```

In a new terminal opened at the repository root, source the ROS and workspace
setup files before using the launch command:

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

## Launch the simulation

For example, launch the robot on terrain 02:

```bash
ros2 launch anymal_d_description gazebo_spawn_interpolated.launch.py terrain:=terrain_02
```

The launch starts Gazebo, publishes the robot description, and spawns ANYmal D
at the requested terrain location. It also generates the controller URDF at
`/tmp/legged_control/anymal.urdf`.

The launch arguments are:

| Argument | Default | Purpose |
| --- | --- | --- |
| `terrain` | `terrain_01` | Terrain world; available names are `terrain_01`, `terrain_02`, and `terrain_03`. |
| `spawn_x` | `0.0` | Spawn x coordinate in the world frame, in metres. |
| `spawn_y` | `0.0` | Spawn y coordinate in the world frame, in metres. |
| `spawn_clearance` | `1.5` | Height above the interpolated terrain at the spawn point, in metres. |
| `use_sim_time` | `true` | Use the Gazebo simulation clock. |

For example, to choose another spawn point and clearance:

```bash
ros2 launch anymal_d_description gazebo_spawn_interpolated.launch.py \
  terrain:=terrain_02 spawn_x:=1.0 spawn_y:=0.5 spawn_clearance:=1.5
```
