# `legged_gazebo`

The `legged_gazebo` ROS 2 package contains terrain models, Gazebo worlds, a hardware simulation plugin, and scripts for generating and visualizing forest worlds. Its terrain assets originate from the [`gazebo_terrain_generator`](https://github.com/saiaravind19/gazebo_terrain_generator) repository.

## Package layout

```text
legged_gazebo/
├── config/                     Plugin configuration
├── include/                    Hardware plugin headers
├── launch/                     ROS 2 launch descriptions
├── models/                     Terrain assets and tree models
├── scripts/                    Forest generation and plotting tools
├── src/                        Hardware plugin implementation
└── worlds/                     Empty and forest Gazebo worlds plus CSVs
```

Terrain input assets live in `models/<terrain>/`. The terrain directory and its source world filename must match, for example `models/terrain_02/terrain_02.world`. Forest worlds and their generated CSV files live in `worlds/`.

## Build and source the package

From the ROS 2 workspace root:

```bash
cd ~/forma/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select legged_gazebo --symlink-install
source install/setup.bash
```

The package installs its `launch/`, `worlds/`, `models/`, and `config/` directories under `share/legged_gazebo`. The Python scripts are run from the source tree and are not installed by this package's CMake rules.

## Add a terrain with [`gazebo_terrain_generator`](https://github.com/saiaravind19/gazebo_terrain_generator)

The terrain generator is included at [`src/gazebo_terrain_generator/`](https://github.com/saiaravind19/gazebo_terrain_generator). Before starting it, set its output base directly to this package's `models/` directory. Run these commands from `ros2_ws`:

```bash
cd src/gazebo_terrain_generator
export GAZEBO_TERRAIN_OUTPUT_PATH="$(realpath ../legged_control/legged_gazebo/models)"
uv run scripts/server.py
```

Then use the generator's browser interface to create the terrain. The export must be set in the same terminal before launching the server so its output is written directly under this package's `models/` directory.

Keep the output terrain name new or intentionally replaceable: the generator removes an existing output folder with the same name. Its output should have this form:

```text
src/legged_control/legged_gazebo/models/terrain_04/
├── terrain_04.world
├── model.config
└── mesh/
    ├── height_map.png
    ├── aerial.png
    └── normal_map.png
```

Return to the workspace root before running the commands below:

```bash
cd ~/forma/ros2_ws
```

## Generate a forest world

Run the forest generator with a terrain directory name. Its default terrain is `terrain_01`; specify another terrain with `--terrain`:

```bash
python3 src/legged_control/legged_gazebo/scripts/generate_forest_world.py \
  --terrain terrain_02
```

The script reads `models/terrain_02/terrain_02.world`, finds its heightmap and aerial image, places trees on the visible aerial patch, and writes:

```text
src/legged_control/legged_gazebo/worlds/forest_terrain_02.world
src/legged_control/legged_gazebo/worlds/forest_terrain_02_heightmap.csv
src/legged_control/legged_gazebo/worlds/forest_terrain_02_tree_pose.csv
src/legged_control/legged_gazebo/worlds/forest_terrain_02_patch_coord.csv
```

The CSVs contain heightmap `x,y,z` samples, tree `name,x,y,z,diameter` poses, and patch-boundary `x,y,z` samples. Tree placement is deterministic by default, with seed `42`. Choose another seed to make a different layout:

```bash
python3 src/legged_control/legged_gazebo/scripts/generate_forest_world.py \
  --terrain terrain_02 --seed 123
```

The optional `--output PATH` selects a different world output path. The corresponding CSV filenames are based on that output name. The default output world and CSVs are overwritten when the generator is rerun with the same terrain and output path.

The generator requires Python packages NumPy and Pillow in the Python environment used to run it.

## Plot the terrain and forest (Optionnal)

After generating the forest CSVs, run the plotting script with the forest world's path stem. The `.world` extension is optional:

```bash
python3 src/legged_control/legged_gazebo/scripts/plot_forest_world_3d.py \
  src/legged_control/legged_gazebo/worlds/forest_terrain_02
```

The plotter reads the heightmap, tree-pose, and patch-coordinate CSVs. It opens a 3D terrain surface with a height colorbar, displays tree positions in red on the surface, and draws the patch boundary as a line over the terrain. The boundary line is assembled from sampled boundary coordinates and may look jagged for irregular patches.

The plotter requires NumPy and Matplotlib in the Python environment used to run it.

## Launch Gazebo

The launch file chooses `worlds/forest_<terrain>.world` from the `terrain` argument. The default is `terrain_01`:

```bash
ros2 launch legged_gazebo forest_sim.launch.py
```

Select another installed forest world like this:

```bash
ros2 launch legged_gazebo forest_sim.launch.py terrain:=terrain_02
```

Currently, the launch file documents `terrain_01`, `terrain_02`, and `terrain_03`. If you add a terrain and generate a new forest world, rebuild the package so the new model and world are installed, then source the workspace again before launching:

```bash
colcon build --packages-select legged_gazebo --symlink-install
source install/setup.bash
ros2 launch legged_gazebo forest_sim.launch.py terrain:=terrain_04
```

To start a world directly in Gazebo without the ROS 2 launch file, use its installed path:

```bash
gz sim "$(ros2 pkg prefix legged_gazebo)/share/legged_gazebo/worlds/forest_terrain_02.world"
```
