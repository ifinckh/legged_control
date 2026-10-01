#!/usr/bin/env sh

set -eu

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <description.urdf|description.xacro> <robot_type>" >&2
  exit 2
fi

description_file=$1
robot_type=$2
output_dir=/tmp/legged_control
output_file="$output_dir/$robot_type.urdf"
temporary_file="$output_file.tmp.$$"

mkdir -p "$output_dir"
trap 'rm -f "$temporary_file"' EXIT HUP INT TERM

# A plain URDF is valid input to xacro. Do not pass the Unitree-specific
# robot_type mapping in that case: ordinary URDF files do not declare it.
case "$description_file" in
  *.xacro)
    ros2 run xacro xacro "$description_file" "robot_type:=$robot_type" > "$temporary_file"
    ;;
  *)
    ros2 run xacro xacro "$description_file" > "$temporary_file"
    ;;
esac

# Do not leave a truncated controller model behind when xacro fails.
mv "$temporary_file" "$output_file"
trap - EXIT HUP INT TERM
