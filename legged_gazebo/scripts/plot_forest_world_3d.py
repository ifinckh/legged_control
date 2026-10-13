#!/usr/bin/env python3
"""Display a forest world from its generated heightmap, tree, and patch CSVs."""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader)


def sample_grid(grid, xs, ys, x, y):
    """Bilinearly sample a regular heightmap grid at world x,y coordinates."""
    x = np.clip(x, xs[0], xs[-1])
    y = np.clip(y, ys[0], ys[-1])
    ix = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    iy = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(ys) - 2)
    dx = (x - xs[ix]) / (xs[ix + 1] - xs[ix])
    dy = (y - ys[iy]) / (ys[iy + 1] - ys[iy])
    return (
        grid[iy, ix] * (1 - dx) * (1 - dy)
        + grid[iy, ix + 1] * dx * (1 - dy)
        + grid[iy + 1, ix] * (1 - dx) * dy
        + grid[iy + 1, ix + 1] * dx * dy
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot the terrain surface, patch boundary, and tree positions."
    )
    parser.add_argument(
        "world_stem",
        type=Path,
        help="World path without .world (e.g. worlds/forest_terrain_03).",
    )
    args = parser.parse_args()
    stem = args.world_stem
    if stem.suffix == ".world":
        stem = stem.with_suffix("")

    heightmap_path = stem.with_name(f"{stem.name}_heightmap.csv")
    trees_path = stem.with_name(f"{stem.name}_tree_pose.csv")
    boundary_path = stem.with_name(f"{stem.name}_patch_coord.csv")
    for path in (heightmap_path, trees_path, boundary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required CSV not found: {path}")

    terrain_rows = read_csv(heightmap_path)
    tree_rows = read_csv(trees_path)
    boundary_rows = read_csv(boundary_path)
    x_values = np.array([float(row["x"]) for row in terrain_rows])
    y_values = np.array([float(row["y"]) for row in terrain_rows])
    z_values = np.array([float(row["z"]) for row in terrain_rows])
    xs, ys = np.unique(x_values), np.unique(y_values)
    if len(xs) * len(ys) != len(terrain_rows):
        raise ValueError("Heightmap CSV does not contain a complete rectangular grid")
    order = np.lexsort((x_values, y_values))
    surface = z_values[order].reshape(len(ys), len(xs))
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface_plot = ax.plot_surface(
        X, Y, surface, cmap="terrain", linewidth=0,
        antialiased=True, rstride=max(1, len(ys) // 120),
        cstride=max(1, len(xs) // 120), alpha=0.9,
    )
    fig.colorbar(surface_plot, ax=ax, shrink=0.65, pad=0.1, label="Height (m)")
    if boundary_rows:
        boundary_x = np.array([float(row["x"]) for row in boundary_rows])
        boundary_y = np.array([float(row["y"]) for row in boundary_rows])
        # Boundary CSV samples are not stored in contour order. Angular order
        # around their centroid gives a continuous outline for this patch.
        center_x, center_y = boundary_x.mean(), boundary_y.mean()
        boundary_order = np.argsort(
            np.arctan2(boundary_y - center_y, boundary_x - center_x)
        )
        boundary_x, boundary_y = boundary_x[boundary_order], boundary_y[boundary_order]
        boundary_z = sample_grid(surface, xs, ys, boundary_x, boundary_y) + 0.04
        ax.plot(np.r_[boundary_x, boundary_x[0]],
                np.r_[boundary_y, boundary_y[0]],
                np.r_[boundary_z, boundary_z[0]],
                color="black", linewidth=1.6, label="Patch boundary")
    if tree_rows:
        tree_x = np.array([float(row["x"]) for row in tree_rows])
        tree_y = np.array([float(row["y"]) for row in tree_rows])
        tree_z = sample_grid(surface, xs, ys, tree_x, tree_y) + 0.06
        ax.scatter(tree_x, tree_y, tree_z, color="red", s=24,
                   depthshade=False, label="Tree positions")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(stem.name)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
