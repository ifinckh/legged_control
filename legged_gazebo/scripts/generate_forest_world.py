#!/usr/bin/env python3
"""Add generated trees to a terrain world while preserving its contents.

The selected source world is read from ``models/<terrain>/``. Its text,
comments, plugins, GUI settings, and other world settings are retained. The
heightmap visual and collision offsets are aligned, the aerial mask uses its
declared texture scale, local asset paths are adjusted for ``worlds/``, and
    generated tree models are appended. CSV files with terrain and tree
    groundtruth are written beside the output world.

Examples:
    python3 scripts/generate_forest_world_2.py
    python3 scripts/generate_forest_world_2.py --terrain terrain_02
"""

import argparse
import csv
import math
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PACKAGE_ROOT / "models"
WORLDS_DIR = PACKAGE_ROOT / "worlds"

DEFAULT_TERRAIN = "terrain_01"
DEFAULT_SEED = 42
TREE_HEIGHT = 7.0
TREE_MIN_RADIUS = 0.12
TREE_MAX_RADIUS = 0.30
GRID_SPACING = 6.0
POSITION_SIGMA = 2.0
BOUNDARY_MARGIN = 0.5
TREE_GROUND_PENETRATION = 0.5


def parse_terrain_info(world_path):
    """Read terrain dimensions, assets, offset, and aerial texture scale."""
    root = ET.parse(world_path).getroot()
    heightmap = next(
        (item for item in root.findall(".//visual/geometry/heightmap")
         if item.findtext("uri", "").strip().endswith("height_map.png")),
        None,
    )
    if heightmap is None:
        raise ValueError(f"No visual heightmap found in {world_path}")

    uri = heightmap.findtext("uri", "").strip()
    size = heightmap.findtext("size", "").split()
    if not uri or len(size) != 3:
        raise ValueError(f"Invalid heightmap uri or size in {world_path}")
    heightmap_path = (world_path.parent / uri).resolve()
    if not heightmap_path.is_file():
        raise FileNotFoundError(f"Heightmap does not exist: {heightmap_path}")

    texture = heightmap.find("texture")
    aerial_uri = texture.findtext("diffuse", "").strip() if texture is not None else ""
    if not aerial_uri:
        raise ValueError(f"Heightmap visual has no aerial texture in {world_path}")
    aerial_path = (world_path.parent / aerial_uri).resolve()
    if not aerial_path.is_file():
        raise FileNotFoundError(f"Aerial texture does not exist: {aerial_path}")
    texture_size = float(texture.findtext("size", "0").strip())
    if texture_size <= 0:
        raise ValueError(f"Invalid aerial texture size in {world_path}")

    position = heightmap.findtext("pos", "0 0 0").split()
    if len(position) != 3:
        raise ValueError(f"Expected three values in heightmap <pos> in {world_path}")
    origin_x, origin_y, _origin_z = map(float, position)
    return (
        heightmap_path, aerial_path, *map(float, size), texture_size,
        origin_x, origin_y,
    )


def load_heightmap(path):
    """Load a heightmap and normalize its values to [0, 1]."""
    array = np.asarray(Image.open(path), dtype=np.float32)
    if array.ndim == 3:
        array = array[..., 0]
    array -= array.min()
    maximum = array.max()
    if maximum > 0:
        array /= maximum
    return array


def load_aerial_mask(path):
    """Mark aerial pixels that belong to the terrain patch, excluding gray fill."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    return np.any(image != 128, axis=2)


def write_heightmap_csv(path, heightmap, size_x, size_y, size_z, origin_x, origin_y):
    """Write every heightmap sample as a world-frame x,y,z row."""
    height, width = heightmap.shape
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("x", "y", "z"))
        for pixel_y in range(height):
            local_y = size_y / 2.0 - pixel_y * size_y / (height - 1)
            world_y = origin_y + local_y
            for pixel_x in range(width):
                local_x = -size_x / 2.0 + pixel_x * size_x / (width - 1)
                world_x = origin_x + local_x
                world_z = float(heightmap[pixel_y, pixel_x]) * size_z
                writer.writerow((f"{world_x:.6f}", f"{world_y:.6f}", f"{world_z:.6f}"))


def write_tree_poses_csv(path, trees):
    """Write tree model poses and trunk diameters as CSV groundtruth."""
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("name", "x", "y", "z", "diameter"))
        for tree in trees:
            writer.writerow((
                tree["name"], f'{tree["x"]:.6f}', f'{tree["y"]:.6f}',
                f'{tree["z"]:.6f}', f'{2.0 * tree["radius"]:.6f}',
            ))


def write_patch_boundary_csv(path, aerial_mask, heightmap, size_x, size_y, size_z,
                            texture_size, origin_x, origin_y):
    """Write world-frame coordinates of aerial-mask boundary pixels.

    Boundary samples are the centers of patch pixels that touch a gray pixel
    or the edge of the aerial image. Their coordinates use the same UV scale
    as the tree mask mapping.
    """
    height, width = aerial_mask.shape
    # Ignore parts of the source image beyond the terrain's sampled UV extent.
    pixel_x = np.arange(width) / (width - 1) * texture_size
    pixel_y = np.arange(height) / (height - 1) * texture_size
    visible = (
        (pixel_x[None, :] <= size_x)
        & (pixel_y[:, None] <= size_y)
    )
    patch_mask = aerial_mask & visible
    padded = np.pad(patch_mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary_y, boundary_x = np.nonzero(patch_mask & ~interior)

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("x", "y", "z"))
        for pixel_x, pixel_y in zip(boundary_x, boundary_y):
            local_x = pixel_x / (width - 1) * texture_size - size_x / 2.0
            local_y = size_y / 2.0 - pixel_y / (height - 1) * texture_size
            world_x, world_y = origin_x + local_x, origin_y + local_y
            z = terrain_height(local_x, local_y, heightmap, size_x, size_y, size_z)
            writer.writerow((f"{world_x:.6f}", f"{world_y:.6f}", f"{z:.6f}"))


def world_to_pixel(x, y, width, height, size_x, size_y):
    return (x / size_x + 0.5) * (width - 1), (0.5 - y / size_y) * (height - 1)


def terrain_to_aerial_pixel(x, y, width, height, size_x, size_y, texture_size):
    """Map terrain-local coordinates through Ogre2's heightmap UV transform.

    Ogre2 applies the aerial texture from UV offset (0, 0), scaled by
    (terrain_size_x / texture_size, terrain_size_y / texture_size). This
    means that a terrain dimension smaller than the texture size samples only
    part of the source image from its UV origin; it does not center the crop.
    """
    u = (x + size_x / 2.0) / texture_size
    v = (size_y / 2.0 - y) / texture_size
    return u * (width - 1), v * (height - 1)


def bilinear_sample(array, px, py):
    height, width = array.shape
    px = min(max(px, 0.0), width - 1.0)
    py = min(max(py, 0.0), height - 1.0)
    x0, y0 = int(math.floor(px)), int(math.floor(py))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    dx, dy = px - x0, py - y0
    top = array[y0, x0] * (1.0 - dx) + array[y0, x1] * dx
    bottom = array[y1, x0] * (1.0 - dx) + array[y1, x1] * dx
    return float(top * (1.0 - dy) + bottom * dy)


def terrain_height(x, y, heightmap, size_x, size_y, size_z):
    height, width = heightmap.shape
    px, py = world_to_pixel(x, y, width, height, size_x, size_y)
    return bilinear_sample(heightmap, px, py) * size_z


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def sample_grid_centers(size_x, size_y, spacing):
    xs, ys = [], []
    x = -size_x / 2.0 + spacing / 2.0
    while x < size_x / 2.0:
        xs.append(x)
        x += spacing
    y = -size_y / 2.0 + spacing / 2.0
    while y < size_y / 2.0:
        ys.append(y)
        y += spacing
    return xs, ys


def generate_tree_data(
    size_x, size_y, size_z, texture_size, heightmap, aerial_mask,
    origin_x, origin_y, seed,
):
    """Generate trees only on the aerial patch, rooted slightly below its surface."""
    rng = random.Random(seed)
    xs, ys = sample_grid_centers(size_x, size_y, GRID_SPACING)
    x_min, x_max = -size_x / 2.0 + BOUNDARY_MARGIN, size_x / 2.0 - BOUNDARY_MARGIN
    y_min, y_max = -size_y / 2.0 + BOUNDARY_MARGIN, size_y / 2.0 - BOUNDARY_MARGIN
    half_cell = GRID_SPACING / 2.0
    trees = []
    for index, (y0, x0) in enumerate(((y, x) for y in ys for x in xs), start=1):
        x = clamp(rng.gauss(x0, POSITION_SIGMA), x0 - half_cell, x0 + half_cell)
        y = clamp(rng.gauss(y0, POSITION_SIGMA), y0 - half_cell, y0 + half_cell)
        x, y = clamp(x, x_min, x_max), clamp(y, y_min, y_max)

        # Use the same zero-offset UV scale as Ogre2Heightmap. In particular,
        # this crops the aerial mask from its UV origin when size_y is smaller
        # than texture_size, matching the image visible on the terrain.
        pixel_x, pixel_y = terrain_to_aerial_pixel(
            x, y, aerial_mask.shape[1], aerial_mask.shape[0],
            size_x, size_y, texture_size,
        )
        mask_x, mask_y = int(round(pixel_x)), int(round(pixel_y))
        if not (0 <= mask_x < aerial_mask.shape[1] and 0 <= mask_y < aerial_mask.shape[0]):
            continue
        if not aerial_mask[mask_y, mask_x]:
            continue

        radius = rng.uniform(TREE_MIN_RADIUS, TREE_MAX_RADIUS)
        yaw = rng.uniform(0.0, 2.0 * math.pi)
        ground_z = terrain_height(x, y, heightmap, size_x, size_y, size_z)
        # align_heightmap_offsets places the output heightmap base at z=0.
        ground_z -= TREE_GROUND_PENETRATION
        trees.append({
            "name": f"tree_{len(trees) + 1:04d}",
            "x": x + origin_x,
            "y": y + origin_y,
            "z": ground_z + TREE_HEIGHT / 2.0,
            "radius": radius, "height": TREE_HEIGHT, "yaw": yaw,
        })
    return trees


def text_element(parent, tag, value):
    element = ET.SubElement(parent, tag)
    element.text = str(value)
    return element


def add_tree_model(model, tree):
    text_element(model, "static", "true")
    text_element(model, "pose", f'{tree["x"]:.4f} {tree["y"]:.4f} {tree["z"]:.4f} 0 0 {tree["yaw"]:.6f}')
    link = ET.SubElement(model, "link", {"name": "trunk"})
    visual = ET.SubElement(link, "visual", {"name": "trunk_visual"})
    cylinder = ET.SubElement(ET.SubElement(visual, "geometry"), "cylinder")
    text_element(cylinder, "radius", f'{tree["radius"]:.4f}')
    text_element(cylinder, "length", f'{tree["height"]:.4f}')
    material = ET.SubElement(visual, "material")
    text_element(material, "ambient", "0.22 0.08 0.025 1")
    text_element(material, "diffuse", "0.40 0.14 0.04 1")
    text_element(material, "specular", "0.05 0.05 0.05 1")
    collision = ET.SubElement(link, "collision", {"name": "trunk_collision"})
    cylinder = ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder")
    text_element(cylinder, "radius", f'{tree["radius"]:.4f}')
    text_element(cylinder, "length", f'{tree["height"]:.4f}')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Append trees to an existing terrain world with minimal edits."
    )
    parser.add_argument(
        "--terrain",
        default=DEFAULT_TERRAIN,
        help="Terrain directory inside models/ (default: terrain_01).",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: worlds/forest_<terrain>.world).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Tree placement seed (default: {DEFAULT_SEED}).")
    return parser.parse_args()


def align_heightmap_offsets(source, origin_x, origin_y):
    """Keep visual and collision x/y origins aligned and normalize both z offsets."""
    root = ET.fromstring(source)
    collision = next(
        (item for item in root.findall(".//collision")
         if item.find("./geometry/heightmap") is not None),
        None,
    )
    if collision is None:
        raise ValueError("Source world has no heightmap collision")
    pose = collision.find("pose")
    if pose is None or not pose.text:
        raise ValueError("Heightmap collision has no <pose>")

    fields = pose.text.split()
    if len(fields) != 6:
        raise ValueError("Expected six values in heightmap collision <pose>")
    fields[0:3] = [f"{origin_x:g}", f"{origin_y:g}", "0"]

    # Change just the pose text in the original source, retaining all other
    # whitespace and comments. Match the collision's original pose exactly.
    old_pose = pose.text
    new_pose = " ".join(fields)
    pattern = re.compile(r"(<collision\b[^>]*>(?:(?!</collision>).)*?<pose\s*>\s*)"
                         + re.escape(old_pose.strip())
                         + r"(\s*</pose>)", re.DOTALL)
    updated, count = pattern.subn(lambda match: match.group(1) + new_pose + match.group(2), source, count=1)
    if count != 1:
        raise ValueError("Could not safely locate heightmap collision pose in source text")

    # The visual heightmap uses <heightmap><pos>, while the collision uses
    # <collision><pose>. Keep their horizontal origins identical and put both
    # z offsets at zero so the sampled tree heights use the same vertical base.
    visual_pos = re.compile(
        r"(<visual\b[^>]*>(?:(?!</visual>).)*?<heightmap\b[^>]*>"
        r"(?:(?!</heightmap>).)*?<pos\s*>\s*)[^<]+(\s*</pos>)",
        re.DOTALL,
    )
    updated, count = visual_pos.subn(
        lambda match: match.group(1) + f"{origin_x:g} {origin_y:g} 0.0" + match.group(2),
        updated,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not safely locate visual heightmap <pos>")
    return updated


def append_trees(source, trees):
    """Serialize generated models and insert them immediately before </world>."""
    tree = ET.fromstring(source)
    world = tree.find("world")
    if world is None:
        raise ValueError("Source SDF has no <world>")

    models = []
    for item in trees:
        model = ET.Element("model", {"name": item["name"]})
        add_tree_model(model, item)
        models.append(ET.tostring(model, encoding="unicode"))

    closing = re.search(r"</world\s*>", source)
    if closing is None:
        raise ValueError("Could not locate closing </world> tag")
    addition = "\n\n" + "\n\n".join("    " + line for model in models
                                                   for line in model.splitlines()) + "\n"
    return source[:closing.start()] + addition + source[closing.start():]


def main():
    args = parse_args()
    terrain_dir = Path(args.terrain)
    if not terrain_dir.is_absolute():
        terrain_dir = MODELS_DIR / terrain_dir
    terrain_dir = terrain_dir.resolve()

    if not terrain_dir.is_dir():
        raise FileNotFoundError(f"Terrain directory does not exist: {terrain_dir}")
    terrain_name = terrain_dir.name
    source_path = terrain_dir / f"{terrain_name}.world"
    if not source_path.is_file():
        raise FileNotFoundError(
            "Expected the canonical terrain world at "
            f"{source_path} (world filename must match its terrain directory)."
        )

    output_path = args.output or (WORLDS_DIR / f"forest_{terrain_name}.world")
    if not output_path.is_absolute():
        output_path = PACKAGE_ROOT / output_path
    output_path = output_path.resolve()

    source = source_path.read_text(encoding="utf-8")
    (
        heightmap_path, aerial_path, size_x, size_y, size_z, texture_size,
        origin_x, origin_y,
    ) = parse_terrain_info(source_path)
    heightmap = load_heightmap(heightmap_path)
    aerial_mask = load_aerial_mask(aerial_path)
    trees = generate_tree_data(
        size_x, size_y, size_z, texture_size, heightmap, aerial_mask,
        origin_x, origin_y, args.seed,
    )

    source = align_heightmap_offsets(source, origin_x, origin_y)
    # Relative source URIs resolve from models/<terrain>; output URIs must
    # instead resolve from worlds/.
    # These paths are relative to the source model directory. The output is
    # in worlds/, so point all source terrain assets back into models/<name>.
    source = re.sub(
        r"<(uri|diffuse|normal)>\s*mesh/([^<]+)</(uri|diffuse|normal)>",
        lambda match: (
            f"<{match.group(1)}>../models/{terrain_dir.name}/mesh/{match.group(2)}"
            f"</{match.group(3)}>"
        ),
        source,
    )
    source = append_trees(source, trees)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8")
    heightmap_csv = output_path.with_name(f"{output_path.stem}_heightmap.csv")
    tree_poses_csv = output_path.with_name(f"{output_path.stem}_tree_pose.csv")
    patch_coord_csv = output_path.with_name(f"{output_path.stem}_patch_coord.csv")
    write_heightmap_csv(
        heightmap_csv, heightmap, size_x, size_y, size_z, origin_x, origin_y
    )
    write_tree_poses_csv(tree_poses_csv, trees)
    write_patch_boundary_csv(
        patch_coord_csv, aerial_mask, heightmap, size_x, size_y, size_z,
        texture_size, origin_x, origin_y,
    )
    print(f"Source world:  {source_path}")
    print(f"Heightmap:     {heightmap_path}")
    print(f"Trees:         {len(trees)}")
    print(f"Wrote:         {output_path}")
    print(f"Heightmap CSV: {heightmap_csv}")
    print(f"Tree poses:    {tree_poses_csv}")
    print(f"Patch coords:  {patch_coord_csv}")


if __name__ == "__main__":
    main()
