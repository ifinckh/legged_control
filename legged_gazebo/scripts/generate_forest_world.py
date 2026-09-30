#!/usr/bin/env python3
"""
Generate a complete Gazebo Sim forest world from one of the fixed terrain
directories.

Expected project structure:

forma_simulation/
├── models/
│   ├── textures/
│   │   ├── dirt_diffuse.png
│   │   ├── grass_diffuse.png
│   │   ├── rock_diffuse.png
│   │   ├── flat_normal.png
│   │   └── bark_diffuse.png
│   ├── terrain_01/
│   │   ├── model.sdf
│   │   └── height_map.png
│   ├── terrain_02/
│   │   ├── model.sdf
│   │   └── height_map.png
│   └── terrain_03/
│       ├── model.sdf
│       └── height_map.png
├── scripts/
│   └── generate_forest_world.py
└── worlds/

The important design choice here is that the terrain visual/collision is
copied from the selected terrain's model.sdf. This preserves the existing
heightmap material, including its multiple elevation textures and blending,
instead of replacing it with a new PBR material.

Trees are currently simple brown cylinders:
- height fixed at 5 m by default
- radius varies between 0.12 and 0.28 m by default
- deterministic placement with a configurable seed

Examples:
    python3 scripts/generate_forest_world.py
    python3 scripts/generate_forest_world.py --terrain terrain_02
    python3 scripts/generate_forest_world.py --terrain terrain_03 --seed 123
"""

import argparse
import copy
import math
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PACKAGE_ROOT / "models"
WORLDS_DIR = PACKAGE_ROOT / "worlds"

DEFAULT_TERRAIN = "terrain_01"
DEFAULT_SEED = 42

# Tree parameters
TREE_HEIGHT = 7.0
TREE_MIN_RADIUS = 0.12
TREE_MAX_RADIUS = 0.30

# Placement
GRID_SPACING = 6.0
POSITION_SIGMA = 2.0
BOUNDARY_MARGIN = 0.5


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def text_element(parent, tag, text):
    element = ET.SubElement(parent, tag)
    element.text = str(text)
    return element


def indent_xml(element, level=0):
    """Pretty-print an XML tree using only the Python standard library."""
    indentation = "\n" + level * "  "

    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "

        for child in element:
            indent_xml(child, level + 1)

        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = indentation
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


# ---------------------------------------------------------------------------
# Terrain discovery
# ---------------------------------------------------------------------------

def resolve_terrain_directory(argument):
    """Resolve a terrain directory and its generated .world file."""
    path = Path(argument)
    if not path.is_absolute():
        path = MODELS_DIR / path
    path = path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(f"Terrain directory does not exist:\n  {path}")

    preferred = path / f"{path.name}.world"
    if preferred.is_file():
        world_file = preferred
    else:
        candidates = sorted(path.glob("*.world"))
        if not candidates:
            raise FileNotFoundError(
                f"Terrain directory does not contain a .world file:\n  {path}"
            )
        world_file = candidates[0]

    return path, world_file


def resolve_local_uri(model_sdf_path, uri):
    """Resolve a local/file:// or model:// URI to a file on disk."""
    uri = uri.strip()

    if uri.startswith("file://"):
        return Path(uri[7:]).expanduser().resolve()

    if uri.startswith("model://"):
        remainder = uri[len("model://"):]
        parts = remainder.split("/", 1)

        if len(parts) != 2:
            raise ValueError(f"Unsupported model URI: {uri}")

        model_name, relative_path = parts
        candidate = MODELS_DIR / model_name / relative_path

        if candidate.is_file():
            return candidate.resolve()

        raise FileNotFoundError(
            f"Could not resolve URI {uri!r}.\nTried:\n  {candidate}"
        )

    return (model_sdf_path.parent / uri).resolve()


def find_terrain_geometry(world_path):
    """Read the generated terrain world for its heightmap and collision."""
    tree = ET.parse(world_path)
    root = tree.getroot()

    visual = None
    collision = None

    for candidate in root.findall(".//visual"):
        hm = candidate.find("./geometry/heightmap")
        if hm is not None and hm.findtext("uri", "").endswith("height_map.png"):
            visual = candidate
            break

    for candidate in root.findall(".//collision"):
        hm = candidate.find("./geometry/heightmap")
        if hm is not None and hm.findtext("uri", "").endswith("height_map.png"):
            collision = candidate
            break

    if visual is None:
        raise ValueError(f"No heightmap visual found in {world_path}")

    return tree, visual.find("./geometry/heightmap"), visual, collision


def parse_terrain_info(world_path):
    """Extract the visual heightmap path and size from the generated world."""
    tree = ET.parse(world_path)
    root = tree.getroot()

    # Keep the original working coordinate convention: tree positions are
    # expressed in the terrain's local/world coordinates used by the old
    # generator. We only take the map dimensions and heightmap from the new
    # generated world file.
    heightmap = None
    for candidate in root.findall(".//visual/geometry/heightmap"):
        uri = candidate.findtext("uri", default="").strip()
        if uri.endswith("height_map.png"):
            heightmap = candidate
            break

    if heightmap is None:
        raise ValueError(f"No visual heightmap found in {world_path}")

    uri = heightmap.findtext("uri")
    size = heightmap.findtext("size")

    if not uri or not size:
        raise ValueError(f"Visual heightmap in {world_path} is missing uri or size")

    values = size.split()
    if len(values) != 3:
        raise ValueError(f"Expected 3 values in <size>, got {values}")

    terrain_size_x, terrain_size_y, terrain_size_z = map(float, values)
    heightmap_path = (world_path.parent / uri).resolve()

    if not heightmap_path.is_file():
        raise FileNotFoundError(
            f"Heightmap referenced by {world_path} does not exist:\n  {heightmap_path}"
        )

    return heightmap_path, terrain_size_x, terrain_size_y, terrain_size_z


# ---------------------------------------------------------------------------
# URI rewriting for generated worlds
# ---------------------------------------------------------------------------

def rewrite_terrain_uris(element, terrain_name, output_path):
    """
    Rewrite terrain resource references so the generated world can be
    loaded without the old ``model://forma_terrain`` model being installed.

    The selected height_map.png is put under:
        models/<terrain_name>/height_map.png

    Shared material textures are put under:
        models/textures/<filename>

    In the original heightmap material, texture paths occur in elements
    such as <diffuse> and <normal>, not only in <uri>. Therefore all
    texture-bearing tags are handled here.
    """
    texture_tags = {"uri", "diffuse", "normal"}

    for child in element.iter():
        if child.text is None:
            continue

        value = child.text.strip()

        if child.tag not in texture_tags:
            continue

        # Only rewrite actual image/resource paths.
        lower = value.lower()
        is_image = lower.endswith((
            ".png", ".jpg", ".jpeg", ".pgm", ".bmp", ".tif", ".tiff"
        ))
        is_model_texture = value.startswith("model://")

        if not is_image and not is_model_texture:
            continue

        filename = Path(value.split("?", 1)[0]).name

        if filename == "height_map.png":
            child.text = f"../models/{terrain_name}/mesh/height_map.png"
        else:
            # All terrain appearance textures are shared.
            child.text = f"../models/textures/{filename}"


def prepare_terrain_visual(source_visual, terrain_name, terrain_size_x, terrain_size_y, terrain_size_z):
    """Build the same known-working terrain visual, using the new heightmap."""
    visual = ET.Element("visual", {"name": "terrain_visual"})
    geometry = ET.SubElement(visual, "geometry")
    heightmap = ET.SubElement(geometry, "heightmap")

    for diffuse_name in ("dirt_diffuse.png", "grass_diffuse.png", "rock_diffuse.png"):
        texture = ET.SubElement(heightmap, "texture")
        text_element(texture, "diffuse", f"../models/textures/{diffuse_name}")
        text_element(texture, "normal", "../models/textures/flat_normal.png")
        text_element(texture, "size", "8")

    blend = ET.SubElement(heightmap, "blend")
    text_element(blend, "min_height", "10")
    text_element(blend, "fade_dist", "5")

    blend = ET.SubElement(heightmap, "blend")
    text_element(blend, "min_height", "25")
    text_element(blend, "fade_dist", "6")

    text_element(heightmap, "uri", f"../models/{terrain_name}/mesh/height_map.png")
    text_element(
        heightmap,
        "size",
        f"{terrain_size_x:g} {terrain_size_y:g} {terrain_size_z:g}",
    )
    text_element(heightmap, "pos", "0 0 0")

    return visual


def prepare_terrain_collision(source_collision, terrain_name):
    """Clone the original terrain collision, if one exists."""
    if source_collision is None:
        return None

    collision = copy.deepcopy(source_collision)
    rewrite_terrain_uris(collision, terrain_name, None)
    return collision


# ---------------------------------------------------------------------------
# Heightmap sampling
# ---------------------------------------------------------------------------

def load_heightmap(path):
    """Load the heightmap and normalize it to [0, 1]."""
    image = Image.open(path)
    array = np.asarray(image, dtype=np.float32)

    if array.ndim == 3:
        array = array[..., 0]

    array -= array.min()
    maximum = array.max()

    if maximum > 0:
        array /= maximum

    return array


def world_to_pixel(
    x,
    y,
    width,
    height,
    terrain_size_x,
    terrain_size_y,
):
    """Convert world coordinates to image pixel coordinates."""
    px = (x / terrain_size_x + 0.5) * (width - 1)
    py = (0.5 - y / terrain_size_y) * (height - 1)
    return px, py


def bilinear_sample(array, px, py):
    """Bilinearly sample a normalized heightmap."""
    height, width = array.shape

    px = min(max(px, 0.0), width - 1.0)
    py = min(max(py, 0.0), height - 1.0)

    x0 = int(math.floor(px))
    x1 = min(x0 + 1, width - 1)
    y0 = int(math.floor(py))
    y1 = min(y0 + 1, height - 1)

    dx = px - x0
    dy = py - y0

    v00 = array[y0, x0]
    v10 = array[y0, x1]
    v01 = array[y1, x0]
    v11 = array[y1, x1]

    v0 = v00 * (1.0 - dx) + v10 * dx
    v1 = v01 * (1.0 - dx) + v11 * dx

    return float(v0 * (1.0 - dy) + v1 * dy)


def terrain_height(
    x,
    y,
    heightmap,
    terrain_size_x,
    terrain_size_y,
    terrain_size_z,
):
    """Return terrain height at world coordinate (x, y)."""
    height, width = heightmap.shape

    px, py = world_to_pixel(
        x,
        y,
        width,
        height,
        terrain_size_x,
        terrain_size_y,
    )

    normalized = bilinear_sample(heightmap, px, py)
    return normalized * terrain_size_z


# ---------------------------------------------------------------------------
# Tree placement
# ---------------------------------------------------------------------------

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def sample_grid_centers(size_x, size_y, spacing):
    xs = []
    ys = []

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
    terrain_size_x,
    terrain_size_y,
    heightmap,
    terrain_size_z,
    seed,
):
    """Generate deterministic tree positions and variable radii."""
    rng = random.Random(seed)

    xs, ys = sample_grid_centers(
        terrain_size_x,
        terrain_size_y,
        GRID_SPACING,
    )

    x_min = -terrain_size_x / 2.0 + BOUNDARY_MARGIN
    x_max = terrain_size_x / 2.0 - BOUNDARY_MARGIN
    y_min = -terrain_size_y / 2.0 + BOUNDARY_MARGIN
    y_max = terrain_size_y / 2.0 - BOUNDARY_MARGIN

    cell_half = GRID_SPACING / 2.0
    trees = []
    index = 1

    for y0 in ys:
        for x0 in xs:
            x = clamp(
                rng.gauss(x0, POSITION_SIGMA),
                x0 - cell_half,
                x0 + cell_half,
            )
            y = clamp(
                rng.gauss(y0, POSITION_SIGMA),
                y0 - cell_half,
                y0 + cell_half,
            )

            x = clamp(x, x_min, x_max)
            y = clamp(y, y_min, y_max)

            radius = rng.uniform(
                TREE_MIN_RADIUS,
                TREE_MAX_RADIUS,
            )

            yaw = rng.uniform(0.0, 2.0 * math.pi)

            ground_z = terrain_height(
                x,
                y,
                heightmap,
                terrain_size_x,
                terrain_size_y,
                terrain_size_z,
            )

            trees.append({
                "name": f"tree_{index:04d}",
                "x": x,
                "y": y,
                "z": ground_z + TREE_HEIGHT / 2.0,
                "radius": radius,
                "height": TREE_HEIGHT,
                "yaw": yaw,
            })

            index += 1

    return trees


# ---------------------------------------------------------------------------
# Tree SDF
# ---------------------------------------------------------------------------

def add_tree_model(world, tree):
    """
    Add one top-level static tree model.

    Deliberately no PBR texture here. A simple brown material avoids
    introducing another shader/material dependency while the terrain
    rendering is being validated.
    """
    model = ET.SubElement(
        world,
        "model",
        {"name": tree["name"]},
    )

    text_element(model, "static", "true")

    text_element(
        model,
        "pose",
        (
            f'{tree["x"]:.4f} '
            f'{tree["y"]:.4f} '
            f'{tree["z"]:.4f} '
            f'0 0 {tree["yaw"]:.6f}'
        ),
    )

    link = ET.SubElement(model, "link", {"name": "trunk"})

    visual = ET.SubElement(
        link,
        "visual",
        {"name": "trunk_visual"},
    )

    geometry = ET.SubElement(visual, "geometry")
    cylinder = ET.SubElement(geometry, "cylinder")

    text_element(cylinder, "radius", f'{tree["radius"]:.4f}')
    text_element(cylinder, "length", f'{tree["height"]:.4f}')

    # Basic Gazebo material: no PBR texture/shader dependency.
    material = ET.SubElement(visual, "material")
    text_element(material, "ambient", "0.22 0.08 0.025 1")
    text_element(material, "diffuse", "0.40 0.14 0.04 1")
    text_element(material, "specular", "0.05 0.05 0.05 1")

    collision = ET.SubElement(
        link,
        "collision",
        {"name": "trunk_collision"},
    )

    collision_geometry = ET.SubElement(collision, "geometry")
    collision_cylinder = ET.SubElement(
        collision_geometry,
        "cylinder",
    )

    text_element(
        collision_cylinder,
        "radius",
        f'{tree["radius"]:.4f}',
    )
    text_element(
        collision_cylinder,
        "length",
        f'{tree["height"]:.4f}',
    )


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------

def create_world(
    terrain_name,
    source_visual,
    source_collision,
    trees,
):
    sdf = ET.Element("sdf", {"version": "1.9"})

    world = ET.SubElement(
        sdf,
        "world",
        {"name": f"forest_{terrain_name}"},
    )

    # Keep physics simple; the terrain collision itself is copied from the
    # terrain model. Gazebo/DART may still report its known heightmap
    # collision limitation; that is independent of rendering.
    physics = ET.SubElement(
        world,
        "physics",
        {
            "name": "default_physics",
            "type": "dart",
        },
    )
    text_element(physics, "max_step_size", "0.001")
    text_element(physics, "real_time_factor", "1")

    gravity = ET.SubElement(world, "gravity")
    gravity.text = "0 0 -9.81"

    # Terrain model
    terrain_model = ET.SubElement(
        world,
        "model",
        {"name": terrain_name},
    )
    text_element(terrain_model, "static", "true")

    terrain_link = ET.SubElement(
        terrain_model,
        "link",
        {"name": "terrain_link"},
    )

    terrain_link.append(source_visual)

    if source_collision is not None:
        terrain_link.append(source_collision)

    # Trees
    for tree in trees:
        add_tree_model(world, tree)

    # Sun
    light = ET.SubElement(
        world,
        "light",
        {
            "name": "sun",
            "type": "directional",
        },
    )

    text_element(light, "cast_shadows", "true")
    text_element(light, "pose", "0 0 20 0.5 0.2 0")
    text_element(light, "diffuse", "0.8 0.8 0.8 1")
    text_element(light, "specular", "0.2 0.2 0.2 1")

    direction = ET.SubElement(light, "direction")
    direction.text = "-0.3 0.1 -0.9"

    return sdf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Gazebo forest world from a fixed terrain model."
        )
    )

    parser.add_argument(
        "--terrain",
        default=DEFAULT_TERRAIN,
        help=(
            "Terrain directory inside models/ "
            "(default: terrain_01)."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output SDF path. Defaults to "
            "worlds/forest_<terrain>.world."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Tree placement/radius seed (default: {DEFAULT_SEED}).",
    )

    parser.add_argument(
        "--tree-height",
        type=float,
        default=TREE_HEIGHT,
        help=f"Tree height in metres (default: {TREE_HEIGHT}).",
    )

    parser.add_argument(
        "--min-radius",
        type=float,
        default=TREE_MIN_RADIUS,
        help=f"Minimum tree radius in metres (default: {TREE_MIN_RADIUS}).",
    )

    parser.add_argument(
        "--max-radius",
        type=float,
        default=TREE_MAX_RADIUS,
        help=f"Maximum tree radius in metres (default: {TREE_MAX_RADIUS}).",
    )

    return parser.parse_args()


def main():
    global TREE_HEIGHT
    global TREE_MIN_RADIUS
    global TREE_MAX_RADIUS

    args = parse_args()

    TREE_HEIGHT = args.tree_height
    TREE_MIN_RADIUS = args.min_radius
    TREE_MAX_RADIUS = args.max_radius

    if TREE_HEIGHT <= 0:
        raise ValueError("--tree-height must be greater than zero.")

    if TREE_MIN_RADIUS <= 0 or TREE_MAX_RADIUS <= 0:
        raise ValueError("Tree radii must be greater than zero.")

    if TREE_MIN_RADIUS > TREE_MAX_RADIUS:
        raise ValueError("--min-radius cannot be greater than --max-radius.")

    terrain_dir, terrain_world = resolve_terrain_directory(args.terrain)
    terrain_name = terrain_dir.name

    if args.output is None:
        output_path = WORLDS_DIR / f"forest_{terrain_name}.world"
    else:
        output_path = args.output
        if not output_path.is_absolute():
            output_path = PACKAGE_ROOT / output_path
        output_path = output_path.resolve()

    (
        heightmap_path,
        terrain_size_x,
        terrain_size_y,
        terrain_size_z,
    ) = parse_terrain_info(terrain_world)

    _, _, source_visual, source_collision = find_terrain_geometry(terrain_world)

    heightmap = load_heightmap(heightmap_path)

    trees = generate_tree_data(
        terrain_size_x,
        terrain_size_y,
        heightmap,
        terrain_size_z,
        args.seed,
    )

    terrain_visual = prepare_terrain_visual(
        source_visual,
        terrain_name,
        terrain_size_x,
        terrain_size_y,
        terrain_size_z,
    )

    terrain_collision = prepare_terrain_collision(
        source_collision,
        terrain_name,
    )

    world = create_world(
        terrain_name,
        terrain_visual,
        terrain_collision,
        trees,
    )

    indent_xml(world)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(world).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    print(f"Terrain:       {terrain_name}")
    print(f"Heightmap:     {heightmap_path}")
    print(
        "Terrain size:  "
        f"{terrain_size_x:g} x {terrain_size_y:g} x {terrain_size_z:g}"
    )
    print(f"Trees:         {len(trees)}")
    print(f"Tree height:   {TREE_HEIGHT:g} m")
    print(
        f"Tree radius:   {TREE_MIN_RADIUS:g} - "
        f"{TREE_MAX_RADIUS:g} m"
    )
    print(f"Seed:          {args.seed}")
    print(f"Wrote:         {output_path}")


if __name__ == "__main__":
    main()
