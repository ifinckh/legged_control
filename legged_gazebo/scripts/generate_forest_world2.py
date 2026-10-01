#!/usr/bin/env python3
"""Add simple tree trunks to an existing Gazebo terrain-generator world.

The input world is copied and modified in place structurally: plugins, physics,
lighting, GUI, terrain visual, and terrain collision are preserved. Only tree
models are appended, resource paths are made relative to the output world, and
the terrain visual can optionally be switched from satellite imagery to a
dirt/grass/rock blend.

Dependencies: Pillow and NumPy.
"""

import argparse
import math
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "worlds"
DEFAULT_TEXTURE_DIR = PACKAGE_ROOT / "models" / "textures"

GRID_SPACING = 6.0
# Standard deviation of the Gaussian jitter applied to each grid point (metres).
POSITION_SIGMA = 2.0
# Keep tree centres slightly away from the rectangular heightmap edge (metres).
BOUNDARY_MARGIN = 0.5
TREE_HEIGHT = 7.0
TREE_MIN_RADIUS = 0.12
TREE_MAX_RADIUS = 0.30


def text_element(parent: ET.Element, tag: str, value: object) -> ET.Element:
    """Create an XML child element and set its text to ``value``."""
    child = ET.SubElement(parent, tag)
    child.text = str(value)
    return child


def indent_xml(element: ET.Element, level: int = 0) -> None:
    """Indent XML while leaving existing non-whitespace text untouched."""
    indentation = "\n" + level * "  "
    if len(element):
        if element.text is None or not element.text.strip():
            element.text = indentation + "  "
        for child in element:
            indent_xml(child, level + 1)
        if element[-1].tail is None or not element[-1].tail.strip():
            element[-1].tail = indentation
    elif level and (element.tail is None or not element.tail.strip()):
        element.tail = indentation


def parse_pose(text: str | None) -> list[float]:
    """Parse an SDF six-value pose, defaulting missing values to zero."""
    values = [float(v) for v in (text or "0 0 0 0 0 0").split()]
    values += [0.0] * (6 - len(values))
    return values[:6]


def local_resource_path(world_path: Path, uri: str | None) -> Path | None:
    """Resolve a local relative/file URI; return None for model:// and other URIs."""
    uri = (uri or "").strip()
    if not uri or uri.startswith(("model://", "http://", "https://")):
        return None
    if uri.startswith("file://"):
        return Path(uri[7:]).expanduser().resolve()
    path = Path(uri).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (world_path.parent / path).resolve()


def find_heightmap(world_root: ET.Element) -> tuple[ET.Element, ET.Element, ET.Element, ET.Element, ET.Element | None]:
    """Return the terrain model/link, visual, heightmap and collision elements."""
    matches = []
    for visual in world_root.findall(".//visual"):
        heightmap = visual.find("./geometry/heightmap")
        if heightmap is not None and heightmap.findtext("uri", "").strip():
            matches.append((visual, heightmap))
    if not matches:
        raise ValueError("No visual heightmap with a <uri> was found in the input world.")
    # Prefer the common gazebo_terrain_generator heightmap filename.
    visual, heightmap = next(
        ((v, h) for v, h in matches
         if Path(h.findtext("uri", "").strip().split("?", 1)[0]).name == "height_map.png"),
        matches[0],
    )
    link = next((parent for parent in world_root.iter("link") if visual in list(parent)), None)
    model = None
    if link is not None:
        model = next((parent for parent in world_root.iter("model") if link in list(parent)), None)
    if model is None:
        raise ValueError("Could not identify the terrain model containing the heightmap visual.")

    collision = None
    if link is not None:
        for candidate in link.findall("./collision"):
            hm = candidate.find("./geometry/heightmap")
            if hm is not None:
                collision = candidate
                break
    return model, link, visual, heightmap, collision


def load_heightmap(path: Path) -> np.ndarray:
    """Load a heightmap image and normalize its pixel values to the range [0, 1]."""
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.float64)
    if array.ndim == 3:
        array = array[..., 0]
    low = float(array.min())
    high = float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    else:
        array = np.zeros_like(array, dtype=np.float64)
    return array


def load_patch_mask(
    aerial_path: Path | None,
    explicit_mask_path: Path | None = None,
    alpha_threshold: int = 8,
    target_size: tuple[int, int] | None = None,
    background_tolerance: int = 12,
) -> np.ndarray:
    """Detect the patch from alpha, or from the opaque grey background in aerial.png."""
    if explicit_mask_path:
        with Image.open(explicit_mask_path) as image:
            mask_image = image.convert("L")
            if target_size:
                mask_image = mask_image.resize(target_size, Image.Resampling.NEAREST)
            return np.asarray(mask_image, dtype=np.uint8) > alpha_threshold

    if aerial_path is None or not aerial_path.is_file():
        raise ValueError(
            "Could not resolve the satellite/aerial texture. Pass --mask-image "
            "with white pixels inside the patch and black pixels outside."
        )

    with Image.open(aerial_path) as image:
        rgba = image.convert("RGBA")
        rgb = np.asarray(rgba.convert("RGB"), dtype=np.int16)
        alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)

    if alpha.min() < 255:
        mask = alpha > alpha_threshold
    else:
        # The supplied terrain-generator aerial.png is RGB with a grey
        # background (128,128,128), not a transparent background.
        background = np.array([128, 128, 128], dtype=np.int16)
        difference = np.max(np.abs(rgb - background), axis=2)
        mask = difference > background_tolerance

    if not np.any(mask):
        raise ValueError(
            f"Could not detect a patch boundary in '{aerial_path}'. "
            "Supply --mask-image (white inside, black outside)."
        )

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if target_size:
        mask_image = mask_image.resize(target_size, Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def image_pixel(
    x: float, y: float, width: int, height: int,
    size_x: float, size_y: float, center_x: float, center_y: float,
) -> tuple[float, float]:
    """Convert world XY coordinates to floating-point image pixel coordinates."""
    px = ((x - center_x) / size_x + 0.5) * (width - 1)
    py = (0.5 - (y - center_y) / size_y) * (height - 1)
    return px, py


def sample_height(heightmap: np.ndarray, px: float, py: float) -> float:
    """Bilinearly interpolate the normalized heightmap at pixel coordinates."""
    h, w = heightmap.shape
    px = min(max(px, 0.0), w - 1.0)
    py = min(max(py, 0.0), h - 1.0)
    x0, y0 = int(math.floor(px)), int(math.floor(py))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = px - x0, py - y0
    top = heightmap[y0, x0] * (1 - dx) + heightmap[y0, x1] * dx
    bottom = heightmap[y1, x0] * (1 - dx) + heightmap[y1, x1] * dx
    return float(top * (1 - dy) + bottom * dy)


def mask_contains(mask: np.ndarray, px: float, py: float) -> bool:
    """Return whether the nearest mask pixel marks this location as inside the patch."""
    h, w = mask.shape
    ix = int(round(min(max(px, 0), w - 1)))
    iy = int(round(min(max(py, 0), h - 1)))
    return bool(mask[iy, ix])


def generate_trees(
    heightmap: np.ndarray, mask: np.ndarray,
    size_x: float, size_y: float, size_z: float,
    terrain_pos: list[float], seed: int, spacing: float,
    tree_height: float, min_radius: float, max_radius: float,
) -> list[dict[str, float | str]]:
    """Generate reproducible tree poses on a jittered grid inside the patch.
    
        World XY positions are sampled across the heightmap extent, then filtered by
        the patch mask. Heights are interpolated from the normalized heightmap and
        scaled by the SDF heightmap's vertical size and position.
        """
    rng = random.Random(seed)
    h, w = heightmap.shape
    mh, mw = mask.shape
    if (mh, mw) != (h, w):
        raise ValueError(
            f"Patch mask size {mw}x{mh} does not match heightmap size {w}x{h}."
        )

    cx, cy, cz = terrain_pos[:3]
    x_min, x_max = cx - size_x / 2 + BOUNDARY_MARGIN, cx + size_x / 2 - BOUNDARY_MARGIN
    y_min, y_max = cy - size_y / 2 + BOUNDARY_MARGIN, cy + size_y / 2 - BOUNDARY_MARGIN
    trees = []
    index = 1

    # Grid-based placement with Gaussian jitter, as in the existing generator.
    x0 = cx - size_x / 2 + spacing / 2
    while x0 < cx + size_x / 2:
        y0 = cy - size_y / 2 + spacing / 2
        while y0 < cy + size_y / 2:
            x = max(x0 - spacing / 2, min(rng.gauss(x0, POSITION_SIGMA), x0 + spacing / 2))
            y = max(y0 - spacing / 2, min(rng.gauss(y0, POSITION_SIGMA), y0 + spacing / 2))
            x = max(x_min, min(x, x_max))
            y = max(y_min, min(y, y_max))

            px, py = image_pixel(x, y, w, h, size_x, size_y, cx, cy)
            if mask_contains(mask, px, py):
                ground_z = cz + sample_height(heightmap, px, py) * size_z
                trees.append({
                    "name": f"tree_{index:04d}",
                    "x": x, "y": y, "z": ground_z + tree_height / 2,
                    "radius": rng.uniform(min_radius, max_radius),
                    "height": tree_height,
                    "yaw": rng.uniform(0, 2 * math.pi),
                })
                index += 1
            y0 += spacing
        x0 += spacing
    return trees


def add_tree_model(world: ET.Element, tree: dict[str, float | str]) -> None:
    """Append a static cylinder tree with matching visual and collision geometry."""
    model = ET.SubElement(world, "model", {"name": tree["name"]})
    text_element(model, "static", "true")
    text_element(model, "pose",
                 f'{tree["x"]:.4f} {tree["y"]:.4f} {tree["z"]:.4f} 0 0 {tree["yaw"]:.6f}')
    link = ET.SubElement(model, "link", {"name": "trunk"})
    visual = ET.SubElement(link, "visual", {"name": "trunk_visual"})
    geometry = ET.SubElement(visual, "geometry")
    cylinder = ET.SubElement(geometry, "cylinder")
    text_element(cylinder, "radius", f'{tree["radius"]:.4f}')
    text_element(cylinder, "length", f'{tree["height"]:.4f}')
    material = ET.SubElement(visual, "material")
    text_element(material, "ambient", "0.22 0.08 0.025 1")
    text_element(material, "diffuse", "0.40 0.14 0.04 1")
    text_element(material, "specular", "0.05 0.05 0.05 1")
    collision = ET.SubElement(link, "collision", {"name": "trunk_collision"})
    collision_geometry = ET.SubElement(collision, "geometry")
    collision_cylinder = ET.SubElement(collision_geometry, "cylinder")
    text_element(collision_cylinder, "radius", f'{tree["radius"]:.4f}')
    text_element(collision_cylinder, "length", f'{tree["height"]:.4f}')


def rewrite_local_resource_paths(root: ET.Element, input_world: Path, output_world: Path) -> None:
    """Keep local asset references working after moving the world to worlds/."""
    path_tags = {"uri", "diffuse", "normal"}
    for element in root.iter():
        if element.tag not in path_tags or not element.text:
            continue
        value = element.text.strip()
        if not value or value.startswith(("model://", "http://", "https://")):
            continue
        resolved = local_resource_path(input_world, value)
        if resolved is None or not resolved.exists():
            continue
        try:
            element.text = Path(
                __import__("os").path.relpath(resolved, output_world.parent)
            ).as_posix()
        except ValueError:
            # Different drives on Windows; retain original path if relpath fails.
            pass


def set_ground_material(heightmap: ET.Element, output_world: Path, texture_dir: Path, height_span:float, height_offset: float = 0.0,) -> None:
    """Replace terrain textures and calculate blend thresholds from terrain height.

    Args:
        heightmap: SDF ``<heightmap>`` element whose material is being changed.
        output_world: Destination world file, used to make asset paths relative.
        texture_dir: Directory containing the terrain diffuse and normal textures.
        height_span: Total vertical span of the heightmap in metres (SDF size Z).
        height_offset: Vertical position of the heightmap from its SDF ``<pos>``.

    The thresholds are measured from the bottom of the heightmap's vertical
    range, not from world elevation zero. They are placed at one-third and
    two-thirds of that range. Fade distances are one-sixth of the span.
    """    
    for child in list(heightmap):
        if child.tag in {"texture", "blend"}:
            heightmap.remove(child)

    textures = [
        ("rock_diffuse.png", 8),
        ("grass_diffuse.png", 8),
        ("dirt_diffuse.png", 8),
    ]
    for filename, tiling in textures:
        path = (texture_dir / filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing terrain texture: {path}")
        texture = ET.Element("texture")
        text_element(texture, "diffuse", Path(
            __import__("os").path.relpath(path, output_world.parent)
        ).as_posix())
        normal = (texture_dir / "flat_normal.png").resolve()
        if not normal.is_file():
            raise FileNotFoundError(f"Missing terrain normal map: {normal}")
        text_element(texture, "normal", Path(
            __import__("os").path.relpath(normal, output_world.parent)
        ).as_posix())
        text_element(texture, "size", str(tiling))
        heightmap.insert(0, texture)

    # Scale material transitions to the terrain's vertical span.
    # This avoids fixed thresholds that only suit one particular terrain.
    if height_span <= 0:
        raise ValueError(f"height_span must be positive; got {height_span}.")

    first_threshold = height_span / 3.0
    second_threshold = 2.0 * height_span / 3.0
    fade_distance = height_span / 6.0

    for threshold in (first_threshold, second_threshold):
        blend = ET.Element("blend")
        text_element(blend, "min_height", f"{threshold:.4f}")
        text_element(blend, "fade_dist", f"{fade_distance:.4f}")
        heightmap.append(blend)


def parse_args() -> argparse.Namespace:
    """Define and parse the command-line options for world generation."""
    parser = argparse.ArgumentParser(
        description="Copy an existing Gazebo terrain world, add trees, and write it to worlds/."
    )
    parser.add_argument("world", type=Path, help="Input terrain-generator .world file.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output world path (default: <package>/worlds/forest_<world-name>.world).")
    parser.add_argument("--terrain-style", choices=("satellite", "ground"),
                        default="ground",
                        help="Keep original imagery, or replace it with dirt/grass/rock textures.")
    parser.add_argument("--mask-image", type=Path, default=None,
                        help="Optional black/white patch mask: white = valid patch, black = outside.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacing", type=float, default=GRID_SPACING)
    parser.add_argument("--tree-height", type=float, default=TREE_HEIGHT)
    parser.add_argument("--min-radius", type=float, default=TREE_MIN_RADIUS)
    parser.add_argument("--max-radius", type=float, default=TREE_MAX_RADIUS)
    parser.add_argument("--texture-dir", type=Path, default=DEFAULT_TEXTURE_DIR,
                        help="Directory containing dirt_diffuse.png, grass_diffuse.png, rock_diffuse.png and flat_normal.png.")
    return parser.parse_args()


def main() -> None:
    """Load a terrain world, add trees, optionally change its material, and save it.
    
        The source world is preserved structurally; the output receives a forest-
        specific name and is written to the requested output directory.
        """
    args = parse_args()
    input_world = args.world.expanduser().resolve()
    if not input_world.is_file():
        raise FileNotFoundError(f"Input world does not exist: {input_world}")
    if args.spacing <= 0 or args.tree_height <= 0:
        raise ValueError("--spacing and --tree-height must be positive.")
    if args.min_radius <= 0 or args.max_radius <= 0 or args.min_radius > args.max_radius:
        raise ValueError("Tree radii must be positive and --min-radius <= --max-radius.")

    output_world = args.output
    if output_world is None:
        output_world = DEFAULT_OUTPUT_DIR / f"forest_{input_world.stem}.world"
    elif not output_world.is_absolute():
        output_world = (PACKAGE_ROOT / output_world)
    output_world = output_world.expanduser().resolve()
    output_world.parent.mkdir(parents=True, exist_ok=True)

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    xml_tree = ET.parse(input_world, parser=parser)
    root = xml_tree.getroot()
    world = root.find("./world")
    if world is None:
        raise ValueError(f"No <world> element found in {input_world}")

    _, _, visual, heightmap_element, _ = find_heightmap(root)
    uri = heightmap_element.findtext("uri", "").strip()
    heightmap_path = local_resource_path(input_world, uri)
    if heightmap_path is None or not heightmap_path.is_file():
        raise FileNotFoundError(
            f"Could not resolve heightmap URI {uri!r} relative to {input_world}. "
            "If the world uses model:// URIs, ensure the model is available locally "
            "and pass a world with resolvable local paths."
        )

    size_text = heightmap_element.findtext("size", "").split()
    if len(size_text) != 3:
        raise ValueError("Heightmap <size> must contain X Y Z values.")
    size_x, size_y, size_z = map(float, size_text)
    terrain_pos = parse_pose(heightmap_element.findtext("pos"))
    heightmap = load_heightmap(heightmap_path)

    # The first texture's diffuse image is the satellite/aerial image in the
    # terrain-generator world. The alpha channel defines the irregular patch.
    aerial_path = None
    for texture in heightmap_element.findall("./texture"):
        diffuse = texture.findtext("diffuse", "").strip()
        if diffuse:
            aerial_path = local_resource_path(input_world, diffuse)
            break
    mask_path = args.mask_image.expanduser().resolve() if args.mask_image else None
    mask = load_patch_mask(aerial_path, mask_path, target_size=(heightmap.shape[1], heightmap.shape[0]))

    trees = generate_trees(
        heightmap, mask, size_x, size_y, size_z, terrain_pos,
        args.seed, args.spacing, args.tree_height, args.min_radius, args.max_radius
    )

    if args.terrain_style == "ground":
        set_ground_material(
            heightmap_element,
            output_world,
            args.texture_dir.resolve(),
            height_span=size_z,
            height_offset=terrain_pos[2],
        )
    # Preserve the complete source world and append tree models to that world.
    for tree in trees:
        add_tree_model(world, tree)

    # Rebase local mesh/texture paths for the new world location.
    rewrite_local_resource_paths(root, input_world, output_world)

    # Give the copied world a forest-specific name without altering other settings.
    world.set("name", f"forest_{world.get('name', input_world.stem)}")
    indent_xml(root)
    xml_tree.write(output_world, encoding="utf-8", xml_declaration=True)

    print(f"Input world:    {input_world}")
    print(f"Heightmap:      {heightmap_path}")
    print(f"Patch image:    {aerial_path if aerial_path else mask_path}")
    print(f"Terrain style:  {args.terrain_style}")
    print(f"Terrain size:   {size_x:g} x {size_y:g} x {size_z:g} m")
    print(f"Trees added:    {len(trees)}")
    print(f"Output world:   {output_world}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
