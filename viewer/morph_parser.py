"""Source 2 morph and flex data from the MRPH block.

The compiled morph set stores every flex delta in one texture atlas.
A morph owns a list of rects; each rect maps a block of atlas texels
onto a block of the vertex grid (vertex id = row * width + column).
Texel channels are quantised deltas: ``channel / 255 * range + offset``,
one vector per bundle type (position speed, normal wrinkle).

Layout follows ValveResourceFormat's Morph block reader.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from viewer.source2 import Source2Error, read_kv1

POSITION_SPEED = "POSITION_SPEED"
NORMAL_WRINKLE = "NORMAL_WRINKLE"

_BUNDLE_NAMES = {
    "MORPH_BUNDLE_TYPE_POSITION_SPEED": POSITION_SPEED,
    "BUNDLE_TYPE_POSITION_SPEED": POSITION_SPEED,
    "MORPH_BUNDLE_TYPE_NORMAL_WRINKLE": NORMAL_WRINKLE,
    "BUNDLE_TYPE_NORMAL_WRINKLE": NORMAL_WRINKLE,
}

_DELTA_EPSILON = 1e-5


class MorphError(ValueError):
    pass


class MorphBundle:
    """One bundle entry inside a rect: atlas origin and delta quantisation."""

    __slots__ = ("u_source", "v_source", "offsets", "ranges")

    def __init__(self, entry: dict) -> None:
        self.u_source = float(entry.get("m_flULeftSrc", 0.0))
        self.v_source = float(entry.get("m_flVTopSrc", 0.0))
        offsets = entry.get("m_offsets") or [0.0, 0.0, 0.0, 0.0]
        ranges = entry.get("m_ranges") or [1.0, 1.0, 1.0, 1.0]
        self.offsets = [float(value) for value in offsets[:4]]
        self.ranges = [float(value) for value in ranges[:4]]


class MorphRect:
    """A rect copying atlas texels into a block of vertex grid cells."""

    __slots__ = ("x", "y", "width", "height", "bundles", "_u_width", "_v_height")

    def __init__(self, entry: dict) -> None:
        self.x = int(entry.get("m_nXLeftDst", 0))
        self.y = int(entry.get("m_nYTopDst", 0))
        self._u_width = float(entry.get("m_flUWidthSrc", 0.0))
        self._v_height = float(entry.get("m_flVHeightSrc", 0.0))
        self.width = 0
        self.height = 0
        self.bundles: List[MorphBundle] = [
            MorphBundle(bundle)
            for bundle in (entry.get("m_bundleDatas") or [])
            if isinstance(bundle, dict)
        ]


class MorphData:
    __slots__ = ("name", "rects")

    def __init__(self, name: str) -> None:
        self.name = name
        self.rects: List[MorphRect] = []


class MorphSet:
    __slots__ = (
        "width",
        "height",
        "bundle_types",
        "morphs",
        "flex_names",
        "flex_controllers",
        "flex_rules",
        "atlas_path",
    )

    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.bundle_types: List[str] = []
        self.morphs: List[MorphData] = []
        self.flex_names: List[str] = []
        self.flex_controllers: List[dict] = []
        self.flex_rules: List[Tuple[int, list]] = []
        self.atlas_path = ""


def _bundle_name(value) -> Optional[str]:
    if isinstance(value, str):
        return _BUNDLE_NAMES.get(value.strip().upper(), None)
    if isinstance(value, int):
        return {0: None, 1: POSITION_SPEED, 2: NORMAL_WRINKLE}.get(value)
    return None


def parse_morph_block(blob: bytes) -> MorphSet:
    """Decode an MRPH block into its morph, atlas and flex metadata."""
    try:
        root, _truncated = read_kv1(blob)
    except Source2Error as error:
        raise MorphError(str(error)) from error

    if not isinstance(root, dict):
        raise MorphError("MRPH payload did not decode to an object")

    morph_set = MorphSet()
    morph_set.width = int(root.get("m_nWidth") or 0)
    morph_set.height = int(root.get("m_nHeight") or 0)
    if morph_set.width <= 0 or morph_set.height <= 0:
        raise MorphError("MRPH vertex grid is empty")

    morph_set.atlas_path = str(root.get("m_pTextureAtlas") or "")
    morph_set.bundle_types = [
        _bundle_name(value) or "NONE"
        for value in (root.get("m_bundleTypes") or [])
    ]

    for entry in root.get("m_morphDatas") or []:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("m_name") or "")
        if not name:
            continue
        morph = MorphData(name)
        for rect_entry in entry.get("m_morphRectDatas") or []:
            if isinstance(rect_entry, dict):
                morph.rects.append(MorphRect(rect_entry))
        morph_set.morphs.append(morph)

    for entry in root.get("m_FlexDesc") or []:
        if isinstance(entry, dict):
            morph_set.flex_names.append(str(entry.get("m_szFacs") or ""))

    for entry in root.get("m_FlexControllers") or []:
        if not isinstance(entry, dict):
            continue
        morph_set.flex_controllers.append({
            "name": str(entry.get("m_szName") or ""),
            "type": str(entry.get("m_szType") or ""),
            "min": float(entry.get("min", 0.0) or 0.0),
            "max": float(entry.get("max", 1.0) or 1.0),
        })

    for entry in root.get("m_FlexRules") or []:
        if not isinstance(entry, dict):
            continue
        ops = [
            (str(op.get("m_OpCode") or ""), int(op.get("m_Data") or 0))
            for op in (entry.get("m_FlexOps") or [])
            if isinstance(op, dict)
        ]
        morph_set.flex_rules.append((int(entry.get("m_nFlex") or 0), ops))

    return morph_set


def decode_deltas(
    morph_set: MorphSet,
    atlas: np.ndarray,
    bundle_type: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return ``{morph name: (vertex ids, xyz deltas)}`` for one bundle.

    Only cells whose decoded vector is non-zero are kept, which keeps the
    shape-key overrides small.
    """
    if atlas is None or atlas.ndim != 3 or atlas.shape[2] < 3:
        return {}

    pixels = atlas.astype(np.float32) / 255.0
    texture_height, texture_width = pixels.shape[:2]
    grid_width, grid_height = morph_set.width, morph_set.height
    cell_count = grid_width * grid_height

    decoded: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for morph in morph_set.morphs:
        grid = np.zeros((cell_count, 3), dtype=np.float32)
        touched = np.zeros(cell_count, dtype=bool)

        for rect in morph.rects:
            rect_width = int(round(rect._u_width * texture_width))
            rect_height = int(round(rect._v_height * texture_height))
            if rect_width <= 0 or rect_height <= 0:
                continue

            for index, bundle in enumerate(rect.bundles):
                if index >= len(morph_set.bundle_types):
                    break
                if morph_set.bundle_types[index] != bundle_type:
                    continue

                u = int(round(bundle.u_source * texture_width))
                v = int(round(bundle.v_source * texture_height))
                block = pixels[v:v + rect_height, u:u + rect_width, :3]
                if block.shape[0] != rect_height or block.shape[1] != rect_width:
                    continue

                values = (
                    block * np.asarray(bundle.ranges[:3], dtype=np.float32)
                    + np.asarray(bundle.offsets[:3], dtype=np.float32)
                )

                rows = rect.y + np.arange(rect_height)
                columns = rect.x + np.arange(rect_width)
                row_mask = (rows >= 0) & (rows < grid_height)
                column_mask = (columns >= 0) & (columns < grid_width)
                if not row_mask.any() or not column_mask.any():
                    continue

                values = values[row_mask][:, column_mask]
                flat = (
                    rows[row_mask][:, None] * grid_width
                    + columns[column_mask][None, :]
                ).ravel()
                grid[flat] = values.reshape(-1, 3)
                touched[flat] = True

        if not touched.any():
            continue

        indices = np.nonzero(touched)[0]
        deltas = grid[indices]
        significant = np.linalg.norm(deltas, axis=1) > _DELTA_EPSILON
        if not significant.any():
            continue
        decoded[morph.name] = (indices[significant], deltas[significant])

    return decoded
