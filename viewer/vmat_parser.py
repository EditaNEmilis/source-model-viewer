"""Source 2 compiled material (.vmat_c) import.

A compiled material stores its shader name and parameter arrays in a
binary KV3 v1 DATA block. Only the texture parameters matter to the
viewer: they name the textures a draw call should sample, and the
``g_tColor`` / ``TextureColor`` slot holds the base color map.
"""

import struct
from typing import Dict, List, Optional, Tuple

from viewer.source2 import Source2Error, read_resource_blocks, read_kv1

# Texture parameter names that carry base color, most specific first.
_COLOR_TEXTURE_KEYS = (
    "g_tcolor",
    "texturecolor",
    "g_tcolor1",
    "texturealbedo",
    "g_talbedo",
)

# Names that contain "color" but are not the diffuse map.
_COLOR_KEY_SKIPS = ("mask", "warp", "ao", "tint", "gradient", "falloff")

_NORMAL_TEXTURE_KEYS = ("g_tnormal", "texturenormal", "g_tnormal1")
_AO_TEXTURE_KEYS = ("g_tambientocclusion", "textureambientocclusion", "g_tao")
_MASK_TEXTURE_KEYS = ("g_tcombinedmasks", "texturecombinedmasks")


class VmatError(ValueError):
    pass


class VmatMaterial:
    """A parsed .vmat_c: shader name plus typed parameter maps."""

    __slots__ = (
        "source_path",
        "name",
        "shader",
        "texture_params",
        "int_params",
        "float_params",
        "vector_params",
    )

    def __init__(self) -> None:
        self.source_path = ""
        self.name = ""
        self.shader = ""
        self.texture_params: Dict[str, str] = {}
        self.int_params: Dict[str, int] = {}
        self.float_params: Dict[str, float] = {}
        self.vector_params: Dict[str, Tuple[float, ...]] = {}


def is_vmat_c_file(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            data = handle.read(32)
    except OSError:
        return False
    if len(data) < 16:
        return False
    header_version = struct.unpack_from("<H", data, 4)[0]
    return header_version == 12


def _as_list(value) -> List:
    return value if isinstance(value, list) else []


def _entry_name(entry) -> str:
    name = entry.get("m_name") if isinstance(entry, dict) else None
    return name if isinstance(name, str) else ""


def _texture_ref(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def parse_vmat_c(path: str) -> VmatMaterial:
    with open(path, "rb") as handle:
        data = handle.read()

    try:
        blocks = read_resource_blocks(data)
    except Source2Error as error:
        raise VmatError(str(error)) from error

    if "DATA" not in blocks:
        raise VmatError("VMAT DATA block missing")
    start, size = blocks["DATA"]

    try:
        root, _truncated = read_kv1(data[start:start + size])
    except Source2Error as error:
        raise VmatError(str(error)) from error

    if not isinstance(root, dict):
        raise VmatError("VMAT payload did not decode to an object")

    material = VmatMaterial()
    material.source_path = path
    name = root.get("m_materialName")
    material.name = name if isinstance(name, str) else ""
    shader = root.get("m_shaderName") or root.get("m_shader") or ""
    material.shader = shader if isinstance(shader, str) else ""

    for entry in _as_list(root.get("m_textureParams")):
        name = _entry_name(entry)
        value = _texture_ref(entry.get("m_pValue", entry.get("m_value")))
        if name and value:
            material.texture_params[name] = value

    for entry in _as_list(root.get("m_intParams")):
        name = _entry_name(entry)
        value = entry.get("m_nValue")
        if name and isinstance(value, int):
            material.int_params[name] = value

    for entry in _as_list(root.get("m_floatParams")):
        name = _entry_name(entry)
        value = entry.get("m_flValue")
        if name and isinstance(value, (int, float)):
            material.float_params[name] = float(value)

    for entry in _as_list(root.get("m_vectorParams")):
        name = _entry_name(entry)
        value = entry.get("m_value")
        if name and isinstance(value, list):
            components = [float(v) for v in value if isinstance(v, (int, float))]
            if components:
                material.vector_params[name] = tuple(components)

    return material


def pick_color_texture(material: VmatMaterial) -> Optional[str]:
    """Return the base color texture reference for a material."""
    lowered = {
        name.lower(): value for name, value in material.texture_params.items()
    }

    for key in _COLOR_TEXTURE_KEYS:
        value = lowered.get(key)
        if value:
            return value

    for name, value in material.texture_params.items():
        lowered_name = name.lower()
        if "color" not in lowered_name:
            continue
        if any(skip in lowered_name for skip in _COLOR_KEY_SKIPS):
            continue
        return value
    return None


def pick_named_texture(material: VmatMaterial, keys) -> Optional[str]:
    """Return the first texture reference matching one of ``keys``."""
    lowered = {
        name.lower(): value for name, value in material.texture_params.items()
    }
    for key in keys:
        value = lowered.get(key)
        if value:
            return value
    return None


def pick_normal_texture(material: VmatMaterial) -> Optional[str]:
    return pick_named_texture(material, _NORMAL_TEXTURE_KEYS)


def pick_occlusion_texture(material: VmatMaterial):
    """Return (texture reference, channel index) for the AO map.

    Dedicated ambient occlusion maps store AO in red (BC4). Skin and
    character shaders pack it into the green channel of a combined mask
    texture instead.
    """
    reference = pick_named_texture(material, _AO_TEXTURE_KEYS)
    if reference:
        return reference, 0

    reference = pick_named_texture(material, _MASK_TEXTURE_KEYS)
    if reference:
        return reference, 1
    return None, 0
