"""Source 2 compiled model (.vmdl_c) geometry import.

Uses the shared container/KV1 helpers for the MDAT payload and the
MBUF (VBIB-layout) vertex/index buffers. Only complete draw calls are
imported; trailing scene data (attachments, hitbox sets, morphs) is
ignored.

The bone hierarchy and bind pose come from ``m_skeleton`` in the MDAT
root, and per-vertex blend indices/weights from the VBIB layout, so
VMDL models take part in the existing skeleton, skinning and animation
pipeline. Animation blocks are still open. Materials carry the .vmat
stem as their name, and ``metadata["material_paths"]`` keeps the full
resource path so the renderer can find the matching .vmat_c/.vtex_c
textures.
"""

import math
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from viewer.mdl_parser import MdlParseError
from viewer.morph_parser import (
    MorphError,
    NORMAL_WRINKLE,
    POSITION_SPEED,
    decode_deltas,
    parse_morph_block,
)
from viewer.skeleton import invert_affine_matrix, multiply_matrices
from viewer.smd_parser import (
    SmdBone,
    SmdFrame,
    SmdModel,
    SmdTriangle,
    SmdVertex,
    VertexTarget,
)
from viewer.source2 import Source2Error, read_kv1, read_resource_blocks
from viewer.vtex_parser import VtexError, parse_vtex_c

_KV3_V1_MAGIC = b"\x01\x33\x56\x4b"


def _parse_mdat_draws(
    blob: bytes,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """Decode one v1 MDAT blob, return draw calls + root + truncated flag."""
    draws: List[Dict[str, Any]] = []

    def collect(obj: Dict[str, Any]) -> None:
        if "m_nIndexCount" in obj and "m_nVertexCount" in obj:
            draws.append(obj)

    try:
        root, truncated = read_kv1(
            blob, on_object=collect, tolerate_truncation=True
        )
    except Source2Error as error:
        raise MdlParseError(str(error)) from error
    return draws, root, truncated


def _material_stem(path: str) -> str:
    stem = path.replace("\\", "/").rsplit("/", 1)[-1]
    lowered = stem.lower()
    for suffix in (".vmat_c", ".vmat"):
        if lowered.endswith(suffix):
            return stem[:-len(suffix)] or "vmdl"
    return stem or "vmdl"


def _matrix_from_inv_bind(values) -> Tuple[float, ...]:
    """Expand a 3x4 row-major inverse bind pose into a 4x4 matrix."""
    if not isinstance(values, list) or len(values) < 12:
        return (1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0)
    x = [float(value) for value in values[:12]]
    return (
        x[0], x[1], x[2], x[3],
        x[4], x[5], x[6], x[7],
        x[8], x[9], x[10], x[11],
        0.0, 0.0, 0.0, 1.0,
    )


def _decompose_transform(matrix: Tuple[float, ...]):
    """Split an affine matrix into position and SMD style Euler angles."""
    m00, m01, m02, tx = matrix[0], matrix[1], matrix[2], matrix[3]
    m10, m11, m12, ty = matrix[4], matrix[5], matrix[6], matrix[7]
    m20, m21, m22, tz = matrix[8], matrix[9], matrix[10], matrix[11]

    sy = -max(-1.0, min(1.0, m20))
    ry = math.asin(sy)
    cy = math.cos(ry)

    if abs(cy) > 1e-6:
        rx = math.atan2(m21, m22)
        rz = math.atan2(m10, m00)
    else:
        rx = math.atan2(-m12, m11)
        rz = 0.0

    return ((tx, ty, tz), (rx, ry, rz))


def _parse_skeleton(root: Optional[Dict[str, Any]]):
    """Return (bones, bind transforms) from the MDAT skeleton block.

    ``m_invBindPose`` stores model-space inverse bind matrices in a 3x4
    row-major layout. Inverting and rebasing them onto each parent gives
    the parent-space transforms the rest of the viewer expects.
    """
    skeleton = (root or {}).get("m_skeleton") or {}
    entries = skeleton.get("m_bones") or []
    if not isinstance(entries, list) or not entries:
        return [], {}

    names: List[str] = []
    name_to_id: Dict[str, int] = {}
    for index, entry in enumerate(entries):
        name = str(entry.get("m_boneName") or f"bone_{index}")
        names.append(name)
        name_to_id.setdefault(name, index)

    bones: List[SmdBone] = []
    for index, entry in enumerate(entries):
        parent_name = str(entry.get("m_parentName") or "")
        parent_id = name_to_id.get(parent_name, -1) if parent_name else -1
        if parent_id == index:
            parent_id = -1
        bones.append(SmdBone(bone_id=index, name=names[index], parent_id=parent_id))

    world: Dict[int, Tuple[float, ...]] = {}
    for index, entry in enumerate(entries):
        inv_bind = _matrix_from_inv_bind(entry.get("m_invBindPose"))
        world[index] = invert_affine_matrix(inv_bind)

    transforms = {}
    for bone in bones:
        parent_id = bone.parent_id
        if parent_id >= 0 and parent_id in world:
            parent_inverse = invert_affine_matrix(world[parent_id])
            local = multiply_matrices(parent_inverse, world[bone.bone_id])
        else:
            local = world[bone.bone_id]
        transforms[bone.bone_id] = _decompose_transform(local)
    return bones, transforms


def _decode_vertex_links(
    vdata: bytes,
    index_offset: int,
    weight_offset: int,
    bone_count: int,
) -> List[Tuple[int, float]]:
    """Read up to four blend influences from one vertex record."""
    if bone_count <= 0:
        return []
    indices = vdata[index_offset:index_offset + 4]
    weights = vdata[weight_offset:weight_offset + 4]
    if len(indices) < 4 or len(weights) < 4:
        return []

    links: List[Tuple[int, float]] = []
    total = 0.0
    for slot in range(4):
        weight = weights[slot] / 255.0
        bone_id = indices[slot]
        if weight <= 0.0 or bone_id >= bone_count:
            continue
        links.append((bone_id, weight))
        total += weight

    if total <= 1e-6:
        return []
    return [(bone_id, weight / total) for bone_id, weight in links]


def _decode_buffers(
    data: bytes, base: int, table: int, vb_tbl: int, ib_tbl: int,
    vbuf_id: int, ibuf_id: int,
) -> Tuple[bytes, bytes, int, int]:
    """Return (vertex_bytes, index_bytes, vertex_count, index_count).

    Buffer payloads are addressed from the MBUF block start (base);
    the final index buffer of this file overruns the nominal block
    end by 44 bytes into the next block, so reads go to the whole
    file image, not the trimmed block slice.
    """
    # Data offsets are relative to the descriptor fields
    # (refB = descriptor + 16).
    voff = base + table + vb_tbl + vbuf_id * 24
    count, size, _aoff, _an, doff, _total = struct.unpack_from("<IiIIIi", data, voff)
    stride = size & 0x3FFFFFF
    vstart = voff + 16 + doff
    vdata = bytes(data[vstart:vstart + count * stride])
    ioff = base + table + 8 + ib_tbl + ibuf_id * 24
    icount, isize, _a, _n, idoff, _t = struct.unpack_from("<IiIIIi", data, ioff)
    esize = isize & 0x3FFFFFF
    istart = ioff + 16 + idoff
    idata = bytes(data[istart:istart + icount * esize])
    return vdata, idata, count, icount


def _find_mbuf_table(mbuf: bytes) -> int:
    for off in range(0, 1024, 4):
        _vb_off, vb_n, _ib_off, ib_n = struct.unpack_from("<4I", mbuf, off)
        if 1 <= vb_n <= 64 and 1 <= ib_n <= 64:
            # Sanity: first vertex descriptor must have a sane stride.
            count, size = struct.unpack_from("<Ii", mbuf, off + _vb_off)
            if 0 < (size & 0x3FFFFFF) <= 256 and 0 < count < 10_000_000:
                return off
    raise MdlParseError("MBUF buffer table not found")


_INPUT_ATTRIBUTE_FORMAT = 36
_INPUT_ATTRIBUTE_OFFSET = 40

# DXGI_FORMAT values used by Source 2 vertex layouts.
_DXGI_R32G32B32_FLOAT = 6
_DXGI_R32G32_FLOAT = 16
_DXGI_R10G10B10A2_UNORM = 26
_DXGI_R8G8B8A8_UNORM = 28
_DXGI_R8G8B8A8_SNORM = 31
_DXGI_R16G16_FLOAT = 34
_DXGI_R16G16_UNORM = 35
_DXGI_R16G16_SNORM = 37


def _read_input_layout(mbuf: bytes) -> Dict[str, Tuple[int, int]]:
    """Return {attribute_name: (dxgi_format, byte_offset)} for one VBIB."""
    layout: Dict[str, Tuple[int, int]] = {}
    for name in ("POSITION", "TEXCOORD", "NORMAL", "BLENDINDICES", "BLENDWEIGHT"):
        index = mbuf.find(name.encode("ascii"), 0, 4096)
        if index < 0 or index + _INPUT_ATTRIBUTE_OFFSET + 4 > len(mbuf):
            continue
        fmt, offset = struct.unpack_from(
            "<II", mbuf, index + _INPUT_ATTRIBUTE_FORMAT
        )
        layout[name] = (fmt, offset)
    return layout


def _decode_texcoord(
    vdata: bytes, offset: int, fmt: int
) -> Tuple[float, float]:
    if fmt == _DXGI_R16G16_FLOAT:
        return struct.unpack_from("<2e", vdata, offset)
    if fmt == _DXGI_R32G32_FLOAT:
        return struct.unpack_from("<2f", vdata, offset)
    if fmt == _DXGI_R16G16_SNORM:
        x, y = struct.unpack_from("<2h", vdata, offset)
        return (x / 32767.0, y / 32767.0)
    # R16G16_UNORM and unknown layouts fall back to normalised shorts.
    x, y = struct.unpack_from("<2H", vdata, offset)
    return (x / 65535.0, y / 65535.0)


def _decode_normal(
    vdata: bytes, offset: int, fmt: int
) -> Tuple[float, float, float]:
    if fmt == _DXGI_R8G8B8A8_SNORM:
        x, y, z = struct.unpack_from("<3b", vdata, offset)
        return (x / 127.0, y / 127.0, z / 127.0)
    if fmt == _DXGI_R10G10B10A2_UNORM:
        packed, = struct.unpack_from("<I", vdata, offset)
        return (
            (packed & 0x3FF) / 1023.0 * 2.0 - 1.0,
            ((packed >> 10) & 0x3FF) / 1023.0 * 2.0 - 1.0,
            ((packed >> 20) & 0x3FF) / 1023.0 * 2.0 - 1.0,
        )
    nx, ny, nz, _nw = vdata[offset:offset + 4]
    return (nx / 255 * 2 - 1, ny / 255 * 2 - 1, nz / 255 * 2 - 1)


def _rebuild_smooth_normals(model: SmdModel) -> None:
    """Replace vertex normals with area-weighted geometric normals.

    The NORMAL stream in these files does not decode into a usable
    surface normal (roughly a fifth of the faces agree with it), while
    the geometry itself is sound. Rebuilding from the triangles gives
    consistent lighting and a usable tangent frame for normal maps.
    """
    if not model.vertices or not model.triangles:
        return

    positions = [vertex.position for vertex in model.vertices]
    accumulators = [[0.0, 0.0, 0.0] for _ in positions]

    for triangle in model.triangles:
        i0, i1, i2 = triangle.indices
        x0, y0, z0 = positions[i0]
        x1, y1, z1 = positions[i1]
        x2, y2, z2 = positions[i2]
        ux, uy, uz = x1 - x0, y1 - y0, z1 - z0
        vx, vy, vz = x2 - x0, y2 - y0, z2 - z0
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        for index in (i0, i1, i2):
            accumulator = accumulators[index]
            accumulator[0] += nx
            accumulator[1] += ny
            accumulator[2] += nz

    for vertex, (ax, ay, az) in zip(model.vertices, accumulators):
        length = (ax * ax + ay * ay + az * az) ** 0.5
        if length < 1e-12:
            continue
        vertex.normal = (ax / length, ay / length, az / length)


def _resolve_morph_atlas(reference: str, model_path: str) -> Optional[str]:
    if not reference:
        return None
    base = os.path.dirname(os.path.abspath(model_path))
    relative = reference.replace("\\", "/").strip()
    candidates = [relative + "_c"]
    if not relative.lower().endswith((".vtex", ".vtex_c")):
        candidates.append(relative)
    for candidate in candidates:
        path = os.path.join(base, candidate)
        if os.path.isfile(path):
            return path
    return None


def _attach_morph_targets(
    model: SmdModel,
    blocks: Dict[str, Tuple[int, int]],
    data: bytes,
    model_path: str,
) -> None:
    """Decode the MRPH morph set into vertex targets (shape keys)."""
    start, size = blocks["MRPH"]
    morph_set = parse_morph_block(data[start:start + size])
    model.metadata["morph_count"] = len(morph_set.morphs)
    model.metadata["flex_controllers"] = [
        controller["name"] for controller in morph_set.flex_controllers
    ]
    model.metadata["morph_atlas"] = morph_set.atlas_path

    atlas_path = _resolve_morph_atlas(morph_set.atlas_path, model_path)
    if atlas_path is None:
        model.metadata["morph_error"] = (
            f"morph atlas not found: {morph_set.atlas_path}"
        )
        return

    try:
        atlas = parse_vtex_c(atlas_path)
    except (VtexError, OSError) as error:
        model.metadata["morph_error"] = str(error)
        return

    if atlas.rgba is None:
        model.metadata["morph_error"] = "morph atlas has no RGBA pixels"
        return

    position_deltas = decode_deltas(morph_set, atlas.rgba, POSITION_SPEED)
    normal_deltas = decode_deltas(morph_set, atlas.rgba, NORMAL_WRINKLE)

    vertex_count = len(model.vertices)
    base_positions = np.array(
        [vertex.position for vertex in model.vertices], dtype=np.float32
    )
    base_normals = np.array(
        [vertex.normal for vertex in model.vertices], dtype=np.float32
    )

    targets = [VertexTarget(time=0, overrides={}, name="bind")]

    for index, morph in enumerate(morph_set.morphs, start=1):
        position_entry = position_deltas.get(morph.name)
        if position_entry is None:
            continue

        ids, deltas = position_entry
        valid = ids < vertex_count
        if not valid.all():
            ids, deltas = ids[valid], deltas[valid]
        positions = (base_positions[ids] + deltas).tolist()
        normals = base_normals[ids].tolist()

        normal_entry = normal_deltas.get(morph.name)
        normal_map: Dict[int, Tuple[float, float, float]] = {}
        if normal_entry is not None:
            normal_ids, normal_values = normal_entry
            normal_valid = normal_ids < vertex_count
            if not normal_valid.all():
                normal_ids = normal_ids[normal_valid]
                normal_values = normal_values[normal_valid]
            normal_map = {
                int(vertex_id): (float(value[0]), float(value[1]), float(value[2]))
                for vertex_id, value in zip(normal_ids.tolist(), normal_values.tolist())
            }

        overrides = {}
        for vertex_id, position, normal in zip(ids.tolist(), positions, normals):
            delta = normal_map.get(vertex_id)
            if delta is not None:
                nx = normal[0] + delta[0]
                ny = normal[1] + delta[1]
                nz = normal[2] + delta[2]
                length = (nx * nx + ny * ny + nz * nz) ** 0.5
                if length > 1e-6:
                    normal = (nx / length, ny / length, nz / length)
            overrides[int(vertex_id)] = (
                (float(position[0]), float(position[1]), float(position[2])),
                (float(normal[0]), float(normal[1]), float(normal[2])),
            )

        if overrides:
            targets.append(VertexTarget(time=index, overrides=overrides, name=morph.name))

    if len(targets) > 1:
        model.vertex_targets.extend(targets)
    model.metadata["shape_keys"] = len(targets) - 1


def parse_vmdl_c(path: str) -> SmdModel:
    """Parse a Source 2 compiled model into an SmdModel (static geometry)."""
    with open(path, "rb") as handle:
        data = handle.read()

    try:
        blocks = read_resource_blocks(data)
    except Source2Error as error:
        raise MdlParseError(str(error)) from error
    if "MDAT" not in blocks or "MBUF" not in blocks:
        raise MdlParseError("Not a Source 2 model: MDAT/MBUF blocks missing")

    mdat_off, mdat_size = blocks["MDAT"]
    magic_at = data.find(_KV3_V1_MAGIC, mdat_off, mdat_off + 128)
    if magic_at < 0:
        raise MdlParseError("MDAT KV3 blob magic not found")
    draws, mdat_root, truncated = _parse_mdat_draws(data[magic_at:mdat_off + mdat_size])
    if not draws:
        raise MdlParseError("No complete draw calls in MDAT blob")

    mbuf_off, mbuf_size = blocks["MBUF"]
    mbuf = data[mbuf_off:mbuf_off + mbuf_size]
    table = _find_mbuf_table(mbuf)
    vb_off_tbl, _vb_n, ib_off_tbl, _ib_n = struct.unpack_from(
        "<4I", mbuf, table)
    layout = _read_input_layout(mbuf)
    position_offset = layout.get("POSITION", (_DXGI_R32G32B32_FLOAT, 0))[1]
    texcoord_format, texcoord_offset = layout.get("TEXCOORD", (_DXGI_R16G16_UNORM, 12))
    normal_format, normal_offset = layout.get("NORMAL", (_DXGI_R8G8B8A8_UNORM, 16))
    blend_indices_format, blend_indices_offset = layout.get(
        "BLENDINDICES", (30, 20)
    )
    _blend_weight_format, blend_weight_offset = layout.get("BLENDWEIGHT", (28, 24))

    bones, bind_transforms = _parse_skeleton(mdat_root)
    if not bones:
        bones = [SmdBone(bone_id=0, name="root", parent_id=-1)]
        bind_transforms = {0: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}
    bone_ids = {bone.bone_id for bone in bones}

    model = SmdModel()
    model.bones.extend(bones)
    model.frames.append(SmdFrame(time=0, transforms=bind_transforms))
    model.frame_names[0] = "bind"
    model.has_geometry = True
    material_paths: Dict[str, str] = {}

    for draw in draws:
        vbuf_id = int(draw["m_vertexBuffers"][0]["m_hBuffer"])
        ibuf_id = int(draw["m_indexBuffer"]["m_hBuffer"])
        nverts = int(draw["m_nVertexCount"])
        nindex = int(draw["m_nIndexCount"])
        base = len(model.vertices)
        vdata, idata, vb_count, ib_count = _decode_buffers(
            data, mbuf_off, table, vb_off_tbl, ib_off_tbl,
            vbuf_id, ibuf_id)
        if vb_count < nverts or ib_count < nindex:
            raise MdlParseError("Draw call references out-of-range buffers")
        stride = len(vdata) // vb_count if vb_count else 0
        if stride < 28:
            raise MdlParseError(f"Unexpected vertex stride {stride}")
        material = _material_stem(str(draw.get("m_material", "")))
        model.materials.add(material)
        raw_material = str(draw.get("m_material", "")).replace("\\", "/").strip()
        if material and raw_material:
            material_paths.setdefault(material, raw_material)
        for v in range(nverts):
            off = v * stride
            pos = struct.unpack_from("<3f", vdata, off + position_offset)
            u0, v0 = _decode_texcoord(vdata, off + texcoord_offset, texcoord_format)
            normal = _decode_normal(vdata, off + normal_offset, normal_format)
            links = []
            if blend_indices_format in (30, 28, 31):
                links = _decode_vertex_links(
                    vdata,
                    off + blend_indices_offset,
                    off + blend_weight_offset,
                    len(bones),
                )
            parent_bone = links[0][0] if links else (0 if 0 in bone_ids else -1)
            model.vertices.append(SmdVertex(
                parent_bone=parent_bone,
                position=(pos[0], pos[1], pos[2]),
                normal=normal,
                uv=(u0, v0),
                links=links,
            ))
        idx = struct.unpack_from(f"<{nindex}H", idata, 0)
        if max(idx) >= nverts:
            raise MdlParseError("Draw call index out of vertex range")
        for t in range(0, nindex - 2, 3):
            model.triangles.append(SmdTriangle(
                material=material,
                indices=(base + idx[t], base + idx[t + 1], base + idx[t + 2]),
            ))

    _rebuild_smooth_normals(model)

    if "MRPH" in blocks:
        try:
            _attach_morph_targets(model, blocks, data, path)
        except (MorphError, Source2Error, ValueError) as error:
            model.metadata["morph_error"] = str(error)

    if model.vertices:
        xs = [v.position[0] for v in model.vertices]
        ys = [v.position[1] for v in model.vertices]
        zs = [v.position[2] for v in model.vertices]
        model.min_bound = (min(xs), min(ys), min(zs))
        model.max_bound = (max(xs), max(ys), max(zs))

    model.metadata["source"] = "vmdl_c"
    model.metadata["draw_calls"] = len(draws)
    model.metadata["material_paths"] = material_paths

    animation_blocks = [
        tag for tag in ("ANIM", "ASEQ", "AGRG") if tag in blocks
    ]
    if animation_blocks:
        model.metadata["animation_error"] = (
            "embedded Source 2 animation blocks are not decoded yet ("
            + ", ".join(animation_blocks)
            + ")"
        )

    if truncated:
        model.metadata["vmdl_truncated_tail"] = True
    return model
