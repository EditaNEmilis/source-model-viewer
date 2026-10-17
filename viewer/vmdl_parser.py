"""Source 2 compiled model (.vmdl_c) geometry import.

Uses the shared container/KV1 helpers for the MDAT payload and the
MBUF (VBIB-layout) vertex/index buffers. Only complete draw calls are
imported; trailing scene data (attachments, hitbox sets, morphs) is
ignored.

Skeleton, blend weights and animation are still open: vertices are
parented to a single dummy root bone. Materials carry the .vmat stem as
their name, and ``metadata["material_paths"]`` keeps the full resource
path so the renderer can find the matching .vmat_c/.vtex_c textures.
"""

import struct
from typing import Any, Dict, List, Tuple

from viewer.mdl_parser import MdlParseError
from viewer.smd_parser import (
    SmdBone,
    SmdFrame,
    SmdModel,
    SmdTriangle,
    SmdVertex,
)
from viewer.source2 import Source2Error, read_kv1, read_resource_blocks

_KV3_V1_MAGIC = b"\x01\x33\x56\x4b"


def _parse_mdat_draws(blob: bytes) -> Tuple[List[Dict[str, Any]], bool]:
    """Decode one v1 MDAT blob, return complete draw calls + truncated flag."""
    draws: List[Dict[str, Any]] = []

    def collect(obj: Dict[str, Any]) -> None:
        if "m_nIndexCount" in obj and "m_nVertexCount" in obj:
            draws.append(obj)

    try:
        _root, truncated = read_kv1(
            blob, on_object=collect, tolerate_truncation=True
        )
    except Source2Error as error:
        raise MdlParseError(str(error)) from error
    return draws, truncated


def _material_stem(path: str) -> str:
    stem = path.replace("\\", "/").rsplit("/", 1)[-1]
    lowered = stem.lower()
    for suffix in (".vmat_c", ".vmat"):
        if lowered.endswith(suffix):
            return stem[:-len(suffix)] or "vmdl"
    return stem or "vmdl"


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
    draws, truncated = _parse_mdat_draws(data[magic_at:mdat_off + mdat_size])
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

    model = SmdModel()
    model.bones.append(SmdBone(bone_id=0, name="root", parent_id=-1))
    model.frames.append(SmdFrame(time=0, transforms={0: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}))
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
            model.vertices.append(SmdVertex(
                parent_bone=0,
                position=(pos[0], pos[1], pos[2]),
                normal=normal,
                uv=(u0, v0),
                links=[],
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

    if model.vertices:
        xs = [v.position[0] for v in model.vertices]
        ys = [v.position[1] for v in model.vertices]
        zs = [v.position[2] for v in model.vertices]
        model.min_bound = (min(xs), min(ys), min(zs))
        model.max_bound = (max(xs), max(ys), max(zs))

    model.metadata["source"] = "vmdl_c"
    model.metadata["draw_calls"] = len(draws)
    model.metadata["material_paths"] = material_paths
    if truncated:
        model.metadata["vmdl_truncated_tail"] = True
    return model
