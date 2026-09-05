"""Source 2 compiled model (.vmdl_c) geometry import.

Supports the v1 KV3 MDAT payload (LZ4 + node reader) and the MBUF
(VBIB-layout) vertex/index buffers. Only complete draw calls are
imported; this file's MDAT stream is short of its stated size
(see viewer/temp/mds/SOURCE2_VMDL_FORMAT_NOTES.md), so trailing
scene data (attachments, hitbox sets, morphs) is skipped with a
metadata warning instead of failing the whole load.

Skeleton, blend weights and textures are still open: vertices are
parented to a single dummy root bone and materials carry the .vmat
stem as their name.
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

# KV3 v1 node types (BinaryKV3.NodeType.cs).
_NULL = 1
_BOOLEAN = 2
_INT64 = 3
_UINT64 = 4
_DOUBLE = 5
_STRING = 6
_BLOB = 7
_ARRAY = 8
_OBJECT = 9
_ARRAY_TYPED = 10
_INT32 = 11
_UINT32 = 12
_BOOL_TRUE = 13
_BOOL_FALSE = 14
_INT64_ZERO = 15
_INT64_ONE = 16
_DOUBLE_ZERO = 17
_DOUBLE_ONE = 18

_SINGLETONS = {
    _NULL: None,
    _BOOL_TRUE: True,
    _BOOL_FALSE: False,
    _INT64_ZERO: 0,
    _INT64_ONE: 1,
    _DOUBLE_ZERO: 0.0,
    _DOUBLE_ONE: 1.0,
}

_KV3_V1_MAGIC = 0x4B563301


class _Truncated(Exception):
    """Raised when the KV3 type/buffer streams run out mid-value."""


def lz4_block_decompress(src: bytes) -> bytes:
    """Decompress one raw LZ4 block (no frame header)."""
    out = bytearray()
    i = 0
    n = len(src)
    while i < n:
        token = src[i]
        i += 1
        lit_len = token >> 4
        if lit_len == 15:
            while True:
                if i >= n:
                    raise MdlParseError("LZ4 stream truncated in literal length")
                byte = src[i]
                i += 1
                lit_len += byte
                if byte != 255:
                    break
        if i + lit_len > n:
            raise MdlParseError("LZ4 stream truncated in literals")
        out.extend(src[i:i + lit_len])
        i += lit_len
        if i >= n:
            break
        if i + 2 > n:
            raise MdlParseError("LZ4 stream truncated in match offset")
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        if offset == 0 or offset > len(out):
            raise MdlParseError(f"Bad LZ4 match offset {offset}")
        match_len = (token & 0xF) + 4
        if (token & 0xF) == 15:
            while True:
                if i >= n:
                    raise MdlParseError("LZ4 stream truncated in match length")
                byte = src[i]
                i += 1
                match_len += byte
                if byte != 255:
                    break
        for _ in range(match_len):
            out.append(out[len(out) - offset])
    return bytes(out)


class _V1Reader:
    """Minimal KV3 v1 pull reader over one decompressed blob."""

    def __init__(self, buf: bytes, count4: int, count8: int) -> None:
        self.b4 = bytearray(buf[0:count4 * 4])
        self.b8 = bytearray(buf[count4 * 4:count4 * 4 + count8 * 8])
        self.b1 = bytearray()
        self.strings: List[str] = []
        pos = count4 * 4 + count8 * 8
        count_strings = self._take4()
        for _ in range(count_strings):
            end = buf.find(b"\x00", pos)
            if end < 0:
                raise _Truncated("string table overruns buffer")
            self.strings.append(buf[pos:end].decode("utf-8", errors="replace"))
            pos = end + 1
        self.types = bytearray(buf[pos:])
        self.draws: List[Dict[str, Any]] = []

    def _take4(self) -> int:
        if len(self.b4) < 4:
            raise _Truncated("Bytes4 exhausted")
        value = struct.unpack_from("<i", self.b4, 0)[0]
        del self.b4[:4]
        return value

    def _take8(self, fmt: str) -> Any:
        if len(self.b8) < 8:
            raise _Truncated("Bytes8 exhausted")
        value = struct.unpack(fmt, self.b8[:8])[0]
        del self.b8[:8]
        return value

    def _read_type(self) -> int:
        if not self.types:
            raise _Truncated("type stream exhausted")
        node = self.types.pop(0)
        if node & 0x80:
            node &= 0x7F
            if not self.types:
                raise _Truncated("type flag byte missing")
            self.types.pop(0)
        return node

    def _string_value(self, sid: int) -> str:
        if sid == -1:
            return ""
        if sid < 0 or sid >= len(self.strings):
            raise _Truncated(f"bad string id {sid}")
        return self.strings[sid]

    def read_value(self, node: int) -> Any:
        if node in _SINGLETONS:
            return _SINGLETONS[node]
        if node == _BOOLEAN:
            raise _Truncated("non-singleton boolean needs Bytes1")
        if node in (_INT32, _UINT32):
            return self._take4()
        if node == _STRING:
            return self._string_value(self._take4())
        if node == _BLOB:
            length = self._take4()
            if length != 0:
                raise _Truncated("non-empty v1 blob needs Bytes1")
            return b""
        if node == _ARRAY:
            return [self.read_member(True) for _ in range(self._take4())]
        if node == _ARRAY_TYPED:
            count = self._take4()
            sub = self._read_type()
            return [self.read_value(sub) for _ in range(count)]
        if node == _OBJECT:
            obj: Dict[str, Any] = {}
            for _ in range(self._take4()):
                key, value = self.read_member(False)
                obj[key] = value
            if "m_nIndexCount" in obj and "m_nVertexCount" in obj:
                self.draws.append(obj)
            return obj
        if node in (_INT64, _UINT64):
            return self._take8("<q")
        if node == _DOUBLE:
            return self._take8("<d")
        raise MdlParseError(f"Unsupported KV3 v1 node type {node}")

    def read_member(self, in_array: bool) -> Any:
        node = self._read_type()
        if in_array:
            return self.read_value(node)
        name = self._string_value(self._take4())
        return name, self.read_value(node)


def _read_blocks(data: bytes) -> Dict[str, Tuple[int, int]]:
    blocks: Dict[str, Tuple[int, int]] = {}
    for i in range(8):
        off = 16 + i * 12
        if off + 12 > len(data):
            break
        tag = data[off:off + 4].decode("ascii", errors="replace")
        boff, bsize = struct.unpack_from("<2I", data, off + 4)
        if 0 <= boff < len(data) and 0 < bsize <= len(data):
            blocks[tag] = (boff, min(bsize, len(data) - boff))
    return blocks


def _parse_mdat_draws(blob: bytes) -> Tuple[List[Dict[str, Any]], bool]:
    """Decompress one v1 MDAT blob, return complete draw calls + truncated flag."""
    if len(blob) < 40 or struct.unpack_from("<I", blob, 0)[0] != _KV3_V1_MAGIC:
        raise MdlParseError("MDAT KV3 blob is not version 1")
    _guid = blob[4:20]
    compression, c1, c4, c8, _utotal = struct.unpack_from("<5I", blob, 20)
    if compression != 1:
        raise MdlParseError(f"Unsupported KV3 v1 compression {compression}")
    if c1 != 0:
        raise MdlParseError("Unexpected Bytes1 content in MDAT blob")
    payload = blob[40:]
    buf = lz4_block_decompress(payload)
    reader = _V1Reader(buf, c4, c8)
    truncated = False
    try:
        root_type = reader._read_type()
        if root_type != _OBJECT:
            raise MdlParseError(f"MDAT root is not an object (type {root_type})")
        root = reader.read_value(root_type)
        if not isinstance(root, dict):
            raise MdlParseError("MDAT root did not parse to an object")
    except _Truncated:
        truncated = True
    draws = [d for d in reader.draws if isinstance(d, dict)]
    return draws, truncated


def _material_stem(path: str) -> str:
    stem = path.replace("\\", "/").rsplit("/", 1)[-1]
    return stem[:-5] if stem.lower().endswith(".vmat") else stem or "vmdl"


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


def parse_vmdl_c(path: str) -> SmdModel:
    """Parse a Source 2 compiled model into an SmdModel (static geometry)."""
    with open(path, "rb") as handle:
        data = handle.read()

    blocks = _read_blocks(data)
    if "MDAT" not in blocks or "MBUF" not in blocks:
        raise MdlParseError("Not a Source 2 model: MDAT/MBUF blocks missing")

    mdat_off, mdat_size = blocks["MDAT"]
    magic_at = data.find(b"\x01\x33\x56\x4b", mdat_off, mdat_off + 128)
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

    model = SmdModel()
    model.bones.append(SmdBone(bone_id=0, name="root", parent_id=-1))
    model.frames.append(SmdFrame(time=0, transforms={0: ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}))
    model.frame_names[0] = "bind"
    model.has_geometry = True

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
        for v in range(nverts):
            off = v * stride
            pos = struct.unpack_from("<3f", vdata, off)
            u0, v0 = struct.unpack_from("<2H", vdata, off + 12)
            nx, ny, nz, _nw = vdata[off + 16:off + 20]
            model.vertices.append(SmdVertex(
                parent_bone=0,
                position=(pos[0], pos[1], pos[2]),
                normal=(nx / 255 * 2 - 1, ny / 255 * 2 - 1, nz / 255 * 2 - 1),
                uv=(u0 / 65535.0, v0 / 65535.0),
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

    if model.vertices:
        xs = [v.position[0] for v in model.vertices]
        ys = [v.position[1] for v in model.vertices]
        zs = [v.position[2] for v in model.vertices]
        model.min_bound = (min(xs), min(ys), min(zs))
        model.max_bound = (max(xs), max(ys), max(zs))

    model.metadata["source"] = "vmdl_c"
    model.metadata["draw_calls"] = len(draws)
    if truncated:
        model.metadata["vmdl_truncated_tail"] = True
    return model
