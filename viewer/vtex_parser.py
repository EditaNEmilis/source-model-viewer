"""Source 2 compiled texture (.vtex_c) import.

Layout of the DATA block (ValveResourceFormat ``Texture``):

* 40 byte header: version, flags, reflectivity, width, height, depth,
  format, mip count, picmip, extra-data offset/count.
* extra-data entries (12 bytes each); type 4 carries per-mip LZ4 sizes.
* the pixel data is appended after the DATA block (at the file-size
  field), smallest mip first, mip 0 last.

Compressed formats are handed to the renderer as their original BC
blocks, which OpenGL uploads with ``glCompressedTexImage2D``. Only
uncompressed formats are expanded here.
"""

import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

from viewer.source2 import Source2Error, read_resource_blocks, lz4_block_decompress
from viewer.vtf_parser import (
    _decode_a8,
    _decode_bgra8888,
    _decode_i8,
    _decode_ia88,
    _decode_rgba8888,
)

# VTexFormat (ValveResourceFormat).
FORMAT_NAMES = {
    0: "UNKNOWN",
    1: "DXT1",
    2: "DXT5",
    3: "I8",
    4: "RGBA8888",
    5: "R16",
    6: "RG1616",
    7: "RGBA16161616",
    8: "R16F",
    9: "RG1616F",
    10: "RGBA16161616F",
    11: "R32F",
    12: "RG3232F",
    13: "RGB323232F",
    14: "RGBA32323232F",
    15: "JPEG_RGBA8888",
    16: "PNG_RGBA8888",
    17: "JPEG_DXT5",
    18: "PNG_DXT5",
    19: "BC6H",
    20: "BC7",
    21: "ATI2N",
    22: "IA88",
    23: "ETC2",
    24: "ETC2_EAC",
    25: "R11_EAC",
    26: "RG11_EAC",
    27: "ATI1N",
    28: "BGRA8888",
    29: "WEBP_RGBA8888",
    30: "WEBP_DXT5",
    31: "R32_UINT",
    32: "A8",
    33: "R8_UNORM",
}

# Format id -> (kind, bytes per 4x4 block).
_BLOCK_FORMATS = {
    1: ("dxt1", 8),
    2: ("dxt5", 16),
    20: ("bc7", 16),
    21: ("bc5", 16),
    27: ("bc4", 8),
}

_BYTES_PER_PIXEL = {
    3: 1,     # I8
    4: 4,     # RGBA8888
    22: 2,    # IA88
    28: 4,    # BGRA8888
    32: 1,    # A8
    33: 1,    # R8_UNORM
}

# VTexFlags.
_FLAG_CUBE = 1 << 4
_FLAG_VOLUME = 1 << 5
_FLAG_ARRAY = 1 << 6

# VTexExtraData.
_EXTRA_COMPRESSED_MIP_SIZE = 4


class VtexError(ValueError):
    pass


class VtexTexture:
    """One compiled texture: either RGBA pixels or BC blocks."""

    __slots__ = (
        "source_path",
        "width",
        "height",
        "depth",
        "version",
        "flags",
        "format",
        "format_name",
        "mip_levels",
        "rgba",
        "compression",
        "compressed_mips",
    )

    def __init__(self) -> None:
        self.source_path = ""
        self.width = 0
        self.height = 0
        self.depth = 1
        self.version = 1
        self.flags = 0
        self.format = 0
        self.format_name = "UNKNOWN"
        self.mip_levels = 1
        self.rgba: Optional[np.ndarray] = None
        self.compression: Optional[str] = None
        self.compressed_mips: Optional[List[Tuple[int, int, int, bytes]]] = None


def is_vtex_c_file(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            data = handle.read(64)
    except OSError:
        return False
    if len(data) < 40:
        return False
    return data[16:20] == b"REDI" and data[28:32] == b"DATA"


def _mip_size(fmt: int, width: int, height: int) -> int:
    block = _BLOCK_FORMATS.get(fmt)
    if block is not None:
        blocks_x = max(1, (max(1, width) + 3) // 4)
        blocks_y = max(1, (max(1, height) + 3) // 4)
        return blocks_x * blocks_y * block[1]
    return max(1, width) * max(1, height) * _BYTES_PER_PIXEL.get(fmt, 4)


def _read_extra_data(
    data: bytes, header_start: int, extra_offset: int, extra_count: int
) -> Dict[int, Tuple[int, int]]:
    entries: Dict[int, Tuple[int, int]] = {}
    if extra_count <= 0:
        return entries

    position = header_start + 32 + extra_offset
    for _ in range(extra_count):
        if position + 12 > len(data):
            break
        entry_type, offset, size = struct.unpack_from("<III", data, position)
        payload = position + 4 + offset
        if 0 <= payload <= len(data) and size <= len(data) - payload:
            entries[entry_type] = (payload, size)
        position += 12
    return entries


def _read_compressed_mip_sizes(
    data: bytes, extra: Dict[int, Tuple[int, int]]
) -> Optional[List[int]]:
    entry = extra.get(_EXTRA_COMPRESSED_MIP_SIZE)
    if entry is None:
        return None

    payload, size = entry
    if size < 12:
        raise VtexError("Compressed mip table is too small")
    method, mip_offset, count = struct.unpack_from("<III", data, payload)
    if method == 0:
        return None
    if method != 1:
        raise VtexError(f"Unsupported mip compression method {method}")
    if count > 16:
        raise VtexError(f"Implausible compressed mip count {count}")

    start = payload + 4 + mip_offset
    if start + count * 4 > len(data):
        raise VtexError("Compressed mip table overruns the file")
    return list(struct.unpack_from(f"<{count}i", data, start))


def _read_mips(
    data: bytes,
    pixel_start: int,
    width: int,
    height: int,
    fmt: int,
    mip_count: int,
    compressed_sizes: Optional[List[int]],
) -> List[Tuple[int, int, int, bytes]]:
    mips: List[Tuple[int, int, int, bytes]] = []
    position = pixel_start
    for level in range(mip_count - 1, -1, -1):
        mip_width = max(1, width >> level)
        mip_height = max(1, height >> level)
        raw_size = _mip_size(fmt, mip_width, mip_height)

        stored = raw_size
        if compressed_sizes is not None and level < len(compressed_sizes):
            candidate = compressed_sizes[level]
            if 0 < candidate < raw_size:
                stored = candidate

        chunk = data[position:position + stored]
        if len(chunk) != stored:
            raise VtexError(
                f"Mip {level} overruns the file "
                f"(need {stored}, have {len(chunk)})"
            )
        position += stored

        if stored != raw_size:
            chunk = lz4_block_decompress(chunk)
            if len(chunk) != raw_size:
                raise VtexError(
                    f"Mip {level} decompressed to {len(chunk)} bytes, "
                    f"expected {raw_size}"
                )
        mips.append((level, mip_width, mip_height, chunk))
    return mips


def _decode_uncompressed(
    fmt: int, raw: bytes, width: int, height: int
) -> np.ndarray:
    if fmt == 4:
        return _decode_rgba8888(raw, width, height)
    if fmt == 28:
        return _decode_bgra8888(raw, width, height)
    if fmt == 3:
        return _decode_i8(raw, width, height)
    if fmt == 22:
        return _decode_ia88(raw, width, height)
    if fmt == 32:
        return _decode_a8(raw, width, height)
    if fmt == 33:
        expected = width * height
        pixels = np.frombuffer(raw[:expected], dtype=np.uint8)
        out = np.empty((height, width, 4), dtype=np.uint8)
        gray = pixels.reshape(height, width)
        out[:, :, 0] = gray
        out[:, :, 1] = gray
        out[:, :, 2] = gray
        out[:, :, 3] = 255
        return out
    raise VtexError(
        f"Unsupported uncompressed VTEX format "
        f"{FORMAT_NAMES.get(fmt, f'unknown({fmt})')}"
    )


def parse_vtex_c(path: str) -> VtexTexture:
    """Parse a Source 2 compiled texture.

    Compressed formats keep their BC blocks in ``compressed_mips``;
    uncompressed formats fill ``rgba`` with the largest mip.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    try:
        blocks = read_resource_blocks(data)
    except Source2Error as error:
        raise VtexError(str(error)) from error

    if "DATA" not in blocks:
        raise VtexError("VTEX DATA block missing")

    header_start, data_size = blocks["DATA"]
    if header_start + 40 > len(data):
        raise VtexError("VTEX header overruns the file")

    version, flags = struct.unpack_from("<HH", data, header_start)
    if version != 1:
        raise VtexError(f"Unsupported VTEX version {version}")

    width, height, depth = struct.unpack_from("<3H", data, header_start + 20)
    fmt = data[header_start + 26]
    mip_count = data[header_start + 27]
    extra_offset, extra_count = struct.unpack_from("<II", data, header_start + 32)

    if width < 1 or height < 1:
        raise VtexError(f"Invalid VTEX dimensions {width}x{height}")
    if not 1 <= mip_count <= 16:
        raise VtexError(f"Invalid VTEX mip count {mip_count}")
    if flags & _FLAG_CUBE:
        raise VtexError("Cubemap VTEX textures are not supported")
    if flags & _FLAG_VOLUME or flags & _FLAG_ARRAY or depth > 1:
        raise VtexError("Volume/array VTEX textures are not supported")
    if fmt not in _BLOCK_FORMATS and fmt not in _BYTES_PER_PIXEL:
        raise VtexError(
            f"Unsupported VTEX format {FORMAT_NAMES.get(fmt, f'unknown({fmt})')}"
        )

    texture = VtexTexture()
    texture.source_path = path
    texture.width = width
    texture.height = height
    texture.depth = depth
    texture.version = version
    texture.flags = flags
    texture.format = fmt
    texture.format_name = FORMAT_NAMES.get(fmt, f"UNKNOWN({fmt})")
    texture.mip_levels = mip_count

    extra = _read_extra_data(data, header_start, extra_offset, extra_count)
    compressed_sizes = _read_compressed_mip_sizes(data, extra)

    try:
        mips = _read_mips(
            data,
            header_start + data_size,
            width,
            height,
            fmt,
            mip_count,
            compressed_sizes,
        )
    except Source2Error as error:
        raise VtexError(str(error)) from error

    # mips arrive largest last; reorder to level 0 first.
    mips.sort(key=lambda mip: mip[0])

    block = _BLOCK_FORMATS.get(fmt)
    if block is not None:
        texture.compression = block[0]
        texture.compressed_mips = mips
    else:
        texture.rgba = _decode_uncompressed(fmt, mips[0][3], width, height)

    return texture
