import struct
import numpy as np
from typing import Optional, Tuple

# Image format enum values (Source SDK imageformat.h)
IMAGE_FORMAT_RGBA8888 = 0
IMAGE_FORMAT_ABGR8888 = 1
IMAGE_FORMAT_RGB888 = 2
IMAGE_FORMAT_BGR888 = 3
IMAGE_FORMAT_RGB565 = 4
IMAGE_FORMAT_I8 = 5
IMAGE_FORMAT_IA88 = 6
IMAGE_FORMAT_P8 = 7
IMAGE_FORMAT_A8 = 8
IMAGE_FORMAT_RGB888_BLUESCREEN = 9
IMAGE_FORMAT_BGR888_BLUESCREEN = 10
IMAGE_FORMAT_ARGB8888 = 11
IMAGE_FORMAT_BGRA8888 = 12
IMAGE_FORMAT_DXT1 = 13
IMAGE_FORMAT_DXT3 = 14
IMAGE_FORMAT_DXT5 = 15
IMAGE_FORMAT_BGRX8888 = 16
IMAGE_FORMAT_BGR565 = 17
IMAGE_FORMAT_BGRX5551 = 18
IMAGE_FORMAT_BGRA4444 = 19
IMAGE_FORMAT_DXT1_ONEBITALPHA = 20
IMAGE_FORMAT_BGRA5551 = 21
IMAGE_FORMAT_UV88 = 22
IMAGE_FORMAT_UVWQ8888 = 23
IMAGE_FORMAT_RGBA16161616F = 24
IMAGE_FORMAT_RGBA16161616 = 25
IMAGE_FORMAT_UVLX8888 = 26
IMAGE_FORMAT_R32F = 27
IMAGE_FORMAT_RGB323232F = 28
IMAGE_FORMAT_RGBA32323232F = 29
IMAGE_FORMAT_NV_DST16 = 30
IMAGE_FORMAT_NV_DST24 = 31
IMAGE_FORMAT_NV_INTZ = 32
IMAGE_FORMAT_NV_RAWZ = 33
IMAGE_FORMAT_ATI_DST16 = 34
IMAGE_FORMAT_ATI_DST24 = 35
IMAGE_FORMAT_NV_NULL = 36
IMAGE_FORMAT_ATI2N = 37
IMAGE_FORMAT_ATI1N = 38
IMAGE_FORMAT_NONE = -1

FORMAT_NAMES = {
    IMAGE_FORMAT_RGBA8888: "RGBA8888",
    IMAGE_FORMAT_ABGR8888: "ABGR8888",
    IMAGE_FORMAT_RGB888: "RGB888",
    IMAGE_FORMAT_BGR888: "BGR888",
    IMAGE_FORMAT_RGB565: "RGB565",
    IMAGE_FORMAT_I8: "I8",
    IMAGE_FORMAT_IA88: "IA88",
    IMAGE_FORMAT_P8: "P8",
    IMAGE_FORMAT_A8: "A8",
    IMAGE_FORMAT_RGB888_BLUESCREEN: "RGB888_BLUESCREEN",
    IMAGE_FORMAT_BGR888_BLUESCREEN: "BGR888_BLUESCREEN",
    IMAGE_FORMAT_ARGB8888: "ARGB8888",
    IMAGE_FORMAT_BGRA8888: "BGRA8888",
    IMAGE_FORMAT_DXT1: "DXT1",
    IMAGE_FORMAT_DXT3: "DXT3",
    IMAGE_FORMAT_DXT5: "DXT5",
    IMAGE_FORMAT_BGRX8888: "BGRX8888",
    IMAGE_FORMAT_BGR565: "BGR565",
    IMAGE_FORMAT_BGRX5551: "BGRX5551",
    IMAGE_FORMAT_BGRA4444: "BGRA4444",
    IMAGE_FORMAT_DXT1_ONEBITALPHA: "DXT1_ONEBITALPHA",
    IMAGE_FORMAT_BGRA5551: "BGRA5551",
    IMAGE_FORMAT_UV88: "UV88",
    IMAGE_FORMAT_UVWQ8888: "UVWQ8888",
    IMAGE_FORMAT_RGBA16161616F: "RGBA16161616F",
    IMAGE_FORMAT_RGBA16161616: "RGBA16161616",
    IMAGE_FORMAT_UVLX8888: "UVLX8888",
    IMAGE_FORMAT_R32F: "R32F",
    IMAGE_FORMAT_RGB323232F: "RGB323232F",
    IMAGE_FORMAT_RGBA32323232F: "RGBA32323232F",
    IMAGE_FORMAT_NV_DST16: "NV_DST16",
    IMAGE_FORMAT_NV_DST24: "NV_DST24",
    IMAGE_FORMAT_NV_INTZ: "NV_INTZ",
    IMAGE_FORMAT_NV_RAWZ: "NV_RAWZ",
    IMAGE_FORMAT_ATI_DST16: "ATI_DST16",
    IMAGE_FORMAT_ATI_DST24: "ATI_DST24",
    IMAGE_FORMAT_NV_NULL: "NV_NULL",
    IMAGE_FORMAT_ATI1N: "ATI1N",
    IMAGE_FORMAT_ATI2N: "ATI2N",
}


class VtfError(ValueError):
    pass


class VtfInfo:
    __slots__ = (
        "version_major", "version_minor", "width", "height", "depth",
        "flags", "frames", "first_frame", "mipmap_count",
        "high_res_format", "low_res_format",
        "low_res_width", "low_res_height",
        "bumpmap_scale", "reflectivity",
        "num_resources", "header_size",
    )

    def __init__(self):
        self.version_major = 7
        self.version_minor = 0
        self.width = 0
        self.height = 0
        self.depth = 1
        self.flags = 0
        self.frames = 1
        self.first_frame = 0
        self.mipmap_count = 1
        self.high_res_format = IMAGE_FORMAT_NONE
        self.low_res_format = IMAGE_FORMAT_NONE
        self.low_res_width = 0
        self.low_res_height = 0
        self.bumpmap_scale = 1.0
        self.reflectivity = (0.0, 0.0, 0.0)
        self.num_resources = 0
        self.header_size = 80


def _block_size(fmt, width, height):
    """Return byte size for one 4x4 block or one row of uncompressed data."""
    if fmt in (IMAGE_FORMAT_DXT1, IMAGE_FORMAT_DXT1_ONEBITALPHA):
        return 8
    if fmt in (IMAGE_FORMAT_DXT3, IMAGE_FORMAT_DXT5,
               IMAGE_FORMAT_ATI1N, IMAGE_FORMAT_ATI2N):
        return 16
    return 0


def _mip_image_size(fmt, width, height):
    """Compute byte size for one mip level of one face/frame."""
    w = max(1, width)
    h = max(1, height)

    if fmt in (IMAGE_FORMAT_DXT1, IMAGE_FORMAT_DXT1_ONEBITALPHA):
        bw = max(1, (w + 3) // 4)
        bh = max(1, (h + 3) // 4)
        return bw * bh * 8

    if fmt in (IMAGE_FORMAT_DXT3, IMAGE_FORMAT_DXT5):
        bw = max(1, (w + 3) // 4)
        bh = max(1, (h + 3) // 4)
        return bw * bh * 16

    if fmt in (IMAGE_FORMAT_ATI1N,):
        bw = max(1, (w + 3) // 4)
        bh = max(1, (h + 3) // 4)
        return bw * bh * 8

    if fmt in (IMAGE_FORMAT_ATI2N,):
        bw = max(1, (w + 3) // 4)
        bh = max(1, (h + 3) // 4)
        return bw * bh * 16

    # Uncompressed formats: bytes per pixel
    bpp = _bytes_per_pixel(fmt)
    return w * h * bpp


def _bytes_per_pixel(fmt):
    if fmt in (IMAGE_FORMAT_RGBA8888, IMAGE_FORMAT_ABGR8888,
               IMAGE_FORMAT_ARGB8888, IMAGE_FORMAT_BGRA8888,
               IMAGE_FORMAT_BGRX8888):
        return 4
    if fmt in (IMAGE_FORMAT_RGB888, IMAGE_FORMAT_BGR888,
               IMAGE_FORMAT_RGB888_BLUESCREEN, IMAGE_FORMAT_BGR888_BLUESCREEN):
        return 3
    if fmt in (IMAGE_FORMAT_RGB565, IMAGE_FORMAT_BGR565,
               IMAGE_FORMAT_IA88, IMAGE_FORMAT_BGRA4444,
               IMAGE_FORMAT_BGRA5551, IMAGE_FORMAT_BGRX5551,
               IMAGE_FORMAT_UV88):
        return 2
    if fmt in (IMAGE_FORMAT_I8, IMAGE_FORMAT_A8, IMAGE_FORMAT_P8):
        return 1
    if fmt in (IMAGE_FORMAT_RGBA16161616, IMAGE_FORMAT_RGBA16161616F,
               IMAGE_FORMAT_UVWQ8888, IMAGE_FORMAT_UVLX8888):
        return 8
    if fmt == IMAGE_FORMAT_R32F:
        return 4
    if fmt == IMAGE_FORMAT_RGB323232F:
        return 12
    if fmt == IMAGE_FORMAT_RGBA32323232F:
        return 16
    return 4  # fallback


def parse_vtf(path: str) -> Tuple[VtfInfo, np.ndarray]:
    """
    Parse a VTF file and return (info, rgba_array).
    rgba_array is shape (height, width, 4), dtype uint8.
    Decodes the largest mipmap, frame 0, face 0.
    """
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 16:
        raise VtfError("File too small to be a VTF")

    signature = data[0:4]
    if signature not in (b"VTF\x00", b"VTFX"):
        raise VtfError(f"Invalid VTF signature: {signature!r}")

    version_major, version_minor = struct.unpack_from("<II", data, 4)
    if version_major != 7:
        raise VtfError(f"Unsupported VTF major version {version_major}")

    info = VtfInfo()
    info.version_major = version_major
    info.version_minor = version_minor

    if version_minor <= 1:
        # Very old format, minimal header
        info.width, info.height = struct.unpack_from("<HH", data, 12)
        info.flags, = struct.unpack_from("<I", data, 16)
        info.frames, info.first_frame = struct.unpack_from("<HH", data, 20)
        # Skip reflectivity (12 bytes) + padding
        offset = 36
        info.bumpmap_scale, = struct.unpack_from("<f", data, offset)
        offset += 4
        info.high_res_format, = struct.unpack_from("<i", data, offset)
        offset += 4
        info.mipmap_count, = struct.unpack_from("<B", data, offset)
        offset += 1
        info.low_res_format, = struct.unpack_from("<i", data, offset)
        offset += 4
        info.low_res_width, = struct.unpack_from("<B", data, offset)
        offset += 1
        info.low_res_height, = struct.unpack_from("<B", data, offset)
        offset += 1
        info.header_size = offset
        image_data_start = offset
    else:
        # 7.2+ standard layout
        header_size, = struct.unpack_from("<I", data, 12)
        info.header_size = header_size
        info.width, info.height = struct.unpack_from("<HH", data, 16)
        info.flags, = struct.unpack_from("<I", data, 20)
        info.frames, info.first_frame = struct.unpack_from("<HH", data, 24)
        # padding0 (4 bytes) at 28
        info.reflectivity = struct.unpack_from("<fff", data, 32)
        # padding1 (4 bytes) at 44
        info.bumpmap_scale, = struct.unpack_from("<f", data, 48)
        info.high_res_format, = struct.unpack_from("<i", data, 52)
        info.mipmap_count, = struct.unpack_from("<B", data, 56)
        info.low_res_format, = struct.unpack_from("<i", data, 57)
        info.low_res_width, = struct.unpack_from("<B", data, 61)
        info.low_res_height, = struct.unpack_from("<B", data, 62)
        info.depth, = struct.unpack_from("<H", data, 63)
        if info.depth == 0:
            info.depth = 1

        if version_minor >= 3:
            # padding2 (3 bytes) at 65
            info.num_resources, = struct.unpack_from("<I", data, 68)
            # padding3 (8 bytes) at 72
            # Resource entries start at 80
            image_data_start = _skip_resources_v73(data, info)
        else:
            # 7.2: image data starts at header_size offset
            image_data_start = header_size

    if info.high_res_format == IMAGE_FORMAT_NONE:
        raise VtfError("VTF has no high-resolution image data")

    # Navigate to the largest mipmap (last one stored)
    # Order: mipmaps smallest->largest, then frames, then faces, then z-slices
    offset = image_data_start

    # 7.2 and older store the thumbnail directly before the mip chain.
    # 7.3+ keeps it as a separate resource, and the 0x30 resource offset
    # already points past it, so skipping again would shift the decode.
    if version_minor < 3:
        if info.low_res_format != IMAGE_FORMAT_NONE and info.low_res_width > 0:
            low_size = _mip_image_size(
                info.low_res_format, info.low_res_width, info.low_res_height
            )
            offset += low_size

    # Skip all mipmaps except the last (largest)
    for mip in range(info.mipmap_count - 1):
        mip_w = max(1, info.width >> (info.mipmap_count - 1 - mip))
        mip_h = max(1, info.height >> (info.mipmap_count - 1 - mip))
        mip_d = max(1, info.depth >> (info.mipmap_count - 1 - mip))
        size = _mip_image_size(info.high_res_format, mip_w, mip_h) * mip_d
        size *= info.frames
        face_count = _face_count(info)
        size *= face_count
        offset += size

    # Now at the largest mipmap. Skip to frame 0, face 0.
    # (They're stored in order, so frame 0 face 0 is first)
    target_w = info.width
    target_h = info.height
    target_d = info.depth
    image_size = _mip_image_size(info.high_res_format, target_w, target_h)

    if offset + image_size > len(data):
        raise VtfError(
            f"Image data extends past end of file "
            f"(need {offset + image_size}, have {len(data)})"
        )

    raw = data[offset:offset + image_size]
    rgba = _decode_image(raw, info.high_res_format, target_w, target_h)
    return info, rgba


def _face_count(info):
    """Determine number of faces (6 for cubemaps, 1 otherwise)."""
    if info.flags & 0x4000:  # TEXTUREFLAGS_ENVMAP
        if info.version_minor < 5 and info.first_frame == 0xFFFF:
            return 7  # old-style spheremap + 6 faces
        return 6
    return 1


def _skip_resources_v73(data, info):
    """
    Parse the resource dictionary (VTF 7.3+) and return the offset
    where high-res image data begins.
    """
    # Resource entries start at byte 80 in the header.
    # Each entry is 8 bytes: tag[3] + flags[1] + offset[4]
    resource_start = 80
    entry_size = 8
    max_offset = info.header_size  # at minimum, data starts after header

    for i in range(info.num_resources):
        entry_offset = resource_start + i * entry_size
        if entry_offset + entry_size > len(data):
            break
        tag = data[entry_offset:entry_offset + 3]
        res_flags = data[entry_offset + 3]
        res_offset, = struct.unpack_from("<I", data, entry_offset + 4)

        # Tag {0x30, 0x00, 0x00} is the high-res image data resource
        # Tag {0x01, 0x00, 0x00} is the low-res thumbnail
        # We just track the maximum offset to find where data ends
        if tag == b"\x30\x00\x00":
            # High-res image: this IS the image data, offset points to it
            return res_offset

    # Fallback: assume image data right after header
    return info.header_size


def _decode_image(raw, fmt, width, height):
    """Decode raw image bytes into RGBA uint8 array."""
    if fmt in (IMAGE_FORMAT_DXT1, IMAGE_FORMAT_DXT1_ONEBITALPHA):
        return _decode_dxt1(raw, width, height)
    if fmt == IMAGE_FORMAT_DXT3:
        return _decode_dxt3(raw, width, height)
    if fmt == IMAGE_FORMAT_DXT5:
        return _decode_dxt5(raw, width, height)
    if fmt == IMAGE_FORMAT_BGRA8888:
        return _decode_bgra8888(raw, width, height)
    if fmt == IMAGE_FORMAT_BGRX8888:
        return _decode_bgrx8888(raw, width, height)
    if fmt in (IMAGE_FORMAT_BGR888, IMAGE_FORMAT_BGR888_BLUESCREEN):
        return _decode_bgr888(raw, width, height)
    if fmt in (IMAGE_FORMAT_RGB888, IMAGE_FORMAT_RGB888_BLUESCREEN):
        return _decode_rgb888(raw, width, height)
    if fmt in (IMAGE_FORMAT_RGB565, IMAGE_FORMAT_BGR565):
        return _decode_565(raw, width, height)
    if fmt == IMAGE_FORMAT_BGRA4444:
        return _decode_bgra4444(raw, width, height)
    if fmt in (IMAGE_FORMAT_BGRA5551, IMAGE_FORMAT_BGRX5551):
        return _decode_bgra5551(raw, width, height)
    if fmt == IMAGE_FORMAT_RGBA8888:
        return _decode_rgba8888(raw, width, height)
    if fmt == IMAGE_FORMAT_ABGR8888:
        return _decode_abgr8888(raw, width, height)
    if fmt == IMAGE_FORMAT_ARGB8888:
        return _decode_argb8888(raw, width, height)
    if fmt == IMAGE_FORMAT_I8:
        return _decode_i8(raw, width, height)
    if fmt == IMAGE_FORMAT_IA88:
        return _decode_ia88(raw, width, height)
    if fmt == IMAGE_FORMAT_A8:
        return _decode_a8(raw, width, height)
    if fmt == IMAGE_FORMAT_UV88:
        return _decode_uv88(raw, width, height)

    raise VtfError(
        f"Unsupported VTF image format: "
        f"{FORMAT_NAMES.get(fmt, f'unknown({fmt})')}"
    )


# --- DXT decoders ---

def _rgb565_to_rgb(value):
    r = ((value >> 11) & 0x1F) * 255 // 31
    g = ((value >> 5) & 0x3F) * 255 // 63
    b = (value & 0x1F) * 255 // 31
    return r, g, b


def _decode_dxt1(raw, width, height):
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    out = np.zeros((height, width, 4), dtype=np.uint8)

    for by in range(bh):
        for bx in range(bw):
            block_offset = (by * bw + bx) * 8
            if block_offset + 8 > len(raw):
                break
            c0, c1 = struct.unpack_from("<HH", raw, block_offset)
            lookup, = struct.unpack_from("<I", raw, block_offset + 4)

            r0, g0, b0 = _rgb565_to_rgb(c0)
            r1, g1, b1 = _rgb565_to_rgb(c1)

            if c0 > c1:
                colors = np.array([
                    [r0, g0, b0, 255],
                    [r1, g1, b1, 255],
                    [(2 * r0 + r1) // 3, (2 * g0 + g1) // 3,
                     (2 * b0 + b1) // 3, 255],
                    [(r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3,
                     (b0 + 2 * b1) // 3, 255],
                ], dtype=np.uint8)
            else:
                colors = np.array([
                    [r0, g0, b0, 255],
                    [r1, g1, b1, 255],
                    [(r0 + r1) // 2, (g0 + g1) // 2, (b0 + b1) // 2, 255],
                    [0, 0, 0, 0],
                ], dtype=np.uint8)

            for py in range(4):
                for px in range(4):
                    x = bx * 4 + px
                    y = by * 4 + py
                    if x >= width or y >= height:
                        continue
                    idx = (lookup >> ((py * 4 + px) * 2)) & 3
                    out[y, x] = colors[idx]

    return out


def _decode_dxt3(raw, width, height):
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    out = np.zeros((height, width, 4), dtype=np.uint8)

    for by in range(bh):
        for bx in range(bw):
            block_offset = (by * bw + bx) * 16
            if block_offset + 16 > len(raw):
                break

            # Alpha: 8 bytes, 4 bits per pixel
            alpha_data = raw[block_offset:block_offset + 8]
            alphas = np.zeros(16, dtype=np.uint8)
            for i in range(8):
                alphas[i * 2] = (alpha_data[i] & 0x0F) * 17
                alphas[i * 2 + 1] = ((alpha_data[i] >> 4) & 0x0F) * 17

            # Color: DXT1-style
            c0, c1 = struct.unpack_from("<HH", raw, block_offset + 8)
            lookup, = struct.unpack_from("<I", raw, block_offset + 12)

            r0, g0, b0 = _rgb565_to_rgb(c0)
            r1, g1, b1 = _rgb565_to_rgb(c1)
            colors = np.array([
                [r0, g0, b0],
                [r1, g1, b1],
                [(2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3],
                [(r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3],
            ], dtype=np.uint8)

            for py in range(4):
                for px in range(4):
                    x = bx * 4 + px
                    y = by * 4 + py
                    if x >= width or y >= height:
                        continue
                    pixel_index = py * 4 + px
                    idx = (lookup >> (pixel_index * 2)) & 3
                    out[y, x, :3] = colors[idx]
                    out[y, x, 3] = alphas[pixel_index]

    return out


def _decode_dxt5(raw, width, height):
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    out = np.zeros((height, width, 4), dtype=np.uint8)

    for by in range(bh):
        for bx in range(bw):
            block_offset = (by * bw + bx) * 16
            if block_offset + 16 > len(raw):
                break

            # Alpha block
            a0 = raw[block_offset]
            a1 = raw[block_offset + 1]
            alpha_bits = int.from_bytes(raw[block_offset + 2:block_offset + 8], "little")

            if a0 > a1:
                alpha_lut = [a0, a1]
                for i in range(1, 7):
                    alpha_lut.append(((7 - i) * a0 + i * a1) // 7)
            else:
                alpha_lut = [a0, a1]
                for i in range(1, 5):
                    alpha_lut.append(((5 - i) * a0 + i * a1) // 5)
                alpha_lut.append(0)
                alpha_lut.append(255)

            alphas = np.zeros(16, dtype=np.uint8)
            for i in range(16):
                idx = (alpha_bits >> (i * 3)) & 7
                alphas[i] = alpha_lut[idx]

            # Color block
            c0, c1 = struct.unpack_from("<HH", raw, block_offset + 8)
            lookup, = struct.unpack_from("<I", raw, block_offset + 12)

            r0, g0, b0 = _rgb565_to_rgb(c0)
            r1, g1, b1 = _rgb565_to_rgb(c1)
            colors = np.array([
                [r0, g0, b0],
                [r1, g1, b1],
                [(2 * r0 + r1) // 3, (2 * g0 + g1) // 3, (2 * b0 + b1) // 3],
                [(r0 + 2 * r1) // 3, (g0 + 2 * g1) // 3, (b0 + 2 * b1) // 3],
            ], dtype=np.uint8)

            for py in range(4):
                for px in range(4):
                    x = bx * 4 + px
                    y = by * 4 + py
                    if x >= width or y >= height:
                        continue
                    pixel_index = py * 4 + px
                    idx = (lookup >> (pixel_index * 2)) & 3
                    out[y, x, :3] = colors[idx]
                    out[y, x, 3] = alphas[pixel_index]

    return out


# --- Uncompressed decoders ---

def _decode_bgra8888(raw, width, height):
    expected = width * height * 4
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 4)
    # BGRA -> RGBA
    out = np.empty_like(arr)
    out[:, :, 0] = arr[:, :, 2]
    out[:, :, 1] = arr[:, :, 1]
    out[:, :, 2] = arr[:, :, 0]
    out[:, :, 3] = arr[:, :, 3]
    return out


def _decode_bgrx8888(raw, width, height):
    expected = width * height * 4
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 4)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 0] = arr[:, :, 2]
    out[:, :, 1] = arr[:, :, 1]
    out[:, :, 2] = arr[:, :, 0]
    out[:, :, 3] = 255
    return out


def _decode_bgr888(raw, width, height):
    expected = width * height * 3
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 3)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 0] = arr[:, :, 2]
    out[:, :, 1] = arr[:, :, 1]
    out[:, :, 2] = arr[:, :, 0]
    out[:, :, 3] = 255
    return out


def _decode_rgb888(raw, width, height):
    expected = width * height * 3
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 3)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, :3] = arr
    out[:, :, 3] = 255
    return out


def _decode_565(raw, width, height):
    expected = width * height * 2
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    pixels = np.frombuffer(raw[:expected], dtype=np.uint16).reshape(height, width)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 0] = ((pixels >> 11) & 0x1F) * 255 // 31
    out[:, :, 1] = ((pixels >> 5) & 0x3F) * 255 // 63
    out[:, :, 2] = (pixels & 0x1F) * 255 // 31
    out[:, :, 3] = 255
    return out


def _decode_bgra4444(raw, width, height):
    expected = width * height * 2
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    pixels = np.frombuffer(raw[:expected], dtype=np.uint16).reshape(height, width)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 2] = (pixels & 0x0F) * 17        # B
    out[:, :, 1] = ((pixels >> 4) & 0x0F) * 17  # G
    out[:, :, 0] = ((pixels >> 8) & 0x0F) * 17  # R
    out[:, :, 3] = ((pixels >> 12) & 0x0F) * 17 # A
    return out


def _decode_bgra5551(raw, width, height):
    expected = width * height * 2
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    pixels = np.frombuffer(raw[:expected], dtype=np.uint16).reshape(height, width)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 2] = (pixels & 0x1F) * 255 // 31
    out[:, :, 1] = ((pixels >> 5) & 0x1F) * 255 // 31
    out[:, :, 0] = ((pixels >> 10) & 0x1F) * 255 // 31
    out[:, :, 3] = ((pixels >> 15) & 0x01) * 255
    return out


def _decode_rgba8888(raw, width, height):
    expected = width * height * 4
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    return np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 4).copy()


def _decode_abgr8888(raw, width, height):
    expected = width * height * 4
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 4)
    # ABGR -> RGBA
    out = np.empty_like(arr)
    out[:, :, 0] = arr[:, :, 3]
    out[:, :, 1] = arr[:, :, 2]
    out[:, :, 2] = arr[:, :, 1]
    out[:, :, 3] = arr[:, :, 0]
    return out


def _decode_argb8888(raw, width, height):
    expected = width * height * 4
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 4)
    # ARGB -> RGBA
    out = np.empty_like(arr)
    out[:, :, 0] = arr[:, :, 1]
    out[:, :, 1] = arr[:, :, 2]
    out[:, :, 2] = arr[:, :, 3]
    out[:, :, 3] = arr[:, :, 0]
    return out


def _decode_i8(raw, width, height):
    expected = width * height
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    gray = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 0] = gray
    out[:, :, 1] = gray
    out[:, :, 2] = gray
    out[:, :, 3] = 255
    return out


def _decode_ia88(raw, width, height):
    expected = width * height * 2
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 2)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 0] = arr[:, :, 0]
    out[:, :, 1] = arr[:, :, 0]
    out[:, :, 2] = arr[:, :, 0]
    out[:, :, 3] = arr[:, :, 1]
    return out


def _decode_a8(raw, width, height):
    expected = width * height
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    alpha = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 0] = 255
    out[:, :, 1] = 255
    out[:, :, 2] = 255
    out[:, :, 3] = alpha
    return out


def _decode_uv88(raw, width, height):
    expected = width * height * 2
    if len(raw) < expected:
        raw = raw + b"\x00" * (expected - len(raw))
    arr = np.frombuffer(raw[:expected], dtype=np.uint8).reshape(height, width, 2)
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[:, :, 0] = arr[:, :, 0]
    out[:, :, 1] = arr[:, :, 1]
    out[:, :, 2] = 255
    out[:, :, 3] = 255
    return out