### Compiled MDL skeletal animation extraction ###

import math
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

from viewer.skeleton import quat_to_euler
from viewer.smd_parser import SmdBone, SmdFrame, SmdModel


Transform = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
Quat = Tuple[float, float, float, float]

# studiohdr_t fields used here
ANIM_COUNT_OFFSET = 0xB4
ANIM_INDEX_OFFSET = 0xB8
STUDIOHDR2_OFFSET_FIELD = 0x158

# mstudioanimdesc_t (100 bytes on v44-v48 branches, including the section
# and zero-frame tail)
ANIMDESC_SIZE = 100

# mstudioseqdesc_t (212 bytes on v44-v48 branches)
SEQDESC_SIZE = 212
SEQ_COUNT_OFFSET = 0xBC
SEQ_INDEX_OFFSET = 0xC0
ANIMDESC_ANIMBLOCK_OFFSET = 52
ANIMDESC_ANIMINDEX_OFFSET = 56

# Per bone animation flags (mstudioanim_t, Source SDK studio.h)
STUDIO_ANIM_RAWPOS = 0x01       # vector48 half floats, absolute
STUDIO_ANIM_RAWROT = 0x02       # quaternion48, absolute
STUDIO_ANIM_ANIMPOS = 0x04      # 3 value pointers, position deltas
STUDIO_ANIM_ANIMROT = 0x08      # 3 value pointers, rotation angles
STUDIO_ANIM_DELTA = 0x10        # values are deltas from the bind pose
STUDIO_ANIM_RAWROT2 = 0x20      # quaternion64, absolute

ROT_SCALE = (2.0 * math.pi) / 65536.0  # == pi / 32768
POS_SCALE = 1.0 / 64.0

MAX_SAFE_FRAMES = 100000
MAX_SAFE_BONES = 4096


def _read_cstring(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\x00", offset)
    if end < 0:
        return data[offset:].decode("latin-1", errors="ignore").strip()
    return data[offset:end].decode("latin-1", errors="ignore").strip()


### Quaternion helpers (matching skeleton.euler_to_matrix: Rz * Ry * Rx) ###

def quat_from_euler(euler) -> Quat:
    rx, ry, rz = euler

    cx = math.cos(rx * 0.5)
    sx = math.sin(rx * 0.5)
    cy = math.cos(ry * 0.5)
    sy = math.sin(ry * 0.5)
    cz = math.cos(rz * 0.5)
    sz = math.sin(rz * 0.5)

    qx = (sx, 0.0, 0.0, cx)
    qy = (0.0, sy, 0.0, cy)
    qz = (0.0, 0.0, sz, cz)

    return quat_mul(quat_mul(qz, qy), qx)


def quat_mul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


### Compressed value decoding ###

def _unpack_scaled(raw: int, bits: int) -> float:
    return raw / float((1 << (bits - 1)) - 1) - 1.0


def _decode_quat64(raw: bytes) -> Quat:
    bits = int.from_bytes(raw, "little")

    x = _unpack_scaled(bits & 0x7FF, 11)
    y = _unpack_scaled((bits >> 11) & 0x7FF, 11)
    z = _unpack_scaled((bits >> 22) & 0x3FF, 10)
    wneg = (bits >> 32) & 1

    t = 1.0 - x * x - y * y - z * z
    w = math.sqrt(t) if t > 0.0 else 0.0
    if wneg:
        w = -w

    return (x, y, z, w)


def _decode_quat48(raw: bytes) -> Quat:
    bits = int.from_bytes(raw, "little")

    x = _unpack_scaled(bits & 0x7FFF, 15)
    y = _unpack_scaled((bits >> 15) & 0x7FFF, 15)
    z = _unpack_scaled((bits >> 30) & 0x7FFF, 15)
    wneg = (bits >> 45) & 1

    t = max(0.0, 1.0 - x * x - y * y - z * z)
    w = math.sqrt(t)
    if wneg:
        w = -w

    return (x, y, z, w)


def _decode_vector48(raw: bytes) -> Tuple[float, float, float]:
    if len(raw) < 6:
        return (0.0, 0.0, 0.0)
    values = np.frombuffer(raw[:6], dtype="<f2").astype(np.float32)
    return (float(values[0]), float(values[1]), float(values[2]))


### RLE frame channels ###

def _read_channel(buf: bytes, pos: int, num_frames: int) -> Tuple[list, int]:
    """Read one RLE channel, a chain of spans covering num_frames."""
    spans = []
    used = 0
    covered = 0

    while covered < num_frames:
        if pos + used + 2 > len(buf):
            break

        valid, total = struct.unpack_from("<2B", buf, pos + used)
        used += 2

        if total == 0:
            break

        values = []
        if valid > 0 and pos + used + valid * 2 <= len(buf):
            values = list(struct.unpack_from(f"<{valid}h", buf, pos + used))
        used += valid * 2

        spans.append((total, values))
        covered += total

    return spans, used


def _sample_channel(spans: list, frame: int) -> float:
    k = frame
    last = 0.0

    for total, values in spans:
        if values:
            last = values[-1]
        if k < total:
            index = min(k, len(values) - 1)
            return values[index] if values else 0.0
        k -= total

    return last


### Anim blocks (.ani files) ###
#
# Written when $animblocksize is set in the QC. Layout:
#   - ident 'IDAG' instead of 'IDST', version field follows
#   - the studiohdr length field holds the .ani file size
#   - everything after that is zeroed out through a 408 byte header
#   - raw animation block data follows immediately
# The animblock offset table itself lives on the MDL side (studiohdr2_t).

ANI_HEADER_SIZE = 408


def _load_ani_data(mdl_path: str) -> Optional[bytes]:
    root, _ = os.path.splitext(mdl_path)
    ani_path = root + ".ani"
    if not os.path.isfile(ani_path):
        return None

    with open(ani_path, "rb") as handle:
        return handle.read()


def _find_mdl_animblock_table(mdl_data: bytes, ani_size: int):
    """Locate (count, table_offset) of the animblock table inside the MDL.

    The table sits somewhere in studiohdr2_t; its exact sub-offset varies
    between branches, so candidate slots are validated against the size of
    the loaded .ani payload.
    """
    n = len(mdl_data)
    if n < STUDIOHDR2_OFFSET_FIELD + 4:
        return None

    candidates = []

    hdr2_offset = struct.unpack_from("<i", mdl_data, STUDIOHDR2_OFFSET_FIELD)[0]
    if 0 < hdr2_offset < n:
        for guess in (44, 48):
            candidates.append(hdr2_offset + guess)
        for off in range(hdr2_offset, min(n - 8, hdr2_offset + 512), 4):
            candidates.append(off)

    seen = set()
    for cand in candidates:
        if cand in seen or cand + 8 > n:
            continue
        seen.add(cand)

        count, table_offset = struct.unpack_from("<2i", mdl_data, cand)

        if not (1 <= count <= MAX_SAFE_BONES):
            continue
        if table_offset <= 0 or table_offset + count * 4 > n:
            continue

        try:
            offsets = struct.unpack_from(f"<{count}i", mdl_data, table_offset)
        except struct.error:
            continue

        if any(o < 0 or o > ani_size + ANI_HEADER_SIZE for o in offsets):
            continue
        if any(offsets[i] > offsets[i + 1] for i in range(count - 1)):
            continue

        return (count, table_offset)

    return None


def _plausible_bone_list(buf: bytes, pos: int) -> bool:
    """Cheap sanity walk over a bone header array."""
    limit = len(buf)
    entries = 0
    p = pos
    last_bone = -1

    while p + 4 <= limit and entries < 64:
        bone, flags, offset = struct.unpack_from("<BBh", buf, p)
        if bone >= 254:
            return entries > 0
        if bone < last_bone:
            return False
        if flags == 0:
            return False
        last_bone = bone

        data_pos = p + offset
        if data_pos < 0 or data_pos >= limit:
            return False

        p += 4
        entries += 1

    return False


class _AnimSource:
    """Buffers needed to resolve animation data addresses."""

    def __init__(self, mdl_data: bytes, mdl_path: str):
        self.mdl_data = mdl_data
        self.ani_data: Optional[bytes] = None
        self.block_count = 0
        self.block_table_offset = 0
        self.ani_checked = False
        self.mdl_path = mdl_path

    def resolve_block(self, animblock: int) -> Optional[Tuple[bytes, int]]:
        """Returns (buffer, block start offset) for a 1-based animblock id."""
        if animblock <= 0:
            return None

        if not self.ani_checked:
            self.ani_checked = True
            self.ani_data = _load_ani_data(self.mdl_path)
            if self.ani_data is not None:
                table = _find_mdl_animblock_table(self.mdl_data, len(self.ani_data))
                if table is not None:
                    self.block_count, self.block_table_offset = table

        if self.ani_data is None or self.block_count == 0:
            return None
        if animblock > self.block_count:
            return None

        (entry,) = struct.unpack_from(
            "<i",
            self.mdl_data,
            self.block_table_offset + (animblock - 1) * 4,
        )

        # Table entries may reference the .ani directly or relative to the
        # end of its 408 byte pseudo header; probe both.
        for cand in (entry, entry + ANI_HEADER_SIZE, entry - ANI_HEADER_SIZE):
            if 0 <= cand < len(self.ani_data) and _plausible_bone_list(self.ani_data, cand):
                return (self.ani_data, cand)

        if 0 <= entry < len(self.ani_data):
            return (self.ani_data, entry)
        return None


### Bone entry parsing ###

class _BoneEntry:
    __slots__ = ("bone_id", "flags", "header_pos", "data_pos", "buffer")

    def __init__(self, bone_id: int, flags: int, header_pos: int, data_pos: int, buffer: bytes):
        self.bone_id = bone_id
        self.flags = flags
        self.header_pos = header_pos
        self.data_pos = data_pos
        self.buffer = buffer


def _parse_bone_list(buf: bytes, start: int, limit_entries: int = MAX_SAFE_BONES) -> List[_BoneEntry]:
    """Walk the mstudioanim_t header chain.

    Each 4 byte header is {bone, flags, nextoffset}; the animation data for
    that bone starts immediately after the header and ``nextoffset`` points
    to the next header (zero ends the chain).
    """
    entries: List[_BoneEntry] = []
    pos = start
    end = len(buf) - 4

    while 0 <= pos <= end and len(entries) < limit_entries:
        bone_id, flags, next_offset = struct.unpack_from("<BBh", buf, pos)

        if bone_id >= 254:
            break

        entries.append(_BoneEntry(bone_id, flags, pos, pos + 4, buf))

        if next_offset <= 0:
            break
        pos += next_offset

    return entries


def _eval_bone(
    entry: _BoneEntry,
    frame: int,
    num_frames: int,
    source: _AnimSource,
    animblock: int,
    bind_pos: Tuple[float, float, float],
    bind_quat: Quat,
) -> Optional[Tuple[Tuple[float, float, float], Quat]]:
    """Decode one bone at one frame. Returns None when data is unavailable."""
    flags = entry.flags
    buf = entry.buffer
    pos = entry.header_pos + 4

    # Rotation data starts immediately after the 4 byte header.
    quat: Optional[Quat] = None
    if flags & STUDIO_ANIM_RAWROT:
        quat = _decode_quat48(buf[pos : pos + 6])
        pos += 6
    elif flags & STUDIO_ANIM_RAWROT2:
        quat = _decode_quat64(buf[pos : pos + 8])
        pos += 8
    elif flags & STUDIO_ANIM_ANIMROT:
        channels, used = _read_value_channels(buf, pos, num_frames)
        pos += used
        angles = [
            _sample_channel(channels[0], frame) * ROT_SCALE,
            _sample_channel(channels[1], frame) * ROT_SCALE,
            _sample_channel(channels[2], frame) * ROT_SCALE,
        ]
        if not (flags & STUDIO_ANIM_DELTA):
            bind_euler = quat_to_euler(bind_quat)
            angles = [angles[i] + bind_euler[i] for i in range(3)]
        quat = quat_from_euler(tuple(angles))

    if quat is None:
        quat = bind_quat

    # Position data follows the rotation data.
    position: Optional[Tuple[float, float, float]] = None
    if flags & STUDIO_ANIM_RAWPOS:
        position = _decode_vector48(buf[pos : pos + 6])
        pos += 6
    elif flags & STUDIO_ANIM_ANIMPOS:
        channels, used = _read_value_channels(buf, pos, num_frames)
        pos += used
        values = [
            _sample_channel(channels[0], frame) * POS_SCALE,
            _sample_channel(channels[1], frame) * POS_SCALE,
            _sample_channel(channels[2], frame) * POS_SCALE,
        ]
        if not (flags & STUDIO_ANIM_DELTA):
            values = [values[i] + bind_pos[i] for i in range(3)]
        position = tuple(values)

    if position is None:
        position = bind_pos

    return (position, quat)


def _read_value_channels(buf: bytes, vptr_pos: int, num_frames: int):
    """Read the three RLE channels of one mstudioanim_valueptr_t.

    The three int16 offsets are relative to the value pointer struct and
    point at independently placed channels. Returns the channels and the
    6 bytes the pointer itself occupies.
    """
    if vptr_pos + 6 > len(buf):
        return [[], [], []], 0

    offsets = struct.unpack_from("<3h", buf, vptr_pos)
    channels = []
    for offset in offsets:
        if offset <= 0:
            channels.append([])
            continue
        spans, _used = _read_channel(buf, vptr_pos + offset, num_frames)
        channels.append(spans)

    return channels, 6


### Animation description parsing ###

### Bone table access ###

def _read_mdl_bones(data: bytes):
    """Minimal bone table read (216 byte mstudiobone_t, v25+ layouts)."""
    bone_count, bone_offset = struct.unpack_from("<2i", data, 0x9C)
    bones = []
    base_transforms: Dict[int, Transform] = {}

    bone_count = max(0, min(bone_count, MAX_SAFE_BONES))
    for i in range(bone_count):
        b_offset = bone_offset + i * 216
        if b_offset + 216 > len(data):
            break

        sznameindex, parent = struct.unpack_from("<2i", data, b_offset)
        bone_name = _read_cstring(data, b_offset + sznameindex)
        pos = struct.unpack_from("<3f", data, b_offset + 32)
        rot = struct.unpack_from("<3f", data, b_offset + 60)

        bones.append(SmdBone(bone_id=i, name=bone_name or f"bone_{i}", parent_id=parent))
        base_transforms[i] = (pos, rot)

    return bones, base_transforms


def read_include_model_paths(data: bytes) -> List[str]:
    """Read $includemodel file paths from a Source MDL.

    Layouts differ between branches (main header tail on v49 era files,
    studiohdr2_t on later ones), so candidate (count, index) header slots are
    validated by resolving every entry against readable model paths.
    """
    best: List[str] = []
    best_score = -1
    limit = min(len(data) - 8, 0x400)

    for slot in range(0, limit, 4):
        count, index = struct.unpack_from("<2i", data, slot)

        if not (1 <= count <= 64):
            continue
        if index <= 0 or index + count * 8 > len(data):
            continue

        # Entry variants: 8 byte structs holding the name offset in the second
        # int (v49 era), or plain 4 byte offsets relative to each entry.
        for stride, value_at in ((8, 4), (4, 0)):
            paths: List[str] = []
            for k in range(count):
                entry = index + k * stride
                value_pos = entry + value_at
                if value_pos + 4 > len(data):
                    paths = []
                    break

                value = struct.unpack_from("<i", data, value_pos)[0]
                text = _read_cstring(data, entry + value)

                if not text or "." not in text:
                    paths = []
                    break
                if not all(31 < ord(ch) < 127 for ch in text):
                    paths = []
                    break
                paths.append(text)

            if not paths:
                continue

            # Real include tables resolve to model files; other header slots
            # can accidentally validate against unrelated name tables.
            score = sum(1 for p in paths if p.lower().endswith((".mdl", ".ani")))
            if score == count:
                return paths
            if score > best_score:
                best_score = score
                best = paths

    return best


def _resolve_include_path(base_mdl_path: str, include_path: str) -> Optional[str]:
    import_path = include_path.replace("\\", "/").lstrip("/")
    directory = os.path.dirname(os.path.abspath(base_mdl_path))

    candidates = [
        os.path.join(directory, import_path),
        os.path.join(directory, os.path.basename(import_path)),
    ]

    # Standard game layout: <root>/models/.../base.mdl includes
    # models/.../shared.mdl. Resolve against the directory that holds
    # the models folder, then against a few parent directories.
    parts = os.path.normpath(base_mdl_path).split(os.sep)
    lowered = [part.lower() for part in parts]
    if "models" in lowered:
        models_index = len(lowered) - 1 - lowered[::-1].index("models")
        root = os.sep.join(parts[:models_index])
        candidates.append(os.path.join(root, *import_path.split("/")))

    parent = directory
    for _ in range(4):
        candidates.append(os.path.join(parent, import_path))
        parent = os.path.dirname(parent)
        if not parent or parent == os.path.dirname(parent):
            break

    seen = set()
    for candidate in candidates:
        key = os.path.normcase(candidate)
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(candidate):
            return candidate
    return None


def extract_animation_clips_with_includes(
    mdl_path: str,
    _visited: Optional[set] = None,
) -> List[Tuple[str, SmdModel]]:
    """Collect animation clips from mdl_path plus its $includemodel chain."""
    if _visited is None:
        _visited = set()

    key = os.path.normcase(os.path.abspath(mdl_path))
    if key in _visited:
        return []
    _visited.add(key)

    try:
        with open(mdl_path, "rb") as handle:
            data = handle.read()
    except OSError:
        return []

    if len(data) < ANIM_INDEX_OFFSET + 4:
        return []

    magic, version = struct.unpack_from("<2i", data, 0)
    if magic != 0x54534449 or version < 25:
        return []

    bones, base_transforms = _read_mdl_bones(data)
    clips = extract_mdl_animations(data, mdl_path, bones, base_transforms)

    used_names = {name for name, _ in clips}

    for include_path in read_include_model_paths(data):
        resolved = _resolve_include_path(mdl_path, include_path)
        if resolved is None:
            continue

        for name, model in extract_animation_clips_with_includes(resolved, _visited):
            unique = name
            while unique in used_names:
                unique = f"{unique}_dup"
            used_names.add(unique)
            model.metadata["name"] = unique
            clips.append((unique, model))

    return clips


def _iter_anims(data: bytes):
    count, table_offset = struct.unpack_from("<2i", data, ANIM_COUNT_OFFSET)
    count = max(0, min(int(count), 16384))

    for i in range(count):
        desc_offset = table_offset + i * ANIMDESC_SIZE
        if desc_offset + ANIMDESC_SIZE > len(data):
            break

        (
            baseptr,
            name_offset,
            _fps_bits,
            flags,
            numframes,
            _nummovements,
            _movementindex,
        ) = struct.unpack_from("<7i", data, desc_offset)

        animblock, animindex = struct.unpack_from(
            "<2i", data, desc_offset + ANIMDESC_ANIMBLOCK_OFFSET
        )

        (fps,) = struct.unpack_from("<f", data, desc_offset + 8)

        yield {
            "offset": desc_offset,
            "baseptr": baseptr,
            "name": _read_cstring(data, desc_offset + name_offset),
            "fps": fps,
            "flags": flags,
            "numframes": numframes,
            "animblock": animblock,
            "animindex": animindex,
        }


def _iter_sequences(data: bytes):
    """Yield the local sequence table entries (v44+ layout)."""
    count, table_offset = struct.unpack_from("<2i", data, SEQ_COUNT_OFFSET)
    count = max(0, min(int(count), 16384))

    for i in range(count):
        seq_offset = table_offset + i * SEQDESC_SIZE
        if seq_offset < 0 or seq_offset + SEQDESC_SIZE > len(data):
            break

        _baseptr, label_offset, _activity_offset, flags = struct.unpack_from(
            "<4i", data, seq_offset
        )
        numblends = struct.unpack_from("<i", data, seq_offset + 56)[0]
        anim_index = struct.unpack_from("<i", data, seq_offset + 60)[0]
        groupsize_x = struct.unpack_from("<i", data, seq_offset + 68)[0]
        groupsize_y = struct.unpack_from("<i", data, seq_offset + 72)[0]

        blend_count = numblends
        if blend_count <= 0:
            blend_count = groupsize_x * groupsize_y
        blend_count = max(1, min(blend_count, 64))

        blends = []
        start = seq_offset + anim_index
        if 0 <= anim_index < len(data) and start + blend_count * 2 <= len(data):
            blends = list(struct.unpack_from(f"<{blend_count}h", data, start))

        yield {
            "index": i,
            "offset": seq_offset,
            "name": _read_cstring(data, seq_offset + label_offset),
            "flags": flags,
            "numblends": numblends,
            "blends": blends,
            "groupsize": (groupsize_x, groupsize_y),
        }


def _build_clip_from_desc(
    desc,
    data: bytes,
    source,
    bones: List[SmdBone],
    base_transforms: Dict[int, Transform],
    default_pos,
    default_quat,
) -> Optional[SmdModel]:
    num_frames = int(desc["numframes"])
    fps = float(desc["fps"])
    if num_frames <= 0 or num_frames > MAX_SAFE_FRAMES or fps <= 0.0:
        return None

    animindex = desc["animindex"]
    animblock = desc["animblock"]

    if animindex != 0:
        data_start = desc["offset"] + animindex
        buffer = data
    elif animblock != 0:
        resolved = source.resolve_block(animblock)
        if resolved is None:
            return None
        buffer, data_start = resolved
    else:
        return None

    model = _build_clip(
        buffer,
        data_start,
        source,
        animblock,
        num_frames,
        bones,
        base_transforms,
        default_pos,
        default_quat,
    )

    if model is None or not model.frames:
        return None
    return model


def extract_mdl_animations(
    data: bytes,
    mdl_path: str,
    bones: Optional[List[SmdBone]] = None,
    base_transforms: Optional[Dict[int, Transform]] = None,
) -> List[Tuple[str, SmdModel]]:
    """Extract local sequence clips from Source MDL bytes (version 25+).

    Sequences are preferred because that is what the engine and HLMV list.
    A sequence with several blends uses its first blend for the clip. Models
    without a usable sequence table fall back to the animation descriptions.
    """
    magic, version = struct.unpack_from("<2i", data, 0)
    if magic != 0x54534449 or version < 25:
        return []

    if len(data) < ANIM_INDEX_OFFSET + 4:
        return []

    bones = bones or []
    base_transforms = base_transforms or {}
    source = _AnimSource(data, mdl_path)

    default_pos = (0.0, 0.0, 0.0)
    default_quat = (0.0, 0.0, 0.0, 1.0)

    clips: List[Tuple[str, SmdModel]] = []
    used_names: set = set()

    anims = list(_iter_anims(data))
    sequences = (
        list(_iter_sequences(data))
        if len(data) >= SEQ_INDEX_OFFSET + 4
        else []
    )

    if sequences:
        for sequence in sequences:
            blends = sequence["blends"]
            if not blends:
                continue

            name = sequence["name"] or f"sequence_{sequence['index']}"
            while name in used_names:
                name = f"{name}_dup"

            blend_models = []
            first_desc = None
            for blend_index, desc_index in enumerate(blends):
                if not 0 <= desc_index < len(anims):
                    continue
                desc = anims[desc_index]

                model = _build_clip_from_desc(
                    desc,
                    data,
                    source,
                    bones,
                    base_transforms,
                    default_pos,
                    default_quat,
                )
                if model is None:
                    continue

                if first_desc is None:
                    first_desc = desc
                model.metadata.update(
                    {
                        "frame_rate": float(desc["fps"]),
                        "duration": int(desc["numframes"]) / float(desc["fps"]),
                        "blend_index": blend_index,
                    }
                )
                blend_models.append({"index": blend_index, "model": model})

            if not blend_models:
                continue

            used_names.add(name)
            primary = blend_models[0]["model"]
            primary.metadata.update(
                {
                    "frame_rate": float(first_desc["fps"]),
                    "duration": int(first_desc["numframes"]) / float(first_desc["fps"]),
                    "name": name,
                    "looping": bool(sequence["flags"] & 0x1),
                    "sequence_index": sequence["index"],
                    "numblends": sequence["numblends"],
                    "groupsize": sequence["groupsize"],
                    "sequence_blends": blend_models,
                }
            )
            clips.append((name, primary))
    else:
        for desc in anims:
            name = desc["name"] or f"anim_{len(clips)}"
            while name in used_names:
                name = f"{name}_dup"

            model = _build_clip_from_desc(
                desc,
                data,
                source,
                bones,
                base_transforms,
                default_pos,
                default_quat,
            )
            if model is None:
                continue

            used_names.add(name)
            model.metadata.update(
                {
                    "frame_rate": float(desc["fps"]),
                    "duration": int(desc["numframes"]) / float(desc["fps"]),
                    "name": name,
                    "looping": bool(desc["flags"] & 0x1),
                }
            )
            clips.append((name, model))

    return clips


def _build_clip(
    buf: bytes,
    data_start: int,
    source: _AnimSource,
    animblock: int,
    num_frames: int,
    bones: List[SmdBone],
    base_transforms: Dict[int, Transform],
    default_pos,
    default_quat: Quat,
) -> Optional[SmdModel]:
    if data_start <= 0 or data_start >= len(buf):
        return None

    entries = _parse_bone_list(buf, data_start, limit_entries=max(16, (len(bones) + 1) * 2))
    if not entries:
        return None

    bind_cache: Dict[int, Tuple[Tuple[float, float, float], Quat]] = {}

    def binds_for(bone_id: int):
        cached = bind_cache.get(bone_id)
        if cached is None:
            transform = base_transforms.get(bone_id)
            if transform is not None:
                pos, rot_euler = transform
                cached = (pos, quat_from_euler(rot_euler))
            else:
                cached = (default_pos, default_quat)
            bind_cache[bone_id] = cached
        return cached

    frames: List[SmdFrame] = []

    for frame_index in range(num_frames):
        transforms: Dict[int, Transform] = {}

        for entry in entries:
            bind_pos, bind_quat = binds_for(entry.bone_id)
            result = _eval_bone(entry, frame_index, num_frames, source, animblock, bind_pos, bind_quat)
            if result is None:
                continue
            position, quat = result
            transforms[entry.bone_id] = (position, quat_to_euler(quat))

        frames.append(SmdFrame(time=frame_index, transforms=transforms))

    model = SmdModel(version=1, bones=list(bones), frames=frames)
    return model
