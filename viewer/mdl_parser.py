import math
import os
import re
import struct
from typing import Dict, List, Optional, Tuple

try:
    import numpy as _np
except ImportError:
    _np = None

from viewer.smd_parser import SmdBone, SmdFrame, SmdModel, SmdTriangle, SmdVertex


class MdlParseError(ValueError):
    pass


def is_mdl_file(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            magic = handle.read(4)
        return magic in (b"IDST", b"IDSQ")
    except OSError:
        return False


def is_vmdl_c_file(path: str) -> bool:
    # Source 2 compiled models (.vmdl_c) live in the Valve compiled
    # resource container (file size at +0, block table at +16 with
    # MRPH/MDAT/MBUF/PHYS/CTRL/RERL/REDI/DATA entries), not IDST.
    try:
        with open(path, "rb") as handle:
            data = handle.read(64)
        if len(data) < 32:
            return False
        if data[16:20] != b"MRPH":
            return False
        return b"MDAT" in data and b"MBUF" in data
    except OSError:
        return False


def read_vmdl_c_info(path: str) -> dict:
    """Best-effort inventory of a Source 2 compiled model (.vmdl_c).

    Parses only the resource block table (exact) plus plain-text
    references inside the RERL/REDI/DATA/MDAT blocks (material paths,
    mesh markers). Vertex/index buffers (MBUF) and the KV3 payloads
    still need a real KV3 + VBIB decoder before geometry can load, so
    counts derived from binary markers are reported as estimates.
    """
    import re

    with open(path, "rb") as handle:
        data = handle.read()

    if len(data) < 32 or data[16:20] != b"MRPH":
        raise MdlParseError("Not a Source 2 compiled resource file")

    blocks = {}
    for i in range(8):
        off = 16 + i * 12
        if off + 12 > len(data):
            break
        tag = data[off:off + 4].decode("ascii", errors="replace")
        boff, bsize = struct.unpack_from("<2I", data, off + 4)
        if 0 <= boff < len(data) and 0 < bsize <= len(data):
            blocks[tag] = {"offset": boff, "size": min(bsize, len(data) - boff)}

    def _block_strings(tag: str):
        entry = blocks.get(tag)
        if not entry:
            return []
        seg = data[entry["offset"]:entry["offset"] + entry["size"]]
        return [
            m.group(0).decode("utf-8", errors="replace")
            for m in re.finditer(rb"[ -~]{5,}", seg)
        ]

    rerl_strings = _block_strings("RERL")
    materials = sorted({s for s in rerl_strings if s.endswith(".vmat")})

    redi_strings = _block_strings("REDI")
    sources = sorted({
        s for s in redi_strings
        if s.endswith((".vmdl", ".dmx", ".vmdl_prefab"))
    })

    mdat_strings = _block_strings("MDAT")
    mdat_raw = b""
    if "MDAT" in blocks:
        entry = blocks["MDAT"]
        mdat_raw = data[entry["offset"]:entry["offset"] + entry["size"]]

    model_name = ""
    for s in _block_strings("DATA"):
        if s.endswith(".vmdl"):
            model_name = s
            break

    return {
        "file_size": len(data),
        "blocks": blocks,
        "model_name": model_name,
        "materials": materials,
        "source_refs": sources,
        "mesh_count_estimate": mdat_raw.count(b"CRenderMesh"),
        "drawcall_count_estimate": len(
            re.findall(rb"drawCall", mdat_raw)
        ),
        "data_string_count": len(_block_strings("DATA")),
        # Measured MBUF layout on the sample file (offsets relative to
        # the MBUF block start; see viewer/temp/mds notes). Float-like
        # regions show u16-block maxima near 65535, index-like regions
        # stay below ~23200. Positions are NOT plain f32 triples at
        # these region starts, so a VBIB declaration parser is still
        # required before geometry can be decoded.
        "mbuf_map_measured": {
            "float_like_a": {"range": [0, 335871], "u16_block_max": "~65535"},
            "index_like_b": {"range": [335872, 471039], "u16_max": 22340},
            "float_like_c": {"range": [471040, 629759], "u16_block_max": "~65535"},
            "index_like_d": {"range": [629760, 683775], "u16_max": 5664},
            "float_like_e": {"range": [683776, 704995], "u16_block_max": "~65535"},
        },
    }


def _read_cstring(data: bytes, offset: int) -> str:
    if offset < 0 or offset >= len(data):
        return ""
    end = data.find(b"\x00", offset)
    if end < 0:
        return data[offset:].decode("latin-1", errors="ignore").strip()
    return data[offset:end].decode("latin-1", errors="ignore").strip()


def _find_companion_file(base_path: str, extensions: List[str]) -> Optional[str]:
    root, _ = os.path.splitext(base_path)
    directory = os.path.dirname(base_path)
    base_name = os.path.basename(root).lower()

    for ext in extensions:
        candidate = root + ext
        if os.path.isfile(candidate):
            return candidate

    try:
        for entry in os.listdir(directory):
            entry_lower = entry.lower()
            for ext in extensions:
                if entry_lower == (base_name + ext).lower():
                    return os.path.join(directory, entry)
    except OSError:
        pass

    return None


def _parse_vvd(vvd_path: str, expected_checksum: int) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float], List[Tuple[int, float]]]]:
    with open(vvd_path, "rb") as f:
        data = f.read()

    if len(data) < 80:
        raise MdlParseError("VVD file is too small")

    magic, version, checksum, num_lods = struct.unpack_from("<4i", data, 0)
    if magic != 0x56534449:  # 'IDSV'
        raise MdlParseError(f"Invalid VVD magic header: {magic:#x}")

    lod_vertex_counts = struct.unpack_from("<8i", data, 16)
    num_fixups, fixup_table_start, vertex_data_start, tangent_data_start = struct.unpack_from("<4i", data, 48)

    lod0_count = max(0, lod_vertex_counts[0])

    # Vertex pool ordering depends on how the compiler emitted fixups:
    # - When the lod 0 runs cover every vertex of the lod, they define the
    #   output order directly (classic layout).
    # - Newer compilers interleave runs of every lod; there the concatenation
    #   of ALL runs in file order defines the pool order.
    # - Anything else falls back to reading the pool in original order.
    fixup_runs = []
    fixed_lod0_total = 0
    fixed_all_total = 0
    if num_fixups > 0:
        for i in range(num_fixups):
            if fixup_table_start + i * 12 + 12 > len(data):
                break
            lod, source_vert_id, num_verts = struct.unpack_from("<3i", data, fixup_table_start + i * 12)
            if num_verts <= 0:
                continue
            fixup_runs.append((source_vert_id, num_verts))
            fixed_all_total += num_verts
            if lod == 0:
                fixed_lod0_total += num_verts

    resolved_vertices = []

    def _read_raw_vert(v_idx):
        offset = vertex_data_start + v_idx * 48
        if 0 <= offset and offset + 48 <= len(data):
            return data[offset:offset + 48]
        return None

    if fixup_runs and fixed_lod0_total == lod0_count:
        for source_vert_id, num_verts in fixup_runs:
            for v in range(num_verts):
                raw = _read_raw_vert(source_vert_id + v)
                if raw is not None:
                    resolved_vertices.append(raw)
    elif fixup_runs and fixed_all_total == lod0_count:
        for source_vert_id, num_verts in fixup_runs:
            for v in range(num_verts):
                raw = _read_raw_vert(source_vert_id + v)
                if raw is not None:
                    resolved_vertices.append(raw)
    else:
        for v in range(lod0_count):
            raw = _read_raw_vert(v)
            if raw is not None:
                resolved_vertices.append(raw)

    parsed_vertices = []
    for raw in resolved_vertices:
        w0, w1, w2 = struct.unpack_from("<3f", raw, 0)
        b0, b1, b2 = struct.unpack_from("<3B", raw, 12)
        numbones = struct.unpack_from("<B", raw, 15)[0]
        px, py, pz = struct.unpack_from("<3f", raw, 16)
        nx, ny, nz = struct.unpack_from("<3f", raw, 28)
        u, v = struct.unpack_from("<2f", raw, 40)

        weights = [w0, w1, w2][:numbones]
        bones = [b0, b1, b2][:numbones]
        links = [(int(bones[i]), float(weights[i])) for i in range(numbones) if weights[i] > 0.0]

        parsed_vertices.append(((px, py, pz), (nx, ny, nz), (u, v), links))

    return parsed_vertices


### VTX helpers ###


# Group vertex record layouts by VTX version:
#   v7+: 9 byte records, original mesh vertex id (u16) at +4
#   v6 (beta): 15 byte records, original mesh vertex id (u16) at +12
VTX_VERTEX_FORMATS = ((9, 4), (15, 12))


def _read_group_ids(vtx_data: bytes, gp: int, vert_offset: int, num_verts: int, vert_limit: int):
    """Read the original mesh vertex ids of a strip group, trying both
    known vertex record layouts. A layout only counts when every id falls
    inside the vertex pool. Returns a list or None."""
    n = len(vtx_data)
    for rec, idoff in VTX_VERTEX_FORMATS:
        if gp + vert_offset + num_verts * rec > n:
            continue
        ids = []
        ok = True
        for vi in range(num_verts):
            o = gp + vert_offset + vi * rec + idoff
            if o + 2 > n:
                ok = False
                break
            v = struct.unpack_from("<H", vtx_data, o)[0]
            if v >= vert_limit:
                ok = False
                break
            ids.append(v)
        if ok:
            return ids
    return None


def _valid_vtx_group(vtx_data: bytes, gp: int, vert_limit: int) -> bool:
    """Check whether gp looks like a real VTX strip group header.

    Group layout: {numVerts, vertOffset, numIndices, indexOffset,
    numStrips, stripOffset, flags}; vertex records are 9 bytes holding
    an original mesh vertex id at offset 4.
    """
    n = len(vtx_data)
    if gp < 0 or gp + 26 > n:
        return False

    num_verts, vert_offset, num_indices, index_offset, num_strips, strip_offset = struct.unpack_from(
        "<6i", vtx_data, gp
    )

    if not (0 < num_verts <= 60000):
        return False
    if not (0 < num_indices <= 200000 or num_indices == 0):
        return False
    if not (0 < num_strips <= 4096):
        return False
    if vert_offset <= 0 or strip_offset <= 0 or index_offset < 0:
        return False
    if gp + vert_offset + num_verts * 9 > n:
        return False
    if gp + index_offset + num_indices * 2 > n:
        return False
    if gp + strip_offset + num_strips * 19 > n:
        return False

    step = max(1, num_verts // 32)
    checked = 0
    ids = _read_group_ids(vtx_data, gp, vert_offset, num_verts, vert_limit)
    if ids is None:
        return False
    for vi in range(0, num_verts, step):
        if ids[vi] >= vert_limit:
            return False
        checked += 1
    if checked == 0:
        return False

    # First strip must reference a subset of this group's indices.
    st_num_indices = struct.unpack_from("<i", vtx_data, gp + strip_offset)[0]
    if st_num_indices <= 0 or st_num_indices > num_indices:
        return False

    return True


def _pick_group_stride(vtx_data: bytes, first_group: int, num_groups: int, vert_limit: int):
    """Choose the strip group stride that validates the most groups."""
    best_stride = None
    best_score = 0

    for stride in (25, 33, 29, 36, 41, 45):
        score = 0
        for gi in range(1, num_groups):
            if _valid_vtx_group(vtx_data, first_group + gi * stride, vert_limit):
                score += 1
        if score > best_score:
            best_score = score
            best_stride = stride

    if best_stride is None:
        return 25
    return best_stride


def _pick_strip_size(vtx_data: bytes, group_pos: int, num_indices: int, num_strips: int, strip_offset: int):
    """Choose the strip header size whose index counts add up."""
    for size in (19, 20, 24, 27, 17):
        total = 0
        ok = True
        for si in range(num_strips):
            sp = group_pos + strip_offset + si * size
            if sp + 4 > len(vtx_data):
                ok = False
                break
            ni = struct.unpack_from("<i", vtx_data, sp)[0]
            if ni < 0:
                ok = False
                break
            total += ni
        if ok and total == num_indices:
            return size
    return 19


def _decode_vtx_mesh_groups(vtx_data: bytes, mesh_pos: int, vert_limit: int):
    """Decode every strip group of one VTX mesh.

    Returns a list of (vertex id mapping, triangle index triples) plus the
    strides used, or None when nothing validates.
    """
    n = len(vtx_data)
    if mesh_pos + 8 > n:
        return None

    num_strip_groups, group_offset = struct.unpack_from("<2i", vtx_data, mesh_pos)
    if not (0 < num_strip_groups <= 1024):
        return None
    if not (0 < group_offset < 2000000):
        return None

    first_group = mesh_pos + group_offset
    if not _valid_vtx_group(vtx_data, first_group, vert_limit):
        return None

    group_stride = _pick_group_stride(vtx_data, first_group, num_strip_groups, vert_limit)

    groups = []
    for gi in range(num_strip_groups):
        gp = first_group + gi * group_stride
        if gp + 26 > n:
            break

        num_verts, vert_offset, num_indices, index_offset, num_strips, strip_offset = struct.unpack_from(
            "<6i", vtx_data, gp
        )
        if num_verts <= 0 or num_indices <= 0 or num_strips <= 0:
            continue
        if gp + vert_offset + num_verts * 9 > n:
            continue
        if gp + index_offset + num_indices * 2 > n:
            continue

        ids = _read_group_ids(vtx_data, gp, vert_offset, num_verts, vert_limit)
        if ids is None:
            continue
        mapping = ids

        sg_indices = struct.unpack_from(f"<{num_indices}H", vtx_data, gp + index_offset)

        strip_size = _pick_strip_size(vtx_data, gp, num_indices, num_strips, strip_offset)

        tris = []
        for si in range(num_strips):
            sp = gp + strip_offset + si * strip_size
            if sp + 19 > n:
                break
            st_num_indices, st_index_offset, _, _, _, st_flags = struct.unpack_from(
                "<4ihB", vtx_data, sp
            )

            local = []
            if st_flags & 0x02:
                for k in range(st_num_indices - 2):
                    i0 = sg_indices[st_index_offset + k]
                    i1 = sg_indices[st_index_offset + k + 1]
                    i2 = sg_indices[st_index_offset + k + 2]
                    if k % 2 != 0:
                        i0, i1 = i1, i0
                    if i0 != i1 and i1 != i2 and i0 != i2:
                        local.append((i0, i1, i2))
            else:
                for k in range(0, st_num_indices, 3):
                    if k + 2 < st_num_indices:
                        local.append((
                            sg_indices[st_index_offset + k],
                            sg_indices[st_index_offset + k + 1],
                            sg_indices[st_index_offset + k + 2],
                        ))
            tris.extend(local)

        groups.append((mapping, tris))

    if not groups:
        return None
    return groups


def _locate_vtx_chain(vtx_data: bytes, mdl_topology, vert_limit: int):
    """Find the VTX bodypart/model/lod chain without relying on one fixed
    header layout, validating candidates against MDL topology.

    Branches differ in header size and even in whether offsets are stored
    relative to their struct or absolute to the file, so both variants are
    tried at every level. Candidate chains are scored and the best fully
    consistent one wins; the lod entry must declare exactly a mesh count
    that exists on the MDL side of its bodypart.

    Returns a list of per bodypart dicts with the lod0 mesh array position
    and mesh stride, or None when no consistent chain exists.
    """
    n = len(vtx_data)
    scan_end = min(n - 16, 8192)

    def resolve(pos, value):
        """Candidate targets for an offset field: relative or absolute."""
        out = []
        if 0 < value <= 1000000:
            out.append(value)
        rel = pos + value
        if 0 < rel < n and rel != value:
            out.append(rel)
        return out

    def resolve_bodypart(bp_pos, nummodels, mesh_counts):
        if bp_pos + 8 > n:
            return None

        bp_nummodels, bp_model_offset = struct.unpack_from("<2i", vtx_data, bp_pos)
        if bp_nummodels != nummodels:
            return None

        for model_pos in resolve(bp_pos, bp_model_offset):
            if model_pos + 8 > n:
                continue

            lod_offset = struct.unpack_from("<i", vtx_data, model_pos + 4)[0]
            if lod_offset <= 0:
                continue

            for lod_pos in resolve(model_pos, lod_offset):
                if lod_pos + 12 > n:
                    continue

                num_meshes, mesh_array_offset, _switch_point = struct.unpack_from(
                    "<2if", vtx_data, lod_pos
                )
                if num_meshes not in mesh_counts:
                    continue

                for mesh_array in resolve(lod_pos, mesh_array_offset):
                    if mesh_array <= 0 or mesh_array >= n:
                        continue

                    stride_hits = {}
                    probe_count = min(num_meshes, 8)
                    for stride in (9, 12, 16, 20, 24):
                        hits = 0
                        for mi in range(probe_count):
                            mp = mesh_array + mi * stride
                            if mp + 8 > n:
                                continue
                            nsg = struct.unpack_from("<i", vtx_data, mp)[0]
                            if not (0 < nsg <= 256):
                                continue
                            goff = struct.unpack_from("<i", vtx_data, mp + 4)[0]
                            if not (0 < goff < 2000000):
                                continue
                            if _valid_vtx_group(vtx_data, mp + goff, vert_limit):
                                hits += 1
                        if hits:
                            stride_hits[stride] = hits

                    if not stride_hits:
                        continue

                    stride, hits = max(stride_hits.items(), key=lambda kv: kv[1])
                    if hits < max(1, probe_count // 2):
                        continue

                    entry = {
                        "mesh_array": mesh_array,
                        "stride": stride,
                        "count": num_meshes,
                    }
                    score = 100 + hits * 10 + (25 if num_meshes == max(mesh_counts) else 0)
                    return entry, score

        return None

    best_chain = None
    best_score = -1

    for base in range(0, scan_end, 4):
        chain = []
        score = 0
        ok = True

        for bp_index, (nummodels, mesh_counts) in enumerate(mdl_topology):
            result = resolve_bodypart(base + bp_index * 8, nummodels, mesh_counts)
            if result is None:
                ok = False
                break
            entry, part_score = result
            chain.append(entry)
            score += part_score

        if ok and score > best_score:
            best_score = score
            best_chain = chain

    return best_chain


### HL2 beta (version 37) support ###

BETA_BONE_STRIDE = 196
BETA_MODEL_STRIDE_FALLBACK = 280
BETA_MESH_STRIDE = 68
BETA_VERTEX_STRIDE = 64
BETA_ANIMDESC_SIZE = 56


def _beta_rle_track(data: bytes, pos: int, numframes: int):
    """Decode one 2002 beta RLE channel.

    Stream layout is (valid:u8, total:u8, int16[valid]) chunks with the
    last value held for the remainder of each chunk (GoldSrc style).
    Returns a list of numframes ints, or None when the stream ends early.
    """
    n = len(data)
    out = []
    guard = 0
    while len(out) < numframes and guard < 32768:
        guard += 1
        if pos + 2 > n:
            return None
        valid = data[pos]
        total = data[pos + 1]
        pos += 2
        if total == 0 or valid > total:
            return None
        if pos + 2 * valid > n:
            return None
        vals = list(struct.unpack_from(f"<{valid}h", data, pos)) if valid else [0]
        pos += 2 * valid
        if not vals:
            vals = [0]
        for k in range(total):
            if len(out) >= numframes:
                break
            out.append(vals[k] if k < len(vals) else vals[-1])
    return out if len(out) >= numframes else None


def _beta_cstr(data: bytes, n: int, off: int) -> str:
    if off < 0 or off >= n:
        return ""
    end = data.find(b"\x00", off, min(off + 64, n))
    if end < 0:
        return ""
    try:
        return data[off:end].decode("ascii").strip()
    except ValueError:
        return ""


def _beta_printable(name: str) -> bool:
    return bool(name) and all(31 < ord(c) < 127 for c in name)


def _beta_is_zero_base(bx: float, by: float, bz: float) -> bool:
    # Chain heads (pelvis, spine) store denormal garbage (~1e-42, the
    # offset ints reinterpreted) instead of a real base position.
    return abs(bx) < 1e-9 and abs(by) < 1e-9 and abs(bz) < 1e-9


def _beta_match_bone(bind_pos, numbones, bx, by, bz, taken):
    found = []
    for i in range(numbones):
        if i in taken:
            continue
        px, py, pz = bind_pos[i]
        if abs(bx - px) > 0.02 or abs(by - py) > 0.02:
            continue
        if abs(bz - pz) > 0.02 and abs(bz + pz) > 0.02:
            continue
        found.append(i)
    return found


def _beta_entry_sane(data: bytes, n: int, bind_pos, numbones, o: int) -> bool:
    # Quick structural check for a 32 byte motion entry, no bone match.
    try:
        kind = struct.unpack_from("<i", data, o)[0]
    except struct.error:
        return False
    if kind != 2 and kind != 3:
        return False
    try:
        bx, by, bz = struct.unpack_from("<3f", data, o + 4)
        o1, o2, o3, o4 = struct.unpack_from("<4i", data, o + 16)
    except struct.error:
        return False
    if not all(math.isfinite(v) for v in (bx, by, bz)):
        return False
    if abs(bx) >= 500 or abs(by) >= 500 or abs(bz) >= 500:
        return False
    if not all(0 <= v <= 60000 for v in (o1, o2, o3, o4)):
        return False
    if o1 == 0 and o2 == 0 and o3 == 0:
        return False
    if _beta_is_zero_base(bx, by, bz):
        return True
    return bool(_beta_match_bone(bind_pos, numbones, bx, by, bz, set()))


def _beta_is_singleton(data: bytes, o: int) -> bool:
    if struct.unpack_from("<i", data, o)[0] != 3:
        return False
    bx, by, bz = struct.unpack_from("<3f", data, o + 4)
    return _beta_is_zero_base(bx, by, bz)


def _beta_finger_run(data: bytes, run, bnames, bind_pos, numbones) -> bool:
    nb = 0
    hits = 0
    for o in run:
        bx, by, bz = struct.unpack_from("<3f", data, o + 4)
        if _beta_is_zero_base(bx, by, bz):
            continue
        nb += 1
        if any("inger" in bnames[bi] or "and" in bnames[bi]
               for bi in _beta_match_bone(bind_pos, numbones, bx, by, bz, set())):
            hits += 1
    return nb > 0 and hits * 2 >= nb


def _beta_zero_frac(data: bytes, run) -> float:
    z = 0
    for o in run:
        if _beta_is_zero_base(*struct.unpack_from("<3f", data, o + 4)):
            z += 1
    return z / len(run)


def _beta_body_anchored(data: bytes, run, bind_pos) -> bool:
    # A body main owns legs/pelvis/spine (bones 0-14, never
    # fingers) or is simply big; finger/gesture openers own
    # neither and start their own anim.
    if len(run) >= 21:
        return True
    for o in run:
        bx, by, bz = struct.unpack_from("<3f", data, o + 4)
        if _beta_is_zero_base(bx, by, bz):
            continue
        for bi, (px, py, pz) in enumerate(bind_pos):
            if bi > 14:
                break
            if (abs(bx - px) <= 0.02 and abs(by - py) <= 0.02
                    and (abs(bz - pz) <= 0.02 or abs(bz + pz) <= 0.02)):
                return True
    return False


def _beta_run_bones(data: bytes, run, bind_pos):
    found = set()
    for o in run:
        bx, by, bz = struct.unpack_from("<3f", data, o + 4)
        if _beta_is_zero_base(bx, by, bz):
            continue
        for bi, (px, py, pz) in enumerate(bind_pos):
            if (abs(bx - px) <= 0.02 and abs(by - py) <= 0.02
                    and (abs(bz - pz) <= 0.02 or abs(bz + pz) <= 0.02)):
                found.add(bi)
                break
    return found


def _beta_clip_names(data: bytes, n: int, count: int):
    """Name for every animation record: the file's a_ pool for its
    validated leading run, explicit single-animation sequences after
    that, anim_N for the rest."""
    seq_single = {}
    try:
        seqcount, seqtab = struct.unpack_from("<2i", data, 0xCC)
        if 1 <= seqcount <= 2048 and 0 < seqtab < n and seqtab + seqcount * 188 <= n:
            valid_recs = 0
            for k in range(min(seqcount, 8)):
                o = seqtab + k * 188
                r0 = struct.unpack_from("<i", data, o)[0]
                na = struct.unpack_from("<i", data, o + 52)[0]
                if _beta_printable(_beta_cstr(data, n, o + r0)) and 1 <= na <= 256:
                    valid_recs += 1
            if valid_recs >= min(seqcount, 2):
                for k in range(seqcount):
                    o = seqtab + k * 188
                    try:
                        r0 = struct.unpack_from("<i", data, o)[0]
                        na = struct.unpack_from("<i", data, o + 52)[0]
                        rel = struct.unpack_from("<i", data, o + 56)[0]
                    except struct.error:
                        continue
                    sname = _beta_cstr(data, n, o + r0)
                    if not _beta_printable(sname) or na != 1:
                        continue
                    if not 0 <= rel < n or o + rel + 2 > n:
                        continue
                    try:
                        (idx,) = struct.unpack_from("<h", data, o + rel)
                    except struct.error:
                        continue
                    if 0 <= idx < count and idx not in seq_single:
                        seq_single[idx] = sname
    except struct.error:
        seq_single = {}

    pool = []
    try:
        cluster, current = [], -10**9
        for m in re.finditer(rb"a_[A-Za-z0-9_]{3,40}", data):
            s, e = m.start(), m.end()
            if e < n and data[e:e + 1] == b"\x00" and s - current < 256:
                cluster.append(m.group().decode("ascii"))
            else:
                if len(cluster) > len(pool):
                    pool = cluster
                cluster = [m.group().decode("ascii")] if e < n and data[e:e + 1] == b"\x00" else []
            current = e
        if len(cluster) > len(pool):
            pool = cluster
    except Exception:
        pool = []

    pool_run = 0
    try:
        for idx in range(min(len(pool), count)):
            if idx in seq_single:
                p = pool[idx].lower()
                s = seq_single[idx].lower()
                if p != s and p != "a_" + s and p.lstrip("a_") != s.lstrip("a_"):
                    break
            pool_run = idx + 1
    except Exception:
        pool_run = 0

    names = []
    for idx in range(count):
        if idx < pool_run:
            names.append(pool[idx])
        elif idx in seq_single:
            names.append(seq_single[idx])
        else:
            names.append(f"anim_{idx}")
    return names


def _beta_bind_pose(data: bytes, n: int, bones, boneindex: int, numbones: int):
    """Bind position, euler, and rotation scale per bone, or None."""
    numbones = max(0, min(numbones, len(bones)))
    bind_pos, bind_euler, rot_scale = [], [], []
    for i in range(numbones):
        b = boneindex + i * BETA_BONE_STRIDE
        if b + BETA_BONE_STRIDE > n:
            break
        try:
            px, py, pz = struct.unpack_from("<3f", data, b + 32)
            rx, ry, rz = struct.unpack_from("<3f", data, b + 44)
            sx, sy, sz = struct.unpack_from("<3f", data, b + 68)
        except struct.error:
            break
        if any(not math.isfinite(v) for v in (px, py, pz, rx, ry, rz, sx, sy, sz)):
            break
        bind_pos.append((px, py, pz))
        bind_euler.append((rx, ry, rz))
        rot_scale.append((sx, sy, sz))
    if numbones <= 0 or len(bind_pos) < numbones:
        return None
    return bind_pos, bind_euler, rot_scale, numbones


def _beta_bone_table(data: bytes, boneindex: int, numbones: int):
    """(names, parents) from the bone records; blanks on failure."""
    try:
        names, parents = [], []
        for i in range(numbones):
            b = boneindex + i * BETA_BONE_STRIDE
            ni, parent = struct.unpack_from("<2i", data, b)
            end = data.find(b"\x00", b + ni, b + ni + 64)
            names.append(data[b + ni:end].decode("ascii", errors="replace"))
            parents.append(parent)
        return names, parents
    except (struct.error, ValueError):
        return [""] * numbones, []


def _beta_pelvis_root(names, parents):
    pelvis_idx = None
    root_idx = None
    for i, nm in enumerate(names):
        if "elvis" in nm:
            pelvis_idx = i
        if nm.split(".")[-1] == "Bip01":
            root_idx = i
    # Beta biped pelvis/root roll channels are stored a quarter
    # turn above the decompiled value (exact pi/2 on every clip
    # checked against Crowbar output).
    zfix_bones = set()
    if pelvis_idx is not None:
        zfix_bones = {i for i, p in enumerate(parents) if p == -1}
    return pelvis_idx, root_idx, zfix_bones


def _beta_block_true_len(data: bytes, n: int, grp) -> Optional[int]:
    addrs = []
    for run in grp:
        for o in run:
            try:
                kind = struct.unpack_from("<i", data, o)[0]
            except struct.error:
                continue
            if kind != 2 and kind != 3:
                continue
            try:
                rels = struct.unpack_from("<3i", data, o + 16)
            except struct.error:
                continue
            for rel in rels:
                if rel > 0 and o + rel + 2 <= n:
                    addrs.append(o + rel)
    addrs.sort()
    votes = {}
    for idx, s in enumerate(addrs):
        bound = addrs[idx + 1] if idx + 1 < len(addrs) else None
        if bound is not None and not 0 < bound - s <= 16384:
            continue
        pos = s
        tot = 0
        clean = True
        while bound is None or pos < bound:
            if pos + 2 > n:
                clean = False
                break
            valid = data[pos]
            total = data[pos + 1]
            if total == 0 or valid > total:
                clean = False
                break
            if bound is not None and pos + 2 + 2 * valid > bound:
                clean = False
                break
            tot += total
            pos += 2 + 2 * valid
            if tot > 4096:
                clean = False
                break
        if clean and bound is not None and pos == bound and tot > 0:
            votes[tot] = votes.get(tot, 0) + 1
    if not votes:
        return None
    return max(sorted(votes), key=lambda t: (votes[t], -t))


def _beta_pair_records(data: bytes, n: int, seq_blocks, count: int, table: int):
    """Map record index to block index by exact stream length, with a
    backward repair for skipped true blocks. Returns (block_for,
    block_lens)."""
    block_lens = {}
    block_for = {}
    if seq_blocks is None:
        return block_for, block_lens
    for j, (grp, _singles) in enumerate(seq_blocks):
        block_lens[j] = _beta_block_true_len(data, n, grp)
    bi = 0
    used_blocks = set()
    for i in range(1, count):
        try:
            numframes = struct.unpack_from("<i", data, table + i * 92 + 12)[0]
        except struct.error:
            continue
        if not 8 < numframes <= 2048:
            continue
        pick = None
        for j in range(min(bi - 1, len(seq_blocks) - 1), max(bi - 40, -1), -1):
            if j not in used_blocks and block_lens.get(j) == numframes:
                pick = j
                break
        if pick is None:
            for j in range(bi, min(bi + 40, len(seq_blocks))):
                if j not in used_blocks and block_lens.get(j) == numframes:
                    pick = j
                    break
        if pick is not None:
            block_for[i] = pick
            used_blocks.add(pick)
            if pick >= bi:
                bi = pick + 1
    return block_for, block_lens


def _beta_entry_fields(data: bytes, o: int):
    """(base xyz, (o1..o4)) for a kind 2/3 entry, else None."""
    try:
        kind = struct.unpack_from("<i", data, o)[0]
    except struct.error:
        return None
    if kind != 2 and kind != 3:
        return None
    try:
        base = struct.unpack_from("<3f", data, o + 4)
        offs = struct.unpack_from("<4i", data, o + 16)
    except struct.error:
        return None
    if not (all(math.isfinite(v) for v in base)
            and all(abs(v) < 500 for v in base)
            and all(0 <= v <= 60000 for v in offs)
            and any(v != 0 for v in offs)):
        return None
    return base, offs[:3]


def _beta_collect_entries(data: bytes, n: int, seq_blocks, block_for, i: int, motion: int):
    """Raw motion entries for one record: its assigned block when the
    block holds 3+ tracks, else a window scan around the motion
    pointer. Returns (raw_entries, block_addrs or None)."""
    block_addrs = None
    raw_entries = []
    if seq_blocks is not None and i in block_for:
        grp, singles = seq_blocks[block_for[i]]
        addrs = [o for run in grp for o in run]
        if len(addrs) >= 3:
            block_addrs = set(addrs)
            for ordinal, run in enumerate(grp):
                for o in sorted(run):
                    got = _beta_entry_fields(data, o)
                    if got is None:
                        continue
                    base, offs = got
                    raw_entries.append({
                        "addr": o, "base": base, "offs": offs,
                        "bone": None,
                        "single": o in singles, "run": ordinal,
                    })
    if not raw_entries and 0 < motion < n:
        hi = min(n - 32, motion + 8192)
        o = max(0, motion - 32)
        run_ord = 0
        prev_o = None
        while o < hi and len(raw_entries) < 256:
            try:
                kind = struct.unpack_from("<i", data, o)[0]
            except struct.error:
                break
            if kind == 2 or kind == 3:
                try:
                    base = struct.unpack_from("<3f", data, o + 4)
                    o1, o2, o3, o4 = struct.unpack_from("<4i", data, o + 16)
                except struct.error:
                    break
                offs = (o1, o2, o3, o4)
                if (all(math.isfinite(v) for v in base)
                        and all(abs(v) < 500 for v in base)
                        and all(0 <= v <= 60000 for v in offs)
                        and any(v != 0 for v in offs)):
                    if prev_o is not None and o - prev_o > 64:
                        run_ord += 1
                    prev_o = o
                    raw_entries.append({
                        "addr": o, "base": base, "offs": (o1, o2, o3),
                        "bone": None, "single": False, "run": run_ord,
                    })
            o += 4
    return raw_entries, block_addrs


def _beta_attribute_entries(data: bytes, raw_entries, bind_pos, numbones, bnames):
    """Assign each raw entry a bone. Pass 1 matches entries carrying
    their bone bind position (preferring previous bone + 1 on repeats);
    pass 2 fills zero-base entries positionally from the run anchor;
    pass 2b stacks leftover chain heads below their follower; the
    compact layout assigns full-body zero blocks by address slot.
    Returns (entries, taken) with entries as (bone, addr, o1, o2, o3)."""
    taken = set()
    block_min = min((ent["addr"] for ent in raw_entries), default=None)
    for ent in raw_entries:
        if block_min is None:
            ent["rawslot"] = None
            continue
        diff = ent["addr"] - block_min
        ent["rawslot"] = diff // 32 if diff % 32 == 0 else None
    by_run = {}
    for ent in raw_entries:
        by_run.setdefault(ent["run"], []).append(ent)
    block_anchor = None
    for ordinal in sorted(by_run):
        run_ents = by_run[ordinal]
        prev = None
        for ent in run_ents:
            bx, by, bz = ent["base"]
            if _beta_is_zero_base(bx, by, bz):
                continue
            cands = _beta_match_bone(bind_pos, numbones, bx, by, bz, taken)
            if not cands:
                continue
            if prev is not None and prev + 1 in cands:
                bi = prev + 1
            else:
                bi = cands[0]
            taken.add(bi)
            ent["bone"] = bi
            prev = bi
        anchors = set()
        for ent in run_ents:
            if ent["bone"] is not None and ent["rawslot"] is not None:
                anchors.add(ent["bone"] - ent["rawslot"])
        run_anchor = anchors.pop() if len(anchors) == 1 else None
        if run_anchor is None:
            run_anchor = block_anchor
        elif block_anchor is None:
            block_anchor = run_anchor
        if run_anchor is not None:
            for ent in run_ents:
                if ent["bone"] is not None or ent["rawslot"] is None:
                    continue
                if ent["single"]:
                    continue
                bx, by, bz = ent["base"]
                if not _beta_is_zero_base(bx, by, bz):
                    continue
                want = run_anchor + ent["rawslot"]
                if 0 <= want < numbones and want not in taken:
                    taken.add(want)
                    ent["bone"] = want
    for idx in range(len(raw_entries) - 1, -1, -1):
        ent = raw_entries[idx]
        if ent["bone"] is not None:
            continue
        bx, by, bz = ent["base"]
        if not _beta_is_zero_base(bx, by, bz):
            continue
        follower = None
        for nxt in raw_entries[idx + 1:]:
            if nxt["bone"] is not None:
                follower = nxt["bone"]
                break
            nbx, nby, nbz = nxt["base"]
            if not _beta_is_zero_base(nbx, nby, nbz):
                break
        if follower is None:
            continue
        want = follower - 1
        if want >= 0 and want not in taken:
            taken.add(want)
            ent["bone"] = want
    entries = [
        (ent["bone"], ent["addr"], *ent["offs"])
        for ent in raw_entries
        if ent["bone"] is not None
    ]
    if not entries:
        core = [ent for ent in raw_entries
                if not ent["single"]]
        if (len(core) >= 21 and block_min is not None
                and all(((ent["addr"] - block_min) % 32 == 0)
                        for ent in core)):
            for ent in core:
                bi = (ent["addr"] - block_min) // 32
                if bi < numbones and bi not in taken:
                    taken.add(bi)
                    ent["bone"] = bi
            entries = [
                (ent["bone"], ent["addr"], *ent["offs"])
                for ent in raw_entries
                if ent["bone"] is not None
            ]
        elif len(core) >= 21:
            by_run = {}
            for ent in raw_entries:
                by_run.setdefault(ent["run"], []).append(ent)
            start = 0
            for ordinal in sorted(by_run):
                if ordinal > 0:
                    start = prev_end + 1
                    for j in range(start, numbones):
                        if "lavicle" in bnames[j]:
                            start = j
                            break
                slot = 0
                for ent in by_run[ordinal]:
                    if ent["single"]:
                        continue
                    bi = start + slot
                    if bi < numbones and bi not in taken:
                        taken.add(bi)
                        ent["bone"] = bi
                    slot += 1
                prev_end = start + slot - 1
            entries = [
                (ent["bone"], ent["addr"], *ent["offs"])
                for ent in raw_entries
                if ent["bone"] is not None
            ]
    return entries, taken


def _beta_decode_motion(data: bytes, raw_entries, entries, numframes,
                        bind_euler, rot_scale, root_idx, zfix_bones, taken):
    """Decode RLE rotation channels to {bone: [x|None, y|None, z|None]},
    with branch-cut unwrapping. Mutates taken (root claim)."""
    animated = {}
    if root_idx is not None and root_idx not in taken:
        for ent in raw_entries:
            if ent["bone"] is not None:
                continue
            if not ent["single"]:
                continue
            curves = [None, None, None]
            ok = False
            for ch in (1, 2):
                rel = ent["offs"][ch]
                if rel == 0:
                    continue
                raw = _beta_rle_track(data, ent["addr"] + rel, numframes)
                if raw is None:
                    continue
                base = bind_euler[root_idx][ch]
                if ch == 2 and root_idx in zfix_bones:
                    base -= math.pi / 2.0
                scale = rot_scale[root_idx][ch]
                curve = [base + v * scale for v in raw]
                if any(abs(a) > 50.0 for a in curve):
                    continue
                curves[ch] = curve
                ok = True
            if ok:
                taken.add(root_idx)
                animated[root_idx] = curves
                break
    for bi, eaddr, o1, o2, o3 in entries:
        chans = []
        ok = True
        for ch, rel in enumerate((o1, o2, o3)):
            if rel == 0:
                chans.append(None)
                continue
            raw = _beta_rle_track(data, eaddr + rel, numframes)
            if raw is None:
                ok = False
                break
            base = bind_euler[bi][ch]
            if bi in zfix_bones and ch == 2:
                # Beta pelvis/root roll is stored a quarter turn above
                # the decompiled value (exact pi/2 on every clip
                # checked); without this the pelvis sits sideways.
                base -= math.pi / 2.0
            scale = rot_scale[bi][ch]
            curve = [base + v * scale for v in raw]
            if any(abs(a) > 50.0 for a in curve):
                ok = False
                break
            chans.append(curve)
        if ok and any(c is not None for c in chans):
            animated[bi] = chans
    if animated:
        two_pi = 2.0 * math.pi
        for chans in animated.values():
            for curve in chans:
                if curve is None:
                    continue
                for t in range(1, len(curve)):
                    d = curve[t] - curve[t - 1]
                    if d > math.pi:
                        shift = math.ceil((d - math.pi) / two_pi)
                        for u in range(t, len(curve)):
                            curve[u] -= shift * two_pi
                    elif d < -math.pi:
                        shift = math.ceil((-d - math.pi) / two_pi)
                        for u in range(t, len(curve)):
                            curve[u] += shift * two_pi
    return animated


def _beta_build_frames(bind_pos, bind_euler, numbones, animated, numframes, fps):
    frames = []
    if animated:
        for t in range(numframes):
            pose = {}
            for bi in range(numbones):
                euler = list(bind_euler[bi])
                if bi in animated:
                    for ch, curve in enumerate(animated[bi]):
                        if curve is not None:
                            euler[ch] = curve[t]
                pose[bi] = (bind_pos[bi], tuple(euler))
            frames.append(SmdFrame(time=t / fps, transforms=pose))
    else:
        pose = {bi: (bind_pos[bi], bind_euler[bi]) for bi in range(numbones)}
        frames.append(SmdFrame(time=0, transforms=pose))
    return frames


def _beta_enumerate_runs(data: bytes, n: int, count: int, table: int,
                        bind_pos, numbones):
    """Candidate motion entries from the file scan, grouped into runs
    (address gap of 64 or less continues a run)."""
    lo = max(0, min(
        struct.unpack_from("<i", data, table + i * 92 + 48)[0]
        for i in range(count)
        if 0 < struct.unpack_from("<i", data, table + i * 92 + 48)[0] < n
    ) - 4096)
    scan_hi = min(n - 32, lo + max(n - lo, 0))
    cands = []
    if _np is not None:
        arr = _np.frombuffer(bytearray(data[lo:scan_hi]), dtype=_np.int32)
        idx = _np.where((arr == 2) | (arr == 3))[0]
        for k in idx:
            o = lo + int(k) * 4
            if o + 32 <= n and _beta_entry_sane(data, n, bind_pos, numbones, o):
                cands.append(o)
    else:
        o = lo
        while o < scan_hi:
            if _beta_entry_sane(data, n, bind_pos, numbones, o):
                cands.append(o)
            o += 4
    runs = []
    for o in cands:
        if runs and o - runs[-1][-1] <= 64:
            runs[-1].append(o)
        else:
            runs.append([o])
    return runs


def _beta_group_blocks(data: bytes, n: int, count: int, table: int,
                       bind_pos, numbones, bnames):
    """Group runs into per-anim blocks: a single pelvis track or a
    finger/hand run just after a block extends it, anything else
    starts a new block. Blocks come out in file order, or None when
    too few survive to trust."""
    try:
        runs = _beta_enumerate_runs(data, n, count, table, bind_pos, numbones)
        blocks = []
        cur = []
        cur_bones = set()
        cur_complete = False
        singles = set()

        def close_cur():
            if cur:
                blocks.append(cur)

        def near_cont(run):
            # A contiguous third run (crouch spine tracks at the very
            # next slot) continues a complete block; a following anim
            # starts hundreds of slots later, past its own streams.
            if not cur:
                return False
            nxt = run[0] - cur[0][0]
            end = cur[-1][-1] - cur[0][0]
            return nxt % 32 == 0 and 0 < nxt // 32 - end // 32 <= 64

        for run in runs:
            if not cur:
                # A trailing singleton never opens a block; orphans are
                # dropped rather than consume an anim slot and shift
                # every later anim by one.
                if len(run) == 1 and _beta_is_singleton(data, run[0]):
                    if blocks:
                        prev = blocks[-1]
                        last = prev[-1]
                        if not (len(last) == 1 and _beta_is_singleton(data, last[0])):
                            prev_addrs = [o for r in prev for o in r]
                            if not any(o in singles for o in prev_addrs):
                                prev.append(run)
                                singles.add(run[0])
                    continue
                cur = [run]
                cur_bones = _beta_run_bones(data, run, bind_pos)
                cur_complete = False
                continue
            prev_end = cur[-1][-1]
            gap = run[0] - prev_end
            if (len(run) == 1 and _beta_is_singleton(data, run[0])
                    and gap < 4096):
                cur.append(run)
                singles.add(run[0])
                # Singletons trail their block; nothing attaches after.
                close_cur()
                cur = []
                cur_bones = set()
                cur_complete = False
            elif (gap < 4096 and len(run) <= 20
                    and _beta_body_anchored(data, cur[0], bind_pos)
                    and (not cur_complete or near_cont(run))
                    and ((len(cur[0]) >= 21
                          and (_beta_finger_run(data, run, bnames, bind_pos, numbones)
                               or _beta_zero_frac(data, run) >= 0.5
                               or (_beta_run_bones(data, run, bind_pos) & cur_bones)))
                         or _beta_zero_frac(data, run) >= 0.5)):
                # Second runs extend a body main: finger/hand, chain
                # style, or bone-overlapping after a big main (walk
                # 34+19, idle01 34+19); chain style only after a small
                # main (E3talk 12+20+18), since a finger run there is
                # the next anim, not a third run. Gesture openers are
                # not body-anchored, so their runs stay separate
                # instead of chaining into mega-blocks.
                cur.append(run)
                cur_bones |= _beta_run_bones(data, run, bind_pos)
                if sum(len(r) for r in cur) >= 40:
                    # Big mains (+ second) are usually complete; only
                    # a near run still continues them (see near_cont).
                    cur_complete = True
            else:
                close_cur()
                if len(run) == 1 and _beta_is_singleton(data, run[0]):
                    cur = []
                    cur_bones = set()
                else:
                    cur = [run]
                    cur_bones = _beta_run_bones(data, run, bind_pos)
                cur_complete = False
        close_cur()
        spans = []
        for grp in blocks:
            spans.append((grp[0][0], grp))
        spans.sort(key=lambda s: s[0])
        if len(spans) >= max(8, (count - 1) // 2):
            return [(grp, singles) for _, grp in spans]
        return None
    except (struct.error, ValueError):
        return None


def _beta_animation_clips(mdl_path, data, bones, base_transforms):
    # HL2 beta (v37, 2002 leak branch) rotation decoder.
    #
    # Each 92 byte animation record is
    #   [f0:int][fps:float][flags:int][numframes:int][0:int][f5:int]
    #   [bbox 6 floats][motion:int][zeros]
    # f5 points into a contiguous per-anim pool laid in reverse record
    # order (slots of 92, 48, or 4 bytes; families of similar anims use
    # sliding 4 byte windows into one shared stream). Content is unique
    # per anim and unidentified: smooth u16 curves for most anims, plus
    # one structured case (static aim pose) of 40 spaced mid values with
    # inline data. Ruled out as RLE offsets, frame-0 raws, stream sizes,
    # and motion ranges, so it is currently unused by the decoder.
    # motion points at per-bone 32 byte entries of
    #   [kind:int][base xyz floats][o1..o4:int entry-relative offsets]
    # where kind 3 starts a chain (pelvis root included) and kind 2
    # continues it. Each nonzero offset addresses one RLE rotation
    # channel (x, y, z in order); a zero offset means the channel stays
    # at the bone bind rotation. Decoded angle =
    #   bind_euler + raw * bone_rot_scale
    # with bind euler at bone+44 and rot scales at bone+68 (196 byte
    # bone stride). Entries carry only rotation: bones without motion
    # (and position tracks, including pelvis root motion) are left at
    # the bind pose, so beta clips play in place.
    # Names come from single-animation sequences (explicit anim indices)
    # plus the leading run of the file's a_ name pool once it validates
    # against those sequence names; anything else stays anim_N.
    # Returns (clips, found_table).
    n = len(data)
    if n < 0x400 or not bones:
        return [], False
    try:
        count, table = struct.unpack_from("<2i", data, 0xB4)
    except struct.error:
        return [], False
    if not 1 <= count <= 2048:
        return [], False
    if not 0 < table < n or table + count * 92 > n:
        return [], False

    clip_names = _beta_clip_names(data, n, count)
    try:
        numbones, boneindex = struct.unpack_from("<2i", data, 0x9C)
    except struct.error:
        return [], False
    bound = _beta_bind_pose(data, n, bones, boneindex, numbones)
    if bound is None:
        return [], False
    bind_pos, bind_euler, rot_scale, numbones = bound
    bnames, parents = _beta_bone_table(data, boneindex, numbones)
    pelvis_idx, root_idx, zfix_bones = _beta_pelvis_root(bnames, parents)

    seq_blocks = _beta_group_blocks(
        data, n, count, table, bind_pos, numbones, bnames)

    # Pair records to blocks by exact stream length. Streams in a
    # block are tightly packed (each starts where the previous ends),
    # so bounding each RLE stream at the next stream's start yields
    # its true frame count; the block length is the mode over its
    # streams. Truncated longer blocks can no longer shadow a record
    # (their true length differs), orphans stay unconsumed, and a
    # backward repair grabs a skipped true block (E3talk's trails
    # E3shout's in the file). Tiny holds stay static.
    block_for, block_lens = _beta_pair_records(data, n, seq_blocks, count, table)

    clips = []
    used_names = set()
    # Each anim decodes from its assigned block's entries only, so clips
    # can no longer claim a neighbor's motion through the f48 window.
    # Records without a block fall back to the window scan (old
    # behavior), and stay static when that finds nothing decodable.
    for i in range(count):
        off = table + i * 92
        try:
            (f0,) = struct.unpack_from("<i", data, off)
            (fps,) = struct.unpack_from("<f", data, off + 4)
            flags = struct.unpack_from("<i", data, off + 8)[0]
            numframes = struct.unpack_from("<i", data, off + 12)[0]
            motion = struct.unpack_from("<i", data, off + 48)[0]
        except struct.error:
            continue
        name = clip_names[i]
        if name in used_names:
            suffix = 2
            while f"{name}_{suffix}" in used_names:
                suffix += 1
            name = f"{name}_{suffix}"
        used_names.add(name)
        if not 0 < numframes <= 2048 or not 0.0 < float(fps) <= 240.0:
            continue
        fps = float(fps)
        raw_entries, block_addrs = _beta_collect_entries(
            data, n, seq_blocks, block_for, i, motion)
        entries, taken = _beta_attribute_entries(
            data, raw_entries, bind_pos, numbones, bnames)
        animated = _beta_decode_motion(
            data, raw_entries, entries, numframes,
            bind_euler, rot_scale, root_idx, zfix_bones, taken)
        frames = _beta_build_frames(
            bind_pos, bind_euler, numbones, animated, numframes, fps)
        model = SmdModel(version=37, bones=bones, frames=frames)
        model.metadata.update({
            "frame_rate": fps,
            "duration": (numframes / fps) if animated else 0.0,
            "name": name,
            "looping": bool(flags & 0x1),
            "beta_block": (min(block_addrs)
                           if block_addrs is not None else None),
            "beta_source": ("block" if block_addrs is not None
                            else ("window" if raw_entries else "static")),
            "beta_len": block_lens.get(block_for.get(i))
                        if seq_blocks is not None else None,
        })
        clips.append((name, model))
    return clips, True


def _parse_mdl_beta(mdl_path):
    """Parse HL2 beta era models (MDL version 37, Axel/leak branch).

    Struct layouts follow the branch studio.h: vertices are 64 byte records
    stored inside the MDL (boneweights, pos, normal, uv), meshes are 68
    bytes, bones 196. Triangles come from the companion VTX strips.
    """
    with open(mdl_path, "rb") as f:
        data = f.read()

    def unpack(fmt, off):
        return struct.unpack_from(fmt, data, off)

    def cstr(off):
        if off < 0 or off >= len(data):
            return ""
        end = data.find(b"\x00", off)
        return data[off:end].decode("latin-1", errors="ignore").strip()

    version = unpack("<i", 4)[0]

    # Textures (header 0xE0/0xE4, 32 byte entries, name offset relative to entry)
    numtextures, textureindex = unpack("<2i", 0xE0)
    numtextures = max(0, min(numtextures, 512))
    textures = []
    for i in range(numtextures):
        entry = textureindex + i * 32
        if entry + 4 > len(data):
            break
        name_off = unpack("<i", entry)[0]
        textures.append(cstr(entry + name_off))

    # Skin tables (header 0xF0/0xF4/0xF8)
    numskinref, numskinfamilies, skinindex = unpack("<3i", 0xF0)
    skin_table = []
    if numskinref > 0 and numskinfamilies > 0:
        total_skins = min(numskinref * numskinfamilies, 65536)
        try:
            raw_skins = unpack(f"<{total_skins}h", skinindex)
            skin_table = list(raw_skins[:numskinref])
        except struct.error:
            pass

    # Bones (196 byte stride, pos/value at +32, rot at +44)
    numbones, boneindex = unpack("<2i", 0x9C)
    numbones = max(0, min(numbones, 512))
    bones = []
    base_transforms = {}
    for i in range(numbones):
        b = boneindex + i * BETA_BONE_STRIDE
        if b + BETA_BONE_STRIDE > len(data):
            break
        name_off = unpack("<i", b)[0]
        parent = unpack("<i", b + 4)[0]
        px, py, pz = unpack("<3f", b + 32)
        rx, ry, rz = unpack("<3f", b + 44)
        bones.append(SmdBone(bone_id=i, name=cstr(b + name_off) or f"bone_{i}", parent_id=parent))
        base_transforms[i] = ((px, py, pz), (rx, ry, rz))

    reference_frame = SmdFrame(time=0, transforms=dict(base_transforms))

    # Bodyparts (header 0xFC/0x100); entries 16 bytes:
    # {sznameindex, nummodels, base, modelindex}
    numbodyparts, bodypartindex = unpack("<2i", 0xFC)
    numbodyparts = max(0, min(numbodyparts, 256))

    model_vertices = []
    model_triangles = []
    model_materials = set()

    min_bound = [float("inf")] * 3
    max_bound = [float("-inf")] * 3

    vtx_path = _find_companion_file(
        mdl_path, [".dx90.vtx", ".dx80.vtx", ".dx7_2bone.vtx", ".vtx", ".sw.vtx"]
    )
    vtx_data = b""
    if vtx_path:
        with open(vtx_path, "rb") as f:
            vtx_data = f.read()

    # Read bodypart entries first. Each modelindex is relative to its own
    # bodypart entry, not to the bodypart table base.
    bp_entries = []
    for bpi in range(numbodyparts):
        bpp = bodypartindex + bpi * 16
        if bpp + 16 > len(data):
            break
        nummodels = unpack("<i", bpp + 4)[0]
        modelindex = unpack("<i", bpp + 12)[0]
        nummodels = max(0, min(nummodels, 128))
        bp_entries.append((bpp, nummodels, modelindex))
    bp_nummodels_list = [entry[1] for entry in bp_entries]
    total_models = sum(bp_nummodels_list)

    def slot_valid(p):
        if p + 112 > len(data):
            return False
        nm = cstr(p)
        if not nm:
            return False
        nummeshes = unpack("<i", p + 72)[0]
        numverts = unpack("<i", p + 80)[0]
        return 0 <= nummeshes <= 4096 and 0 <= numverts <= 300000

    # Locate the sequential model slots. Each slot starts with char name[64]
    # and models of later bodyparts simply follow the previous ones.
    slots = []
    first_model_offset = bp_entries[0][2] if bp_entries else None
    if first_model_offset is not None:
        pos = bodypartindex + first_model_offset

        # stride detection: distance to the next valid slot
        stride = BETA_MODEL_STRIDE_FALLBACK
        if total_models > 1:
            for cand in range(pos + 64, min(pos + 4096, len(data) - 112), 4):
                if cstr(cand).lower().endswith(".smd") and slot_valid(cand):
                    stride = cand - pos
                    break

        while len(slots) < total_models:
            if not slot_valid(pos):
                break
            slots.append(pos)
            pos += stride

    # Assign slots to bodyparts in order, deriving the VTX topology from
    # the real slot addresses instead of guessing the model stride.
    topology = []
    total_verts = 0
    chosen_pools = []
    slot_i = 0
    for _bpp, nummodels, _modelindex in bp_entries:
        take = slots[slot_i:slot_i + nummodels]
        slot_i += nummodels
        counts = []
        for mo in take:
            try:
                nm_count = unpack("<i", mo + 72)[0]
                nv_count = unpack("<i", mo + 80)[0]
            except struct.error:
                nm_count, nv_count = 0, 0
            if nm_count >= 4096:
                nm_count = 0
            counts.append(max(0, nm_count))
            if nm_count:
                total_verts += max(0, min(nv_count, 100000))
        topology.append((nummodels, counts))
        chosen = None
        for mo in take:
            if unpack("<i", mo + 72)[0] > 0:
                chosen = mo
                break
        if chosen is None and take:
            chosen = take[0]
        if chosen is None:
            chosen_pools.append(None)
            continue

        nummeshes = unpack("<i", chosen + 72)[0]
        meshindex = unpack("<i", chosen + 76)[0]
        numverts = unpack("<i", chosen + 80)[0]
        vertexindex = unpack("<i", chosen + 84)[0]
        chosen_pools.append(
            {
                "addr": chosen,
                "nummeshes": max(0, min(nummeshes, 4096)),
                "mesh_table": chosen + meshindex,
                "pool": chosen + vertexindex,
                "numverts": max(0, min(numverts, 100000)),
            }
        )

    vtx_chain = None
    if vtx_data and total_verts > 0:
        # Bodygroups without meshes have no vtx representation; locate the
        # chain using only bodyparts that actually carry geometry. When the
        # full set does not match (a variant missing from the strips), retry
        # while dropping the smallest bodyparts so one gap cannot sink the
        # rest of the model.
        base = []
        base_idx = []
        for t_idx, (nm_t, counts_t) in enumerate(topology):
            pos_counts = [c for c in counts_t if c > 0]
            if pos_counts:
                base.append((nm_t, pos_counts))
                base_idx.append(t_idx)

        def _mesh_total(entry):
            return sum(entry[1])

        attempts = []
        if base:
            attempts.append(list(range(len(base))))
            order = sorted(
                range(len(base)), key=lambda k: (_mesh_total(base[k]), k)
            )
            for drop in order:
                if len(base) > 1:
                    attempts.append([k for k in range(len(base)) if k != drop])

        for attempt in attempts:
            sub = [base[k] for k in attempt]
            located = _locate_vtx_chain(vtx_data, sub, max(1, total_verts))
            if located:
                vtx_chain = [None] * len(topology)
                for pos, k in enumerate(attempt):
                    vtx_chain[base_idx[k]] = located[pos]
                break

    # Read vertex pools for every chosen model
    pools = {}
    for pi, cp in enumerate(chosen_pools):
        if cp is None:
            continue
        recs = []
        pool_pos = cp["pool"]
        for vi in range(cp["numverts"]):
            o = pool_pos + vi * BETA_VERTEX_STRIDE
            if o + 64 > len(data):
                break
            w = unpack("<4f", o)
            bn = unpack("<4h", o + 16)
            numb = unpack("<h", o + 24)[0]
            numb = max(0, min(numb, 4))
            px, py, pz = unpack("<3f", o + 32)
            nx, ny, nz = unpack("<3f", o + 44)
            u_, v_ = unpack("<2f", o + 56)
            links = [
                (int(bn[k]), float(w[k]))
                for k in range(numb)
                if 0 <= bn[k] < len(bones) and w[k] > 0.0
            ]
            recs.append(((px, py, pz), (nx, ny, nz), (u_, v_), links))
        pools[pi] = recs

    # Build geometry per bodypart using VTX strips
    for bpi in range(numbodyparts):
        if bpi >= len(chosen_pools):
            continue
        cp = chosen_pools[bpi]
        if cp is None:
            continue

        nummeshes = cp["nummeshes"]
        mesh_table = cp["mesh_table"]
        recs = pools.get(bpi, [])
        if not recs:
            continue

        chain_bp = None
        if vtx_chain and bpi < len(vtx_chain):
            chain_bp = vtx_chain[bpi]

        for mesh_i in range(nummeshes):
            mh = mesh_table + mesh_i * BETA_MESH_STRIDE
            if mh + 16 > len(data):
                break
            mat_id = unpack("<i", mh)[0]
            num_mesh_verts = unpack("<i", mh + 8)[0]
            vertexoffset = unpack("<i", mh + 12)[0]

            actual_mat_id = mat_id
            if skin_table and 0 <= mat_id < len(skin_table):
                actual_mat_id = skin_table[mat_id]

            mat_name = "default"
            if 0 <= actual_mat_id < len(textures):
                mat_name = textures[actual_mat_id]

            decoded = None
            if chain_bp is not None and mesh_i < chain_bp["count"]:
                vtx_mesh_pos = chain_bp["mesh_array"] + mesh_i * chain_bp["stride"]
                decoded = _decode_vtx_mesh_groups(vtx_data, vtx_mesh_pos, max(1, num_mesh_verts))

            if not decoded:
                continue

            for mapping, tris in decoded:
                for i0, i1, i2 in tris:
                    local = []
                    ok_tri = True
                    for idx in (i0, i1, i2):
                        if idx >= len(mapping):
                            ok_tri = False
                            continue
                        pool_idx = vertexoffset + mapping[idx]
                        if not (0 <= pool_idx < len(recs)):
                            ok_tri = False
                            continue
                        local.append(pool_idx)
                    if not ok_tri or len(local) != 3:
                        continue

                    smd_indices = []
                    for pool_idx in local:
                        pos, norm, uv, links = recs[pool_idx]
                        parent_bone = links[0][0] if links else 0
                        smd_idx = len(model_vertices)
                        model_vertices.append(
                            SmdVertex(parent_bone=parent_bone, position=pos, normal=norm, uv=uv, links=links)
                        )
                        smd_indices.append(smd_idx)
                        for axis in range(3):
                            min_bound[axis] = min(min_bound[axis], pos[axis])
                            max_bound[axis] = max(max_bound[axis], pos[axis])

                    model_triangles.append(
                        SmdTriangle(material=mat_name, indices=(smd_indices[0], smd_indices[1], smd_indices[2]))
                    )
                    model_materials.add(mat_name)

    model = SmdModel(version=version, bones=bones, frames=[reference_frame])
    model.vertices = model_vertices
    model.triangles = model_triangles
    model.materials = model_materials
    model.uv_pre_flipped = False
    model.metadata["material_dirs"] = []

    if model_triangles:
        model.has_geometry = True
        model.min_bound = tuple(min_bound)
        model.max_bound = tuple(max_bound)

    try:
        clips, found_table = _beta_animation_clips(
            mdl_path, data, bones, base_transforms
        )
        if clips:
            model.metadata["animation_clips"] = clips
            model.has_animation = True
        elif found_table:
            model.metadata["animation_error"] = (
                "Beta sequence table found but no clips decoded yet"
            )
    except Exception as error:
        model.metadata["animation_error"] = str(error)

    return model


def parse_mdl(mdl_path: str) -> SmdModel:
    with open(mdl_path, "rb") as f:
        data = f.read()

    if len(data) >= 32 and data[16:20] == b"MRPH":
        if mdl_path.lower().endswith(".vmdl_c"):
            from viewer.vmdl_parser import parse_vmdl_c
            return parse_vmdl_c(mdl_path)
        raise MdlParseError(
            "Not a Source 1 MDL file: this is a Source 2 compiled resource "
            "(.vmdl_c-style container with MRPH/MDAT/MBUF blocks). "
            "Only .vmdl_c model files are supported; use a Source 1 "
            "MDL/SMD/DMX file instead."
        )

    if len(data) >= 4 and data[:4] == b"IDSQ":
        raise MdlParseError(
            "This is an external GoldSrc animation library (IDSQ), not a "
            "renderable model. Open the base model instead; its sequence "
            "groups reference this file."
        )

    if len(data) < 408:
        raise MdlParseError("MDL header too small")

    magic, version, checksum = struct.unpack_from("<3i", data, 0)
    if magic not in (0x54534449, 0x51534449):  # 'IDST' or 'IDSQ'
        raise MdlParseError("Not a valid Source MDL file")

    if version < 25:
        from viewer.goldsrc_parser import parse_hlmdl
        return parse_hlmdl(mdl_path)

    if version < 44:
        # HL2 beta era (2001-2003): different header field order, vertices
        # stored inside the MDL, triangles from the companion VTX strips.
        return _parse_mdl_beta(mdl_path)

    vvd_path = _find_companion_file(mdl_path, [".vvd"])
    vtx_path = _find_companion_file(mdl_path, [".dx90.vtx", ".dx80.vtx", ".vtx", ".sw.vtx"])

    # Geometry companions may be absent on animation-only models;
    # bones and animation sequences still load in that case.
    have_geometry_sources = bool(vvd_path and vtx_path)
    if not have_geometry_sources:
        vvd_vertices = []
        vtx_data = b""
    else:
        vvd_vertices = _parse_vvd(vvd_path, checksum)

        with open(vtx_path, "rb") as f:
            vtx_data = f.read()

        if len(vtx_data) < 36:
            raise MdlParseError("VTX header too small")

    # Bones
    bone_count, bone_offset = struct.unpack_from("<2i", data, 0x9C)
    bones = []
    base_transforms = {}

    for i in range(bone_count):
        b_offset = bone_offset + i * 216
        sznameindex, parent = struct.unpack_from("<2i", data, b_offset)
        bone_name = _read_cstring(data, b_offset + sznameindex)
        pos = struct.unpack_from("<3f", data, b_offset + 32)
        rot = struct.unpack_from("<3f", data, b_offset + 60)

        bones.append(SmdBone(bone_id=i, name=bone_name or f"bone_{i}", parent_id=parent))
        base_transforms[i] = (pos, rot)

    reference_frame = SmdFrame(time=0, transforms=base_transforms)

    # Textures and search directories
    texture_count, texture_offset = struct.unpack_from("<2i", data, 0xCC)
    textures = []
    for i in range(texture_count):
        t_offset = texture_offset + i * 64
        name_offset = struct.unpack_from("<i", data, t_offset)[0]
        textures.append(_read_cstring(data, t_offset + name_offset))

    texturedir_count, texturedir_offset = struct.unpack_from("<2i", data, 0xD4)
    material_dirs = []
    for i in range(texturedir_count):
        dir_offset = struct.unpack_from("<i", data, texturedir_offset + i * 4)[0]
        material_dirs.append(_read_cstring(data, dir_offset).replace("\\", "/").strip("/"))

    # Skin replacement table (maps mesh material indices to active textures)
    skinreference_count, skinrfamily_count, skinreference_index = struct.unpack_from("<3i", data, 0xDC)
    skin_table = []
    if skinreference_count > 0 and skinrfamily_count > 0 and 0 < skinreference_index < len(data):
        total_skins = skinreference_count * skinrfamily_count
        raw_skins = struct.unpack_from(f"<{total_skins}h", data, skinreference_index)
        # Default skin family (family 0)
        skin_table = list(raw_skins[:skinreference_count])

    # Geometry construction (MDL topology first; the VTX chain is found by validation)
    bodypart_count, bodypart_offset = struct.unpack_from("<2i", data, 0xE8)

    mdl_topology = []
    for bp_i in range(bodypart_count):
        bp_offset = bodypart_offset + bp_i * 16
        _, nummodels, _, modelindex = struct.unpack_from("<4i", data, bp_offset)
        mesh_counts = []
        for mi in range(nummodels):
            test_m_offset = bp_offset + modelindex + mi * 148
            mesh_counts.append(struct.unpack_from("<i", data, test_m_offset + 72)[0])
        mdl_topology.append((nummodels, mesh_counts))

    vtx_chain = None
    if have_geometry_sources:
        vtx_chain = _locate_vtx_chain(vtx_data, mdl_topology, max(1, len(vvd_vertices)))

    model = SmdModel(version=1, bones=bones, frames=[reference_frame])
    model_vertices = []
    model_triangles = []
    model_materials = set()

    min_bound = [float("inf"), float("inf"), float("inf")]
    max_bound = [float("-inf"), float("-inf"), float("-inf")]

    dropped_total = 0

    for bp_i in range(bodypart_count):
        bp_offset = bodypart_offset + bp_i * 16
        _, nummodels, _, modelindex = struct.unpack_from("<4i", data, bp_offset)

        available_models = nummodels
        if available_models <= 0:
            continue

        variant_meshes = mdl_topology[bp_i][1]

        chosen_model_idx = 0
        for test_idx, count in enumerate(variant_meshes):
            if count > 0:
                chosen_model_idx = test_idx
                break

        m_i = chosen_model_idx
        m_offset = bp_offset + modelindex + m_i * 148
        _, _, nummeshes, meshindex, _, vertexindex = struct.unpack_from(
            "<ifi3i", data, m_offset + 64
        )

        chain_bp = None
        if vtx_chain and bp_i < len(vtx_chain):
            chain_bp = vtx_chain[bp_i]

        for mesh_i in range(nummeshes):
            mesh_offset = m_offset + meshindex + mesh_i * 116
            mat_id, _, _, vertexoffset = struct.unpack_from("<4i", data, mesh_offset)

            actual_mat_id = mat_id
            if skin_table and 0 <= mat_id < len(skin_table):
                actual_mat_id = skin_table[mat_id]

            mat_name = "default"
            if 0 <= actual_mat_id < len(textures):
                mat_name = textures[actual_mat_id]

            tris_here = 0
            dropped = 0

            decoded = None
            if chain_bp is not None and mesh_i < chain_bp["count"]:
                vtx_mesh_pos = chain_bp["mesh_array"] + mesh_i * chain_bp["stride"]
                decoded = _decode_vtx_mesh_groups(vtx_data, vtx_mesh_pos, max(1, len(vvd_vertices)))

            if decoded:
                for mapping, tris in decoded:
                    for i0, i1, i2 in tris:
                        vvd_local = []
                        ok_tri = True
                        for idx in (i0, i1, i2):
                            if idx >= len(mapping):
                                ok_tri = False
                                dropped += 1
                                continue
                            orig_id = mapping[idx]
                            vvd_idx = vertexindex // 48 + vertexoffset + orig_id
                            if 0 <= vvd_idx < len(vvd_vertices):
                                vvd_local.append(vvd_idx)
                            else:
                                ok_tri = False
                                dropped += 1
                        if not ok_tri or len(vvd_local) != 3:
                            continue

                        smd_indices = []
                        for vvd_idx in vvd_local:
                            pos, norm, uv, links = vvd_vertices[vvd_idx]
                            parent_bone = links[0][0] if links else 0
                            smd_idx = len(model_vertices)

                            model_vertices.append(
                                SmdVertex(
                                    parent_bone=parent_bone,
                                    position=pos,
                                    normal=norm,
                                    uv=uv,
                                    links=links,
                                )
                            )
                            smd_indices.append(smd_idx)

                            for axis in range(3):
                                min_bound[axis] = min(min_bound[axis], pos[axis])
                                max_bound[axis] = max(max_bound[axis], pos[axis])

                        model_triangles.append(
                            SmdTriangle(
                                material=mat_name,
                                indices=(smd_indices[0], smd_indices[1], smd_indices[2]),
                            )
                        )
                        tris_here += 1

            dropped_total += dropped

    model.vertices = model_vertices
    model.triangles = model_triangles
    model.materials = model_materials
    model.uv_pre_flipped = False
    model.metadata["material_dirs"] = material_dirs

    if model_triangles:
        model.has_geometry = True
        model.min_bound = tuple(min_bound)
        model.max_bound = tuple(max_bound)

    # Local animation sequences plus $includemodel chains
    if version >= 25:
        try:
            from viewer.mdl_anim import extract_animation_clips_with_includes

            clips = extract_animation_clips_with_includes(mdl_path)
            if clips:
                model.metadata["animation_clips"] = clips
                model.has_animation = True
        except Exception as error:
            model.metadata["animation_error"] = str(error)

    return model