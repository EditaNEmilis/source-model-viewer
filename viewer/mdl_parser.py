import math
import os
import struct
from typing import Dict, List, Optional, Tuple

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
BETA_MODEL_STRIDE_GUESS = 112
BETA_MODEL_STRIDE_FALLBACK = 280
BETA_MODEL_STRIDE = 112
BETA_MESH_STRIDE = 68
BETA_VERTEX_STRIDE = 64


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

    min_bound = [float("inf")] * 3
    max_bound = [float("-inf")] * 3

    vtx_path = _find_companion_file(
        mdl_path, [".dx90.vtx", ".dx80.vtx", ".dx7_2bone.vtx", ".vtx", ".sw.vtx"]
    )
    vtx_data = b""
    if vtx_path:
        with open(vtx_path, "rb") as f:
            vtx_data = f.read()

    # Walk bodyparts for topology
    topology = []
    bp_nummodels_list = []
    total_models = 0
    total_verts = 0
    first_model_offset = None
    for bpi in range(numbodyparts):
        bpp = bodypartindex + bpi * 16
        if bpp + 16 > len(data):
            break
        nummodels = unpack("<i", bpp + 4)[0]
        modelindex = unpack("<i", bpp + 12)[0]
        nummodels = max(0, min(nummodels, 128))
        if bpi == 0:
            first_model_offset = modelindex
        counts = []
        mo_probe = bodypartindex + modelindex
        for mi in range(nummodels):
            mo = mo_probe + mi * BETA_MODEL_STRIDE_GUESS
            if mo + 112 > len(data):
                counts.append(0)
                continue
            nm_count = unpack("<i", mo + 72)[0]
            if nm_count >= 4096:
                nm_count = 0
            counts.append(max(0, nm_count))
            nv_count = unpack("<i", mo + 80)[0]
            total_verts += max(0, min(nv_count, 100000)) if nm_count else 0
        topology.append((nummodels, counts))
        bp_nummodels_list.append(nummodels)
        total_models += nummodels

    # Locate the sequential model slots. Each slot starts with char name[64]
    # and models of later bodyparts simply follow the previous ones.
    slots = []
    if first_model_offset is not None:
        pos = bodypartindex + first_model_offset

        def slot_valid(p):
            if p + 112 > len(data):
                return False
            nm = cstr(p)
            if not nm:
                return False
            nummeshes = unpack("<i", p + 72)[0]
            numverts = unpack("<i", p + 80)[0]
            return 0 <= nummeshes <= 4096 and 0 <= numverts <= 300000

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

    # Assign slots to bodyparts in order
    chosen_pools = []
    slot_i = 0
    for nummodels in bp_nummodels_list:
        take = slots[slot_i:slot_i + nummodels]
        slot_i += nummodels
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

    if vtx_data and total_verts > 0:
        # Bodygroups without meshes have no vtx representation; locate the
        # chain using only bodyparts that actually carry geometry.
        filtered = []
        keep_idx = []
        for t_idx, (nm_t, counts_t) in enumerate(topology):
            pos_counts = [c for c in counts_t if c > 0]
            if pos_counts:
                filtered.append((nm_t, pos_counts))
                keep_idx.append(t_idx)

        if filtered:
            located = _locate_vtx_chain(vtx_data, filtered, max(1, total_verts))
            if located:
                vtx_chain = [None] * len(topology)
                for k, orig in enumerate(keep_idx):
                    vtx_chain[orig] = located[k]

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

    model = SmdModel(version=version, bones=bones, frames=[reference_frame])
    model.vertices = model_vertices
    model.triangles = model_triangles
    model.materials = set()
    model.uv_pre_flipped = False
    model.metadata["material_dirs"] = []

    if model_triangles:
        model.has_geometry = True
        model.min_bound = tuple(min_bound)
        model.max_bound = tuple(max_bound)

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