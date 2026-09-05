### GoldSrc MDL (HLMDL) ###

import math
import os
import struct

import numpy as np

from viewer.smd_parser import (
    SmdBone,
    SmdFrame,
    SmdModel,
    SmdTriangle,
    SmdVertex,
)

from viewer.skeleton import (
    identity_matrix,
    local_transform_matrix,
    multiply_matrices,
    transform_point,
    transform_vector,
)

STUDIO_NF_FULLBRIGHT = 4
STUDIO_NF_ADDITIVE = 32
STUDIO_NF_MASKED = 64

TRANSPARENT_INDEX = 255


def is_gldsrc_version(version: int) -> bool:
    return 0 < version < 25


def _read_name(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("latin-1", errors="ignore").strip()


def _find_texture_mdl(mdl_path: str):
    root, ext = os.path.splitext(mdl_path)
    candidate = root + "T" + ext
    if os.path.isfile(candidate):
        return candidate

    # Companion texture models (e.g. Half-Life HD *_t.mdl packs) often
    # differ in case; fall back to a case-insensitive directory scan.
    directory = os.path.dirname(mdl_path) or "."
    wanted = os.path.basename(candidate).lower()
    try:
        for entry in os.listdir(directory):
            if entry.lower() == wanted:
                found = os.path.join(directory, entry)
                if os.path.isfile(found):
                    return found
    except OSError:
        pass

    return None


def is_goldsrc_anim_lib(path: str) -> bool:
    # External GoldSrc sequence libraries (e.g. HD pack *01.mdl files
    # referenced by the base model, such as models\agrunt01.mdl) use the
    # IDSQ magic instead of IDST. They hold animation data only.
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"IDSQ"
    except OSError:
        return False


def _is_alpha_mdl(data: bytes, version: int) -> bool:
    # Half-Life 1 alpha MDLs (version 6) use a shorter studiohdr: the
    # bone table always starts at 188 and the file length is stored at
    # offset 72. Retail GoldSrc (version 10) stores numbones at 140.
    if not 0 < version < 10:
        return False
    if len(data) < 132:
        return False
    try:
        (length, numbones, boneindex) = struct.unpack_from("<3i", data, 72)
    except struct.error:
        return False
    if not 0 < numbones <= 512:
        return False
    if boneindex != 188:
        return False
    if length <= 0 or abs(len(data) - length) > 1024:
        return False
    return True


def _alpha_seq_descs(data: bytes, numseq: int, seqindex: int) -> list:
    # Sequence entries are 104 bytes. Reverse engineered from the HL1
    # alpha samples (barney, polyrobo, prdroid):
    #   +32 fps (float), +36 flags,
    #   +40 numevents, +44 eventindex,
    #   +48 numframes, +52 ?, +56 numpivots, +60 pivotindex,
    #   +64 motiontype, +68 ?, +72 ?,
    #   +76/+80/+84 linearmovement[3] (float, +76 is distance for walks),
    #   +88 numblends, +92 animindex (true bone-track base),
    #   +96 ?, +100 ?
    # The old code read animindex from +44 (eventindex). That works when
    # a sequence has no events/pivots (A==B==C) but misaligns by 4 bytes
    # (barney shootgun: 1 event) or 96 bytes (polyrobo walk: 4 events +
    # 4 pivots; dance: 24 events) when they are present, decoding event
    # bytes as bone headers and freezing the clip at the bind pose.
    # Always use +92, falling back to +60/+44 only if +92 is insane.
    descs = []
    if numseq < 1 or seqindex <= 0:
        return descs
    for s in range(min(numseq, 1024)):
        entry = seqindex + s * 104
        if entry + 104 > len(data):
            break
        raw_name = data[entry:entry + 32].split(b"\x00")[0]
        try:
            fps = struct.unpack_from("<f", data, entry + 32)[0]
            (flags, numevents, eventindex, numframes, _u52,
             numpivots, pivotindex, motiontype, _u68, _u72) = struct.unpack_from(
                "<10i", data, entry + 36
            )
            linmove = struct.unpack_from("<3f", data, entry + 76)
            numblends, animindex_c, _u96, _u100 = struct.unpack_from(
                "<4i", data, entry + 88
            )
            animindex_a = eventindex
            animindex_b = pivotindex
        except struct.error:
            break
        name = _read_name(raw_name) or f"seq_{s}"
        if not 0 < numframes <= 4096:
            continue
        animindex = animindex_c
        if not 0 < animindex < len(data):
            animindex = animindex_b
        if not 0 < animindex < len(data):
            animindex = animindex_a
        if not 0 < animindex < len(data):
            continue
        descs.append({
            "name": name,
            "fps": fps if fps > 0.0 else 15.0,
            "flags": flags,
            "numframes": numframes,
            "animindex": animindex,
            "eventindex": eventindex,
            "numevents": numevents,
            "pivotindex": pivotindex,
            "numpivots": numpivots,
            "motiontype": motiontype,
            "linearmovement": linmove,
            "numblends": numblends,
        })
    return descs


def _alpha_track_quad_ok(data: bytes, quad_at: int, numframes: int,
                           bind0: tuple | None = None) -> bool:
    # Structural check only: counts in range, offsets inside the file,
    # first record frame sane. Deliberately NOT bind-anchored: walk and
    # shootgun frame0 legitimately differ from the seq0 bind pose (root
    # motion / recoil), so requiring an exact bind match rejects the true
    # base (polyrobo walk C, barney shootgun C) and freezes the clip.
    try:
        pos_count, pos_offset, rot_count, rot_offset = struct.unpack_from(
            "<4i", data, quad_at
        )
    except struct.error:
        return False
    if pos_count < 0 or rot_count < 0:
        return False
    if pos_count == 0 and rot_count == 0:
        return False
    if pos_count > 4096 or rot_count > 4096:
        return False
    if pos_count > 0:
        if not 0 < pos_offset < len(data):
            return False
        if pos_offset + 16 > len(data):
            return False
        try:
            frame, x, y, z = struct.unpack_from("<i3f", data, pos_offset)
        except struct.error:
            return False
        if not 0 <= frame < numframes + 16:
            return False
        if not all(math.isfinite(v) for v in (x, y, z)):
            return False
        if any(abs(v) > 100000.0 for v in (x, y, z)):
            return False
    if rot_count > 0:
        if not 0 < rot_offset < len(data):
            return False
        if rot_offset + 8 > len(data):
            return False
        try:
            frame, rx, ry, rz = struct.unpack_from("<4h", data, rot_offset)
        except struct.error:
            return False
        if not 0 <= frame < numframes + 16:
            return False
        if any(abs(v) > 72000 for v in (rx, ry, rz)):
            return False
    return True


def _alpha_track_block(data: bytes, animindex: int, numbones: int,
                       numframes: int, bind0: tuple | None = None):
    # Returns per-bone ((frame, pos) list, (frame, rot-radians) list).
    # animindex is the +92 track base (event/pivot blocks live at +44 /
    # +60 and must not be decoded as bone headers). Frame numbers are
    # kept: sparse tracks (e.g. polyrobo walk rot: 19 keys over 85
    # frames) hold by frame, not by record index.
    identity_pos = (0.0, 0.0, 0.0)
    identity_rot = (0.0, 0.0, 0.0)
    tracks = [([], []) for _ in range(numbones)]
    base = animindex
    if base <= 0 or base >= len(data):
        return tracks, identity_pos, identity_rot
    # Safety net: if the base looks structurally wrong (truncated file),
    # try nearby 16-byte alignments instead of freezing at the origin.
    if not _alpha_track_quad_ok(data, base, numframes):
        for shift in (16, 32, 48, 64, 80, 96, 4, -4):
            cand = animindex + shift
            if 0 < cand < len(data) and _alpha_track_quad_ok(
                    data, cand, numframes):
                base = cand
                break
    if base <= 0 or base >= len(data):
        return tracks, identity_pos, identity_rot
    for i in range(numbones):
        try:
            pos_count, pos_offset, rot_count, rot_offset = struct.unpack_from(
                "<4i", data, base + i * 16
            )
        except struct.error:
            break
        pos_recs = []
        if 0 < pos_count <= 4096 and 0 < pos_offset < len(data):
            available = (len(data) - pos_offset) // 16
            for r in range(min(pos_count, available)):
                try:
                    frame, x, y, z = struct.unpack_from(
                        "<i3f", data, pos_offset + r * 16
                    )
                except struct.error:
                    break
                if all(math.isfinite(v) for v in (x, y, z)):
                    if abs(x) > 100000.0 or abs(y) > 100000.0 or abs(z) > 100000.0:
                        continue
                    pos_recs.append((int(frame), (x, y, z)))
        rot_recs = []
        if 0 < rot_count <= 4096 and 0 < rot_offset < len(data):
            available = (len(data) - rot_offset) // 8
            for r in range(min(rot_count, available)):
                try:
                    frame, rx, ry, rz = struct.unpack_from(
                        "<4h", data, rot_offset + r * 8
                    )
                except struct.error:
                    break
                degs = (rx / 100.0, ry / 100.0, rz / 100.0)
                if all(abs(v) <= 720.0 for v in degs):
                    rot_recs.append(
                        (int(frame),
                         tuple(v * math.pi / 180.0 for v in degs))
                    )
        tracks[i] = (pos_recs, rot_recs)
    return tracks, identity_pos, identity_rot


def _sample_alpha_track(recs, frame, fallback):
    # recs: list of (rec_frame, value) sorted by rec_frame. Hold the
    # last key <= frame; before the first key hold the first value;
    # past the last key hold the last value. Sparse rot tracks (19 keys
    # over 85 frames) therefore spread correctly instead of playing in
    # the first N frames and freezing.
    if not recs:
        return fallback
    best = recs[0][1]
    for rec_frame, value in recs:
        if rec_frame <= frame:
            best = value
        else:
            break
    return best


def _alpha_animation(data: bytes, numbones: int, numseq: int, seqindex: int):
    # Returns (bind_pose, clips) where clips are (name, SmdModel)
    # pairs following the renderer.set_animation_clips convention.
    # Missing tracks hold the seq0 bind pose (never the origin), so
    # one bad track never breaks the whole model.
    identity = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    bind_pose: dict = {}
    clips = []
    descs = _alpha_seq_descs(data, numseq, seqindex)
    if descs:
        first_tracks, _, _ = _alpha_track_block(
            data, descs[0]["animindex"], numbones, descs[0]["numframes"],
            None,
        )
        for i in range(numbones):
            pos_recs, rot_recs = first_tracks[i]
            bind_pose[i] = (
                pos_recs[0][1] if pos_recs else identity[0],
                rot_recs[0][1] if rot_recs else identity[1],
            )
    for desc in descs:
        tracks, _, _ = _alpha_track_block(
            data, desc["animindex"], numbones, desc["numframes"]
        )
        frames = []
        for f in range(desc["numframes"]):
            transforms = {}
            for i in range(numbones):
                pos_recs, rot_recs = tracks[i]
                fallback = bind_pose.get(i, identity)
                pos = _sample_alpha_track(pos_recs, f, fallback[0])
                rot = _sample_alpha_track(rot_recs, f, fallback[1])
                transforms[i] = (pos, rot)
            frames.append(SmdFrame(time=f, transforms=transforms))
        if not frames:
            continue
        clip = SmdModel(version=6)
        clip.frames = frames
        clip.metadata.update({
            "frame_rate": desc["fps"],
            "duration": desc["numframes"] / desc["fps"],
            "name": desc["name"],
            "looping": bool(desc["flags"] & 0x1),
        })
        clips.append((desc["name"], clip))
    return bind_pose, clips


def _parse_alpha_mdl(data: bytes) -> SmdModel:
    # Header map reverse engineered from the HL1 alpha samples
    # (barney, polyrobo, prdroid, all version 6):
    # 72:length 76:numbones 80:boneindex 84:numseqgroups 88:seqgroupindex
    # 92:numseq 96:seqindex 100:numtextures 104:textureindex
    # 108:texturedataindex 112:numskinref 116:numskinfamilies
    # 120:skinindex 124:numbodyparts 128:bodypartindex
    # Bone entries are 60 bytes (name[32] + parent + padding). Only the
    # hierarchy is stored there; bind positions/rotations come from the
    # first sequence's track block: 2 entries per bone (pos, rot) of
    # (count, offset). Pos records are (frame i32, 3xf32); rot records
    # are 4xi16 (frame, euler x/y/z in centidegrees). Sequence entries
    # are 104 bytes: name[32], fps, flags, numevents, eventindex,
    # numframes, ?, numpivots, pivotindex, motiontype, ?, ?,
    # linearmovement[3], numblends, animindex (+92, the true track base),
    # ?, ?. Event/pivot blocks at +44/+60 must not be decoded as bones.
    try:
        (
            _length, numbones, boneindex,
            _numseqgroups, _seqgroupindex,
            numseq, _seqindex,
            numtextures, textureindex,
            _texturedataindex,
            numskinref, numskinfamilies, skinindex,
            numbodyparts, bodypartindex,
        ) = struct.unpack_from("<15i", data, 72)
    except struct.error as error:
        raise ValueError(f"Alpha MDL header too small: {error}")

    if not 0 < numbones <= 512:
        raise ValueError(f"Alpha MDL bone count out of range: {numbones}")
    if not 0 <= boneindex < len(data):
        raise ValueError("Alpha MDL bone table out of range")
    if not 0 < numtextures <= 512:
        raise ValueError(f"Alpha MDL texture count out of range: {numtextures}")
    if not 0 <= textureindex < len(data):
        raise ValueError("Alpha MDL texture table out of range")
    if not 0 < numbodyparts <= 32:
        raise ValueError(f"Alpha MDL bodypart count out of range: {numbodyparts}")
    if not 0 <= bodypartindex < len(data):
        raise ValueError("Alpha MDL bodypart table out of range")
    numseq = max(0, min(numseq, 1024))

    texture_entries = _load_textures(data, numtextures, textureindex)

    bind_transforms, alpha_clips = _alpha_animation(
        data, numbones, numseq, _seqindex,
    )

    bones = []

    for i in range(numbones):
        b_offset = boneindex + i * 60
        if b_offset + 60 > len(data):
            break
        raw_name, parent = struct.unpack_from("<32si", data, b_offset)
        name = _read_name(raw_name) or f"bone_{i}"
        bones.append(SmdBone(bone_id=i, name=name, parent_id=parent))
        bind_transforms.setdefault(i, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    if not bones:
        raise ValueError("Alpha MDL has no bones")

    world_matrices = {}

    def _bone_world(bone_id):
        if bone_id in world_matrices:
            return world_matrices[bone_id]
        parent_id = bones[bone_id].parent_id
        if 0 <= parent_id < len(bones) and parent_id != bone_id:
            parent_matrix = _bone_world(parent_id)
        else:
            parent_matrix = identity_matrix()
        matrix = multiply_matrices(
            parent_matrix, local_transform_matrix(bind_transforms[bone_id])
        )
        world_matrices[bone_id] = matrix
        return matrix

    for i in range(len(bones)):
        _bone_world(i)

    skin_table = []
    if numskinref > 0 and numskinfamilies > 0 and 0 < skinindex < len(data):
        total = numskinref * numskinfamilies
        if 0 < total <= 4096 and skinindex + total * 2 <= len(data):
            skin_table = list(
                struct.unpack_from(f"<{total}h", data, skinindex)[:numskinref]
            )

    model = SmdModel(version=6, bones=bones)
    if alpha_clips:
        for _clip_name, clip in alpha_clips:
            clip.bones = bones
        model.frames = alpha_clips[0][1].frames
        model.frame_names = {0: alpha_clips[0][0]}
        model.metadata["animation_clips"] = alpha_clips
        if any(len(clip.frames) > 1 for _name, clip in alpha_clips):
            model.has_animation = True
    else:
        model.frames = [SmdFrame(time=0, transforms=dict(bind_transforms))]

    vertices = []
    triangles = []
    materials = set()
    embedded_textures = {}
    texture_flags = {}

    min_bound = [math.inf] * 3
    max_bound = [-math.inf] * 3

    for bp_i in range(numbodyparts):
        bp_offset = bodypartindex + bp_i * 76
        if bp_offset + 76 > len(data):
            break
        _raw_bp_name, nummodels, _base, modelindex = struct.unpack_from(
            "<64s3i", data, bp_offset
        )
        nummodels = max(0, min(nummodels, 128))
        if nummodels <= 0 or not 0 <= modelindex < len(data):
            continue

        for m_i in range(nummodels):
            m_offset = modelindex + m_i * 112
            if m_offset + 112 > len(data):
                break
            fields = struct.unpack_from("<12i", data, m_offset + 64)
            nummesh = max(0, min(fields[3], 1024))
            meshindex = fields[4]
            numverts = max(0, min(fields[5], 100000))
            vertinfoindex = fields[6]
            numnorms = max(0, min(fields[7], 100000))
            norminfoindex = fields[8]
            frameindex = fields[10]

            if nummesh <= 0 or numverts <= 0:
                continue
            if not 0 <= meshindex < len(data):
                continue
            if not 0 <= vertinfoindex < len(data):
                continue
            if not 0 <= norminfoindex < len(data):
                continue
            if vertinfoindex + numverts > len(data):
                continue
            if norminfoindex + numnorms > len(data):
                continue

            # Per-sequence frame headers (28 bytes each) hold the actual
            # vertex/normal pools; every frame reuses the same pools for
            # these static alpha models, so the first valid one wins.
            vertindex = None
            normindex = None
            if numseq > 0 and 0 <= frameindex < len(data):
                for f_i in range(min(numseq, 64)):
                    f_offset = frameindex + f_i * 28
                    if f_offset + 28 > len(data):
                        break
                    f_fields = struct.unpack_from("<7i", data, f_offset)
                    f_numverts, f_vertindex = f_fields[3], f_fields[4]
                    f_numnorms, f_normindex = f_fields[5], f_fields[6]
                    if (
                        f_numverts == numverts
                        and f_numnorms == numnorms
                        and 0 <= f_vertindex < len(data)
                        and 0 <= f_normindex < len(data)
                        and f_vertindex + numverts * 12 <= len(data)
                        and f_normindex + numnorms * 12 <= len(data)
                    ):
                        vertindex = f_vertindex
                        normindex = f_normindex
                        break

            if vertindex is None or normindex is None:
                continue

            vert_bones = data[vertinfoindex : vertinfoindex + numverts]
            norm_bones = data[norminfoindex : norminfoindex + numnorms]

            positions = np.frombuffer(
                data, np.float32, numverts * 3, vertindex
            ).reshape(numverts, 3)
            normals = np.frombuffer(
                data, np.float32, numnorms * 3, normindex
            ).reshape(numnorms, 3)

            model_positions = [
                transform_point(
                    world_matrices[vb] if vb < len(world_matrices) else identity_matrix(),
                    positions[vi],
                )
                for vi, vb in enumerate(vert_bones)
            ]
            model_normals = [
                transform_vector(
                    world_matrices[nb] if nb < len(world_matrices) else identity_matrix(),
                    normals[ni],
                )
                for ni, nb in enumerate(norm_bones)
            ]

            for mesh_i in range(nummesh):
                mesh_offset = meshindex + mesh_i * 20
                if mesh_offset + 20 > len(data):
                    break
                numtris, triindex, skinref = struct.unpack_from(
                    "<3i", data, mesh_offset
                )
                numtris = max(0, min(numtris, 200000))
                if numtris <= 0 or not 0 <= triindex < len(data):
                    continue
                if triindex + numtris * 24 > len(data):
                    continue

                texture_id = skinref
                if skin_table and 0 <= skinref < len(skin_table):
                    texture_id = skin_table[skinref]
                if not 0 <= texture_id < len(texture_entries):
                    continue

                tex_name, tex_flags, tex_w, tex_h, tex_pix = texture_entries[
                    texture_id
                ]
                materials.add(tex_name)

                if tex_name not in embedded_textures and tex_w > 0 and tex_h > 0:
                    try:
                        embedded_textures[tex_name] = _decode_texture(
                            data,
                            tex_w,
                            tex_h,
                            tex_pix,
                            bool(tex_flags & STUDIO_NF_MASKED),
                        )
                        texture_flags[tex_name] = tex_flags
                    except (ValueError, struct.error):
                        pass

                for t_i in range(numtris):
                    t_offset = triindex + t_i * 24
                    smd_indices = []
                    valid = True
                    # One tri is 3 corners x (vert H, norm H, s H, t H).
                    for c in range(3):
                        base = t_offset + c * 8
                        vert_i, norm_i, s, t = struct.unpack_from("<4H", data, base)
                        if vert_i >= numverts or norm_i >= numnorms:
                            valid = False
                            break
                        bone_id = vert_bones[vert_i]
                        position = (
                            float(model_positions[vert_i][0]),
                            float(model_positions[vert_i][1]),
                            float(model_positions[vert_i][2]),
                        )
                        normal = (
                            float(model_normals[norm_i][0]),
                            float(model_normals[norm_i][1]),
                            float(model_normals[norm_i][2]),
                        )
                        uv = (
                            s / tex_w if tex_w else 0.0,
                            t / tex_h if tex_h else 0.0,
                        )
                        smd_idx = len(vertices)
                        vertices.append(
                            SmdVertex(
                                parent_bone=bone_id,
                                position=position,
                                normal=normal,
                                uv=uv,
                                links=[(bone_id, 1.0)],
                            )
                        )
                        smd_indices.append(smd_idx)
                        for axis in range(3):
                            min_bound[axis] = min(min_bound[axis], position[axis])
                            max_bound[axis] = max(max_bound[axis], position[axis])

                    if valid and len(smd_indices) == 3:
                        triangles.append(
                            SmdTriangle(
                                material=tex_name,
                                indices=tuple(smd_indices),
                            )
                        )

    model.vertices = vertices
    model.triangles = triangles
    model.materials = materials
    model.uv_pre_flipped = False
    model.embedded_textures = embedded_textures
    model.metadata["texture_flags"] = texture_flags
    model.metadata["alpha_mdl"] = True

    if triangles:
        model.has_geometry = True
        model.min_bound = tuple(min_bound)
        model.max_bound = tuple(max_bound)

    return model


def _load_textures(texture_data: bytes, numtextures: int, textureindex: int):
    entries = []

    for i in range(numtextures):
        offset = textureindex + i * 80
        raw_name, flags, width, height, pix = struct.unpack_from(
            "<64s4i", texture_data, offset
        )
        name = _read_name(raw_name) or f"texture_{i}"
        entries.append((name, flags, width, height, pix))

    return entries


def _decode_texture(texture_data: bytes, width, height, pix, masked):
    count = width * height

    indices = np.frombuffer(texture_data, np.uint8, count, pix).reshape(height, width)
    palette = np.frombuffer(texture_data, np.uint8, 768, pix + count).reshape(256, 3)

    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, :3] = palette[indices]
    rgba[:, :, 3] = 255

    if masked:
        rgba[indices == TRANSPARENT_INDEX, 3] = 0

    return rgba


def parse_hlmdl(mdl_path: str) -> SmdModel:
    with open(mdl_path, "rb") as handle:
        data = handle.read()

    if len(data) < 244:
        raise ValueError("GoldSrc MDL header too small")

    magic, version = struct.unpack_from("<4si", data, 0)
    if magic == b"IDSQ":
        raise ValueError(
            "This is an external GoldSrc animation library (IDSQ), not a "
            "renderable model. Open the base model instead; its sequence "
            "groups reference this file."
        )
    if magic != b"IDST":
        raise ValueError("Not a GoldSrc MDL file")

    if _is_alpha_mdl(data, version):
        return _parse_alpha_mdl(data)

    (
        numbones, boneindex,
        numbonecontrollers, bonecontrollerindex,
        numhitboxes, hitboxindex,
        numseq, seqindex,
        numseqgroups, seqgroupindex,
    ) = struct.unpack_from("<10i", data, 140)

    numtextures, textureindex, texturedataindex = struct.unpack_from("<3i", data, 180)
    numskinref, numskinfamilies, skinindex = struct.unpack_from("<3i", data, 192)
    numbodyparts, bodypartindex = struct.unpack_from("<2i", data, 204)

    texture_data = data
    if numtextures == 0 or textureindex == 0:
        companion = _find_texture_mdl(mdl_path)
        if companion is None:
            raise ValueError("MDL has no embedded textures and no companion T.mdl")
        with open(companion, "rb") as handle:
            texture_data = handle.read()
        numtextures, textureindex, _ = struct.unpack_from("<3i", texture_data, 180)

    texture_entries = _load_textures(texture_data, numtextures, textureindex)

    bones = []
    bind_transforms = {}

    for i in range(numbones):
        b_offset = boneindex + i * 112
        raw_name, parent = struct.unpack_from("<32si", data, b_offset)
        pos = struct.unpack_from("<3f", data, b_offset + 64)
        rot = struct.unpack_from("<3f", data, b_offset + 76)

        name = _read_name(raw_name) or f"bone_{i}"
        bones.append(SmdBone(bone_id=i, name=name, parent_id=parent))
        bind_transforms[i] = (pos, rot)

    # GoldSrc stores vertices and normals in bone-local space; the engine
    # transforms each by its bone matrix at render time. Bake the bind pose
    # world matrices in so the mesh renders as-is.
    world_matrices = {}

    def _bone_world(bone_id):
        if bone_id in world_matrices:
            return world_matrices[bone_id]

        parent_id = bones[bone_id].parent_id
        if 0 <= parent_id < numbones and parent_id != bone_id:
            parent_matrix = _bone_world(parent_id)
        else:
            parent_matrix = identity_matrix()

        matrix = multiply_matrices(
            parent_matrix, local_transform_matrix(bind_transforms[bone_id])
        )
        world_matrices[bone_id] = matrix
        return matrix

    for i in range(numbones):
        _bone_world(i)

    skin_table = []
    if numskinref > 0 and numskinfamilies > 0 and 0 < skinindex < len(data):
        total = numskinref * numskinfamilies
        if skinindex + total * 2 <= len(data):
            skin_table = list(
                struct.unpack_from(f"<{total}h", data, skinindex)[:numskinref]
            )

    model = SmdModel(version=1, bones=bones)
    model.frames = [SmdFrame(time=0, transforms=bind_transforms)]

    vertices = []
    triangles = []
    materials = set()
    embedded_textures = {}
    texture_flags = {}

    min_bound = [math.inf] * 3
    max_bound = [-math.inf] * 3

    for bp_i in range(numbodyparts):
        bp_offset = bodypartindex + bp_i * 76
        _raw_bp_name, nummodels, _base, modelindex = struct.unpack_from(
            "<64s3i", data, bp_offset
        )

        bp_tris = 0

        for m_i in range(nummodels):
            m_offset = modelindex + m_i * 112
            (
                _raw_m_name,
                _type,
                _boundingradius,
                nummesh,
                meshindex,
                numverts,
                vertinfoindex,
                vertindex,
                numnorms,
                norminfoindex,
                normindex,
            ) = struct.unpack_from("<64sif8i", data, m_offset)

            if nummesh <= 0 or numverts <= 0:
                continue

            vert_bones = data[vertinfoindex : vertinfoindex + numverts]
            norm_bones = data[norminfoindex : norminfoindex + numnorms]

            positions = np.frombuffer(
                data, np.float32, numverts * 3, vertindex
            ).reshape(numverts, 3)
            normals = np.frombuffer(
                data, np.float32, numnorms * 3, normindex
            ).reshape(numnorms, 3)

            model_positions = [
                transform_point(world_matrices[vert_bones[vi]], positions[vi])
                for vi in range(numverts)
            ]
            model_normals = [
                transform_vector(
                    world_matrices[norm_bones[ni] if ni < len(norm_bones) else 0],
                    normals[ni],
                )
                for ni in range(numnorms)
            ]

            for mesh_i in range(nummesh):
                mesh_offset = meshindex + mesh_i * 20
                numtris, triindex, skinref = struct.unpack_from(
                    "<3i", data, mesh_offset
                )

                texture_id = skinref
                if skin_table and 0 <= skinref < len(skin_table):
                    texture_id = skin_table[skinref]

                if not (0 <= texture_id < len(texture_entries)):
                    continue

                tex_name, tex_flags, tex_w, tex_h, tex_pix = texture_entries[
                    texture_id
                ]
                materials.add(tex_name)

                if tex_name not in embedded_textures and tex_w > 0 and tex_h > 0:
                    try:
                        embedded_textures[tex_name] = _decode_texture(
                            texture_data,
                            tex_w,
                            tex_h,
                            tex_pix,
                            bool(tex_flags & STUDIO_NF_MASKED),
                        )

                        texture_flags[tex_name] = tex_flags
                    except (ValueError, struct.error) as error:
                        pass

                pos = triindex
                tris_made = 0

                # numtris is the mesh's triangle count; the stream holds
                # strip/fan sequences until that many triangles are produced.
                while tris_made < numtris:
                    (count_raw,) = struct.unpack_from("<h", data, pos)
                    pos += 2

                    count = abs(count_raw)
                    if count < 3:
                        break

                    tv = struct.unpack_from(f"<{count * 4}H", data, pos)
                    pos += count * 8

                    corners = [
                        (tv[k * 4], tv[k * 4 + 1], tv[k * 4 + 2], tv[k * 4 + 3])
                        for k in range(count)
                    ]

                    if count_raw > 0:
                        groups = [
                            (corners[k], corners[k + 1], corners[k + 2])
                            if k % 2 == 0
                            else (corners[k + 1], corners[k], corners[k + 2])
                            for k in range(count - 2)
                        ]
                    else:
                        groups = [
                            (corners[0], corners[k + 1], corners[k + 2])
                            for k in range(count - 2)
                        ]

                    for tri in groups:
                        if tris_made >= numtris:
                            break

                        smd_indices = []

                        for vert_i, norm_i, s, t in tri:
                            if vert_i >= numverts or norm_i >= numnorms:
                                continue

                            bone_id = vert_bones[vert_i]
                            position = (
                                float(model_positions[vert_i][0]),
                                float(model_positions[vert_i][1]),
                                float(model_positions[vert_i][2]),
                            )
                            normal = (
                                float(model_normals[norm_i][0]),
                                float(model_normals[norm_i][1]),
                                float(model_normals[norm_i][2]),
                            )
                            uv = (
                                s / tex_w if tex_w else 0.0,
                                t / tex_h if tex_h else 0.0,
                            )

                            smd_idx = len(vertices)
                            vertices.append(
                                SmdVertex(
                                    parent_bone=bone_id,
                                    position=position,
                                    normal=normal,
                                    uv=uv,
                                    links=[(bone_id, 1.0)],
                                )
                            )
                            smd_indices.append(smd_idx)

                            for axis in range(3):
                                min_bound[axis] = min(min_bound[axis], position[axis])
                                max_bound[axis] = max(max_bound[axis], position[axis])

                        if len(smd_indices) == 3:
                            triangles.append(
                                SmdTriangle(
                                    material=tex_name,
                                    indices=tuple(smd_indices),
                                )
                            )
                            tris_made += 1

                bp_tris += tris_made

    model.vertices = vertices
    model.triangles = triangles
    model.materials = materials
    model.uv_pre_flipped = False
    model.embedded_textures = embedded_textures
    model.metadata["texture_flags"] = texture_flags

    if triangles:
        model.has_geometry = True
        model.min_bound = tuple(min_bound)
        model.max_bound = tuple(max_bound)

    return model