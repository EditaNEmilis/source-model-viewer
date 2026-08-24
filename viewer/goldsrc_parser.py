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
    return candidate if os.path.isfile(candidate) else None


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
    if magic not in (b"IDST", b"IDSQ"):
        raise ValueError("Not a GoldSrc MDL file")

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