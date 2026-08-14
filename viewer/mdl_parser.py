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
        return magic == b"IDST"
    except OSError:
        return False


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

    if len(data) < 64:
        raise MdlParseError("VVD file is too small")

    magic, version, checksum, num_lods = struct.unpack_from("<4i", data, 0)
    if magic != 0x56534449:  # 'IDSV'
        raise MdlParseError(f"Invalid VVD magic header: {magic:#x}")

    num_lod_verts = struct.unpack_from("<8i", data, 16)
    num_fixups, fixup_table_start, vertex_data_start, _ = struct.unpack_from("<4i", data, 48)

    resolved_vertices = []

    if num_fixups > 0:
        for i in range(num_fixups):
            lod, source_vert_id, num_verts = struct.unpack_from("<3i", data, fixup_table_start + i * 12)
            if lod >= 0:
                for v in range(num_verts):
                    v_idx = source_vert_id + v
                    offset = vertex_data_start + v_idx * 48
                    if offset + 48 <= len(data):
                        resolved_vertices.append(data[offset:offset + 48])
    else:
        vert_count = num_lod_verts[0]
        for v in range(vert_count):
            offset = vertex_data_start + v * 48
            if offset + 48 <= len(data):
                resolved_vertices.append(data[offset:offset + 48])

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


def parse_mdl(mdl_path: str) -> SmdModel:
    with open(mdl_path, "rb") as f:
        data = f.read()

    if len(data) < 408:
        raise MdlParseError("MDL header too small")

    magic, version, checksum = struct.unpack_from("<3i", data, 0)
    if magic != 0x54534449:  # 'IDST'
        raise MdlParseError("Not a valid Source MDL file")

    vvd_path = _find_companion_file(mdl_path, [".vvd"])
    if not vvd_path:
        raise MdlParseError(f"Missing companion VVD file for {os.path.basename(mdl_path)}")

    vtx_path = _find_companion_file(mdl_path, [".dx90.vtx", ".dx80.vtx", ".vtx", ".sw.vtx"])
    if not vtx_path:
        raise MdlParseError(f"Missing companion VTX file for {os.path.basename(mdl_path)}")

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

    # Geometry construction
    bodypart_count, bodypart_offset = struct.unpack_from("<2i", data, 0xE8)
    vtx_num_bodyparts, vtx_bodypart_offset = struct.unpack_from("<2i", vtx_data, 28)

    model = SmdModel(version=1, bones=bones, frames=[reference_frame])
    model_vertices = []
    model_triangles = []
    model_materials = set()

    min_bound = [float("inf"), float("inf"), float("inf")]
    max_bound = [float("-inf"), float("-inf"), float("-inf")]

    sg_size = 33 if version >= 49 else 25
    strip_header_size = 35 if version >= 49 else 27

    for bp_i in range(min(bodypart_count, vtx_num_bodyparts)):
        bp_offset = bodypart_offset + bp_i * 16
        _, nummodels, _, modelindex = struct.unpack_from("<4i", data, bp_offset)

        vtx_bp_pos = vtx_bodypart_offset + bp_i * 8
        vtx_nummodels, vtx_model_offset = struct.unpack_from("<2i", vtx_data, vtx_bp_pos)

        available_models = min(nummodels, vtx_nummodels)
        if available_models <= 0:
            continue

        # Choose the first sub-model with valid meshes
        chosen_model_idx = 0
        for test_idx in range(available_models):
            test_m_offset = bp_offset + modelindex + test_idx * 148
            test_nummeshes = struct.unpack_from("<i", data, test_m_offset + 72)[0]
            if test_nummeshes > 0:
                chosen_model_idx = test_idx
                break

        m_i = chosen_model_idx
        m_offset = bp_offset + modelindex + m_i * 148
        _, _, nummeshes, meshindex, _, vertexindex = struct.unpack_from("<ifi3i", data, m_offset + 64)

        vtx_m_pos = vtx_bp_pos + vtx_model_offset + m_i * 8
        _, vtx_lod_offset = struct.unpack_from("<2i", vtx_data, vtx_m_pos)

        # LOD 0
        vtx_lod0_pos = vtx_m_pos + vtx_lod_offset
        vtx_nummeshes, vtx_mesh_offset, _ = struct.unpack_from("<2if", vtx_data, vtx_lod0_pos)

        # Check MeshHeader_t stride (12 bytes aligned vs 9 bytes packed)
        mesh_header_stride = 12
        if vtx_nummeshes > 1:
            test_pos_12 = vtx_lod0_pos + vtx_mesh_offset + 12
            test_sg_12 = struct.unpack_from("<i", vtx_data, test_pos_12)[0]
            if not (1 <= test_sg_12 <= 64):
                mesh_header_stride = 9

        for mesh_i in range(min(nummeshes, vtx_nummeshes)):
            mesh_offset = m_offset + meshindex + mesh_i * 116
            mat_id, _, _, vertexoffset = struct.unpack_from("<4i", data, mesh_offset)

            # Resolve through skin table if available
            actual_mat_id = mat_id
            if skin_table and 0 <= mat_id < len(skin_table):
                actual_mat_id = skin_table[mat_id]

            mat_name = "default"
            if 0 <= actual_mat_id < len(textures):
                mat_name = textures[actual_mat_id]

            model_materials.add(mat_name)

            vtx_mesh_pos = vtx_lod0_pos + vtx_mesh_offset + mesh_i * mesh_header_stride
            num_strip_groups, strip_group_header_offset, _ = struct.unpack_from("<2iB", vtx_data, vtx_mesh_pos)

            for sg_i in range(num_strip_groups):
                sg_pos = vtx_mesh_pos + strip_group_header_offset + sg_i * sg_size
                num_verts, vert_offset, num_indices, index_offset, num_strips, strip_offset, _ = struct.unpack_from(
                    "<6iB", vtx_data, sg_pos
                )

                sg_vert_mapping = []
                for v_i in range(num_verts):
                    v_pos = sg_pos + vert_offset + v_i * 9
                    orig_mesh_vert_id = struct.unpack_from("<H", vtx_data, v_pos + 4)[0]
                    vvd_index = vertexindex + vertexoffset + orig_mesh_vert_id
                    sg_vert_mapping.append(vvd_index)

                raw_indices = vtx_data[sg_pos + index_offset:sg_pos + index_offset + num_indices * 2]
                sg_indices = struct.unpack(f"<{num_indices}H", raw_indices)

                for st_i in range(num_strips):
                    st_pos = sg_pos + strip_offset + st_i * strip_header_size
                    st_num_indices, st_index_offset, _, _, _, st_flags = struct.unpack_from("<4ihB", vtx_data, st_pos)

                    triangles = []
                    if st_flags & 0x02:  # TRISTRIP
                        for k in range(st_num_indices - 2):
                            i0 = sg_indices[st_index_offset + k]
                            i1 = sg_indices[st_index_offset + k + 1]
                            i2 = sg_indices[st_index_offset + k + 2]
                            if k % 2 != 0:
                                i0, i1 = i1, i0
                            if i0 != i1 and i1 != i2 and i0 != i2:
                                triangles.append((i0, i1, i2))
                    else:  # TRILIST
                        for k in range(0, st_num_indices, 3):
                            if k + 2 < st_num_indices:
                                triangles.append((
                                    sg_indices[st_index_offset + k],
                                    sg_indices[st_index_offset + k + 1],
                                    sg_indices[st_index_offset + k + 2],
                                ))

                    for i0, i1, i2 in triangles:
                        tri_indices = []
                        for idx in (i0, i1, i2):
                            if idx >= len(sg_vert_mapping):
                                continue
                            vvd_idx = sg_vert_mapping[idx]
                            if 0 <= vvd_idx < len(vvd_vertices):
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
                                tri_indices.append(smd_idx)

                                for axis in range(3):
                                    min_bound[axis] = min(min_bound[axis], pos[axis])
                                    max_bound[axis] = max(max_bound[axis], pos[axis])

                        if len(tri_indices) == 3:
                            model_triangles.append(
                                SmdTriangle(
                                    material=mat_name,
                                    indices=(tri_indices[0], tri_indices[1], tri_indices[2]),
                                )
                            )

    model.vertices = model_vertices
    model.triangles = model_triangles
    model.materials = model_materials
    model.uv_pre_flipped = False
    model.metadata["material_dirs"] = material_dirs

    if model_triangles:
        model.has_geometry = True
        model.min_bound = tuple(min_bound)
        model.max_bound = tuple(max_bound)

    return model