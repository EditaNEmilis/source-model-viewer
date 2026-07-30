import numpy as np

from viewer.skeleton import evaluate_world_matrices


def _normalize_normals(normals):
    if normals.size == 0:
        return normals

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)

    out = np.divide(
        normals,
        lengths,
        out=np.zeros_like(normals),
        where=lengths > 0.00000001,
    )

    zero = lengths[:, 0] < 0.00000001

    if np.any(zero):
        out[zero] = (0.0, 0.0, 1.0)

    return out


class Skinning:
    def __init__(self):
        self.rig = None
        self.vertex_weights = []
        self.weight_bone_ids = None
        self.weight_values = None
        self.bone_index_map = {}
        self.skin_matrix_count = 0
        self.skin_matrices = None

        self.animation = None
        self.frame = 0.0
        self.enabled = False

    def set_rig(self, rig, vertex_weights):
        self.rig = rig
        self.vertex_weights = vertex_weights if vertex_weights else []
        self._build_weight_arrays()

    def _build_weight_arrays(self):
        self.weight_bone_ids = None
        self.weight_values = None
        self.bone_index_map = {}
        self.skin_matrix_count = 0
        self.skin_matrices = None

        if self.rig is None or not self.vertex_weights:
            return

        all_bone_ids = set(self.rig.bone_ids)

        for weights in self.vertex_weights:
            for bone_id, _ in weights:
                if bone_id >= 0:
                    all_bone_ids.add(bone_id)

        self.bone_index_map = {
            bone_id: index + 1
            for index, bone_id in enumerate(sorted(all_bone_ids))
        }
        self.skin_matrix_count = len(all_bone_ids) + 1

        max_weights = 8
        weight_count = max((len(w) for w in self.vertex_weights), default=1)
        used = min(max_weights, weight_count)

        vertex_count = len(self.vertex_weights)

        bone_ids = np.zeros((vertex_count, used), dtype=np.int32)
        values = np.zeros((vertex_count, used), dtype=np.float32)

        for i, weights in enumerate(self.vertex_weights):
            for j, weight_entry in enumerate(weights[:used]):
                bone_id, weight = weight_entry
                bone_ids[i, j] = self.bone_index_map.get(bone_id, 0)
                values[i, j] = weight

        sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, sums, out=values, where=sums > 0.000001)

        self.weight_bone_ids = bone_ids
        self.weight_values = values

    def set_animation(self, animation):
        self.animation = animation
        self.frame = 0.0
        self.skin_matrices = None

    def set_frame(self, frame):
        self.frame = frame
        self._update_matrices()

    def _update_matrices(self):
        if (
            not self.enabled
            or self.animation is None
            or self.rig is None
            or self.skin_matrix_count <= 0
        ):
            self.skin_matrices = None
            return

        local_transforms = self.animation.sample(self.frame)
        world_matrices = evaluate_world_matrices(self.rig, local_transforms)

        if (
            self.skin_matrices is None
            or self.skin_matrices.shape[0] != self.skin_matrix_count
        ):
            self.skin_matrices = np.zeros(
                (self.skin_matrix_count, 4, 4), dtype=np.float32
            )

        self.skin_matrices.fill(0.0)

        for i in range(4):
            self.skin_matrices[:, i, i] = 1.0

        for bone_id, world_matrix in world_matrices.items():
            index = self.bone_index_map.get(bone_id)

            if index is None:
                continue

            inv_bind = self.rig.inv_bind.get(bone_id)

            if inv_bind is None:
                continue

            skin = _multiply(world_matrix, inv_bind)

            self.skin_matrices[index] = np.array(
                skin, dtype=np.float32
            ).reshape(4, 4)

    def apply(self, positions, normals):
        if (
            not self.enabled
            or self.skin_matrices is None
            or self.weight_bone_ids is None
            or self.weight_values is None
        ):
            return positions, normals

        if len(positions) != len(self.weight_values):
            return positions, normals

        weights = self.weight_values
        bone_ids = self.weight_bone_ids
        matrices = self.skin_matrices

        gathered = matrices[bone_ids]

        vertex_count = len(positions)

        positions_h = np.empty((vertex_count, 4), dtype=np.float32)
        positions_h[:, :3] = positions
        positions_h[:, 3] = 1.0

        transformed = np.einsum(
            "vwij,vj->vwi", gathered, positions_h, optimize=True
        )
        skinned_positions = np.sum(transformed * weights[:, :, None], axis=1)

        transformed_normals = np.einsum(
            "vwij,vj->vwi", gathered[:, :, :3, :3], normals, optimize=True
        )
        skinned_normals = np.sum(transformed_normals * weights[:, :, None], axis=1)
        skinned_normals = _normalize_normals(skinned_normals)

        weight_sums = weights.sum(axis=1)
        zero = weight_sums < 0.000001

        if np.any(zero):
            skinned_positions[zero, :3] = positions[zero]
            skinned_normals[zero] = normals[zero]

        return skinned_positions[:, :3], skinned_normals


def _multiply(a, b):
    result = [0.0] * 16

    for row in range(4):
        for col in range(4):
            result[row * 4 + col] = (
                a[row * 4 + 0] * b[0 * 4 + col]
                + a[row * 4 + 1] * b[1 * 4 + col]
                + a[row * 4 + 2] * b[2 * 4 + col]
                + a[row * 4 + 3] * b[3 * 4 + col]
            )

    return tuple(result)