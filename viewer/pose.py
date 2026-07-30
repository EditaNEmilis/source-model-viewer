import math

import numpy as np


def _normalize_normal_tuple(value):
    x, y, z = value
    length = math.sqrt(x * x + y * y + z * z)

    if length < 0.00000001:
        return (0.0, 0.0, 1.0)

    return (x / length, y / length, z / length)


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


def _distance_squared(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]

    return dx * dx + dy * dy + dz * dz


def _floor_key(position, inverse_cell_size):
    return (
        int(math.floor(position[0] * inverse_cell_size)),
        int(math.floor(position[1] * inverse_cell_size)),
        int(math.floor(position[2] * inverse_cell_size)),
    )


def _round_key(position, inverse_cell_size):
    return (
        int(math.floor(position[0] * inverse_cell_size + 0.5)),
        int(math.floor(position[1] * inverse_cell_size + 0.5)),
        int(math.floor(position[2] * inverse_cell_size + 0.5)),
    )


def _match_vertex_id_groups(reference_positions, overrides, bounds):
    if not reference_positions or not overrides:
        return {}, 0

    min_bound = None
    max_bound = None

    if bounds:
        min_bound, max_bound = bounds

    if not min_bound or not max_bound:
        min_bound = [
            min(vertex[0] for vertex in reference_positions),
            min(vertex[1] for vertex in reference_positions),
            min(vertex[2] for vertex in reference_positions),
        ]
        max_bound = [
            max(vertex[0] for vertex in reference_positions),
            max(vertex[1] for vertex in reference_positions),
            max(vertex[2] for vertex in reference_positions),
        ]

    size_x = max_bound[0] - min_bound[0]
    size_y = max_bound[1] - min_bound[1]
    size_z = max_bound[2] - min_bound[2]

    diagonal = math.sqrt(size_x * size_x + size_y * size_y + size_z * size_z)

    point_cell_size = max(0.00001, diagonal / 500.0)
    point_inverse = 1.0 / point_cell_size

    point_grid = {}

    for index, position in enumerate(reference_positions):
        key = _floor_key(position, point_inverse)
        point_grid.setdefault(key, []).append(index)

    group_epsilon = max(0.00005, diagonal * 0.00002)
    group_inverse = 1.0 / group_epsilon

    groups = {}

    for index, position in enumerate(reference_positions):
        key = _round_key(position, group_inverse)
        groups.setdefault(key, []).append(index)

    search_threshold = max(point_cell_size * 4.0, diagonal * 0.002)
    fallback_threshold = search_threshold * 10.0
    fallback_threshold_squared = fallback_threshold * fallback_threshold

    mapping = {}

    for vta_id, override in sorted(overrides.items()):
        target_position = override[0]

        direct_key = _round_key(target_position, group_inverse)

        if direct_key in groups:
            mapping[vta_id] = groups[direct_key]
            continue

        center_key = _floor_key(target_position, point_inverse)

        best_index = -1
        best_distance = float("inf")

        for radius in (0, 1, 2, 4, 8):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if radius != 0 and max(abs(dx), abs(dy), abs(dz)) != radius:
                            continue

                        key = (
                            center_key[0] + dx,
                            center_key[1] + dy,
                            center_key[2] + dz,
                        )

                        candidates = point_grid.get(key)
                        if not candidates:
                            continue

                        for candidate_index in candidates:
                            candidate_position = reference_positions[candidate_index]
                            distance = _distance_squared(target_position, candidate_position)

                            if distance < best_distance:
                                best_distance = distance
                                best_index = candidate_index

            if best_index != -1 and best_distance <= fallback_threshold_squared:
                break

        if best_index == -1 or best_distance > fallback_threshold_squared:
            continue

        matched_position = reference_positions[best_index]
        group_key = _round_key(matched_position, group_inverse)

        if group_key in groups:
            mapping[vta_id] = groups[group_key]

    affected = set()
    for indices in mapping.values():
        affected.update(indices)

    return mapping, len(affected)


class VertexAnimator:
    def __init__(self):
        self.targets = []
        self.frames = []
        self.times = []
        self.names = []
        self.base_positions = None
        self.base_normals = None

        self.current_frame = 0.0
        self.progress = 0.0
        self.intensity = 1.0
        self.mode = "shape"
        self.match_ids = False

        self.ignored_count = 0
        self.max_vertex_id = -1
        self.first_target_count = 0
        self.matched_count = 0
        self.matched_vertex_count = 0

    def frame_count(self):
        return len(self.frames)

    def set_mode(self, mode):
        if mode in ("shape", "sequence"):
            self.mode = mode

    def set_match_ids(self, enabled):
        self.match_ids = bool(enabled)

    def set_intensity(self, intensity):
        self.intensity = max(-10.0, min(10.0, float(intensity)))

    def set_progress(self, progress):
        count = self.frame_count()

        progress = max(0.0, min(1.0, float(progress)))
        self.progress = progress

        if count <= 1:
            self.current_frame = 0.0
        else:
            self.current_frame = progress * (count - 1)

    def set_frame(self, frame):
        count = self.frame_count()

        if count == 0:
            self.current_frame = 0.0
            self.progress = 0.0
            return

        frame = max(0.0, min(float(frame), count - 1))
        self.current_frame = frame

        if count <= 1:
            self.progress = 0.0
        else:
            self.progress = frame / (count - 1)

    def animation_name(self, index=None):
        if index is None:
            if self.mode == "shape":
                index = int(round(self.current_frame))
            else:
                index = int(math.floor(self.current_frame))

        if 0 <= index < len(self.names):
            return self.names[index]

        return ""

    def build_frames(self, targets, static_positions, static_normals, bounds):
        self.targets = targets if targets else []
        self.frames = []
        self.times = []
        self.names = []
        self.base_positions = None
        self.base_normals = None

        self.ignored_count = 0
        self.max_vertex_id = -1
        self.first_target_count = 0
        self.matched_count = 0
        self.matched_vertex_count = 0

        if not self.targets or static_positions is None:
            return

        vertex_count = len(static_positions)

        base_positions = static_positions.astype(np.float32).copy()
        base_normals = static_normals.astype(np.float32).copy()

        sorted_targets = sorted(self.targets, key=lambda target: target.time)

        first = sorted_targets[0]
        self.first_target_count = len(first.overrides)

        id_mapping = {}

        if self.match_ids:
            reference_list = [tuple(position) for position in static_positions]

            id_mapping, matched_vertex_count = _match_vertex_id_groups(
                reference_list,
                first.overrides,
                bounds,
            )
            self.matched_count = len(id_mapping)
            self.matched_vertex_count = matched_vertex_count

        def resolve(vertex_id):
            if self.match_ids:
                mapped = id_mapping.get(vertex_id)

                if mapped:
                    return mapped

            if 0 <= vertex_id < vertex_count:
                return (vertex_id,)

            return ()

        for vertex_id, override in first.overrides.items():
            if vertex_id > self.max_vertex_id:
                self.max_vertex_id = vertex_id

            resolved = resolve(vertex_id)
            if not resolved:
                continue

            position, normal = override
            normalized = _normalize_normal_tuple(normal)

            for rid in resolved:
                if 0 <= rid < vertex_count:
                    base_positions[rid] = position
                    base_normals[rid] = normalized

        for target in sorted_targets:
            positions = base_positions.copy()
            normals = base_normals.copy()

            for vertex_id, override in target.overrides.items():
                if vertex_id > self.max_vertex_id:
                    self.max_vertex_id = vertex_id

                resolved = resolve(vertex_id)

                if not resolved:
                    self.ignored_count += 1
                    continue

                position, normal = override
                normalized = _normalize_normal_tuple(normal)

                for rid in resolved:
                    if 0 <= rid < vertex_count:
                        positions[rid] = position
                        normals[rid] = normalized

            self.frames.append((positions, normals))
            self.times.append(target.time)

            name = target.name.strip() if target.name else ""
            if not name:
                name = f"Target {len(self.frames)}"

            self.names.append(name)

        self.base_positions = base_positions
        self.base_normals = base_normals

    def compute_pose(self, static_positions, static_normals):
        count = self.frame_count()

        if count == 0:
            return static_positions, static_normals

        frame = self.current_frame

        if self.mode == "shape":
            index = int(round(frame))
            index = max(0, min(index, count - 1))
            target_positions, target_normals = self.frames[index]
        else:
            if frame <= 0.0:
                target_positions, target_normals = self.frames[0]
            elif frame >= count - 1:
                target_positions, target_normals = self.frames[-1]
            else:
                index = int(math.floor(frame))
                amount = frame - index

                positions_a, normals_a = self.frames[index]
                positions_b, normals_b = self.frames[index + 1]

                target_positions = positions_a + (positions_b - positions_a) * amount
                target_normals = _normalize_normals(
                    normals_a + (normals_b - normals_a) * amount
                )

        intensity = self.intensity

        if self.base_positions is None or intensity == 1.0:
            return target_positions, target_normals

        if intensity == 0.0:
            return self.base_positions, self.base_normals

        blended_positions = self.base_positions + (
            target_positions - self.base_positions
        ) * intensity

        blended_normals = _normalize_normals(
            self.base_normals + (target_normals - self.base_normals) * intensity
        )

        return blended_positions, blended_normals