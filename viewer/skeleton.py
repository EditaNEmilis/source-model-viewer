import math

from typing import Dict, List, Tuple


Matrix = Tuple[float, ...]
Vector3 = Tuple[float, float, float]
Transform = Tuple[Vector3, Vector3]


def identity_matrix() -> Matrix:
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def multiply_matrices(a: Matrix, b: Matrix) -> Matrix:
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


def translation_matrix(position: Vector3) -> Matrix:
    return (
        1.0, 0.0, 0.0, position[0],
        0.0, 1.0, 0.0, position[1],
        0.0, 0.0, 1.0, position[2],
        0.0, 0.0, 0.0, 1.0,
    )


def euler_to_matrix(rotation: Vector3) -> Matrix:
    rx, ry, rz = rotation

    cx = math.cos(rx)
    sx = math.sin(rx)

    cy = math.cos(ry)
    sy = math.sin(ry)

    cz = math.cos(rz)
    sz = math.sin(rz)

    # Common SMD style Euler conversion.
    # This may need adjustment for some exporters.
    return (
        cy * cz,
        sx * sy * cz - cx * sz,
        cx * sy * cz + sx * sz,
        0.0,

        cy * sz,
        sx * sy * sz + cx * cz,
        cx * sy * sz - sx * cz,
        0.0,

        -sy,
        sx * cy,
        cx * cy,
        0.0,

        0.0,
        0.0,
        0.0,
        1.0,
    )


def local_transform_matrix(transform: Transform) -> Matrix:
    position, rotation = transform
    return multiply_matrices(translation_matrix(position), euler_to_matrix(rotation))


def invert_affine_matrix(matrix: Matrix) -> Matrix:
    r00 = matrix[0]
    r01 = matrix[1]
    r02 = matrix[2]
    tx = matrix[3]

    r10 = matrix[4]
    r11 = matrix[5]
    r12 = matrix[6]
    ty = matrix[7]

    r20 = matrix[8]
    r21 = matrix[9]
    r22 = matrix[10]
    tz = matrix[11]

    inv_tx = -(r00 * tx + r10 * ty + r20 * tz)
    inv_ty = -(r01 * tx + r11 * ty + r21 * tz)
    inv_tz = -(r02 * tx + r12 * ty + r22 * tz)

    return (
        r00, r10, r20, inv_tx,
        r01, r11, r21, inv_ty,
        r02, r12, r22, inv_tz,
        0.0, 0.0, 0.0, 1.0,
    )


def transform_point(matrix: Matrix, point: Vector3) -> Vector3:
    x, y, z = point

    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3],
        matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7],
        matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11],
    )


def transform_vector(matrix: Matrix, vector: Vector3) -> Vector3:
    x, y, z = vector

    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z,
        matrix[4] * x + matrix[5] * y + matrix[6] * z,
        matrix[8] * x + matrix[9] * y + matrix[10] * z,
    )


def lerp_vector3(a: Vector3, b: Vector3, amount: float) -> Vector3:
    return (
        a[0] + (b[0] - a[0]) * amount,
        a[1] + (b[1] - a[1]) * amount,
        a[2] + (b[2] - a[2]) * amount,
    )


class SkeletonRig:
    def __init__(self, bones, reference_transforms: Dict[int, Transform]):
        self.bones = bones
        self.bone_ids = [bone.bone_id for bone in bones]
        self.parent = {bone.bone_id: bone.parent_id for bone in bones}
        self.name_to_id = {bone.name: bone.bone_id for bone in bones}
        self.id_to_name = {bone.bone_id: bone.name for bone in bones}

        self.bind_local: Dict[int, Transform] = {}

        for bone in bones:
            self.bind_local[bone.bone_id] = reference_transforms.get(
                bone.bone_id,
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            )

        self.bind_world: Dict[int, Matrix] = {}
        self.inv_bind: Dict[int, Matrix] = {}

        self._build_bind_pose()

    def _build_bind_pose(self):
        def compute(bone_id: int):
            if bone_id in self.bind_world:
                return self.bind_world[bone_id]

            parent_id = self.parent.get(bone_id, -1)
            parent_matrix = identity_matrix()

            if parent_id in self.bind_local and parent_id != bone_id:
                parent_matrix = compute(parent_id)

            local_matrix = local_transform_matrix(self.bind_local[bone_id])
            world_matrix = multiply_matrices(parent_matrix, local_matrix)

            self.bind_world[bone_id] = world_matrix
            self.inv_bind[bone_id] = invert_affine_matrix(world_matrix)

            return world_matrix

        for bone_id in self.bone_ids:
            compute(bone_id)


def evaluate_world_matrices(
    rig: SkeletonRig,
    local_transforms: Dict[int, Transform],
) -> Dict[int, Matrix]:
    world_matrices: Dict[int, Matrix] = {}

    def compute(bone_id: int) -> Matrix:
        if bone_id in world_matrices:
            return world_matrices[bone_id]

        parent_id = rig.parent.get(bone_id, -1)
        parent_matrix = identity_matrix()

        if parent_id in rig.bind_local and parent_id != bone_id:
            parent_matrix = compute(parent_id)

        transform = local_transforms.get(bone_id, rig.bind_local[bone_id])
        local_matrix = local_transform_matrix(transform)
        world_matrix = multiply_matrices(parent_matrix, local_matrix)

        world_matrices[bone_id] = world_matrix

        return world_matrix

    for bone_id in rig.bone_ids:
        compute(bone_id)

    return world_matrices


class SkeletalAnimation:
    def __init__(self, animation_model, rig: SkeletonRig):
        self.bone_ids = rig.bone_ids
        self.bind_local = rig.bind_local
        self.full_locals: List[Dict[int, Transform]] = []
        self.extra_locals: List[Dict[int, Transform]] = []
        self.unmapped_names: Dict[int, str] = {}
        self.frame_count = 0

        self.anim_bone_count = len(animation_model.bones)
        self.name_mapped_count = 0
        self.driven_ref_ids = set()

        frames = sorted(animation_model.frames, key=lambda frame: frame.time)

        if not frames:
            return

        lower_name_to_id = {
            name.lower(): bone_id
            for name, bone_id in rig.name_to_id.items()
        }

        bone_id_mapping: Dict[int, int] = {}

        for animation_bone in animation_model.bones:
            reference_id = rig.name_to_id.get(animation_bone.name)

            if reference_id is None:
                reference_id = lower_name_to_id.get(animation_bone.name.lower())

            if reference_id is not None:
                bone_id_mapping[animation_bone.bone_id] = reference_id

        self.name_mapped_count = len(bone_id_mapping)

        for animation_bone in animation_model.bones:
            if animation_bone.bone_id not in bone_id_mapping:
                self.unmapped_names[animation_bone.bone_id] = animation_bone.name

        previous: Dict[int, Transform] = dict(rig.bind_local)
        previous_extra: Dict[int, Transform] = {}

        for frame in frames:
            current: Dict[int, Transform] = previous.copy()
            current_extra: Dict[int, Transform] = previous_extra.copy()

            for animation_bone_id, transform in frame.transforms.items():
                reference_id = bone_id_mapping.get(animation_bone_id)

                if reference_id is not None:
                    current[reference_id] = transform
                    self.driven_ref_ids.add(reference_id)
                else:
                    current_extra[animation_bone_id] = transform

            self.full_locals.append(current)
            self.extra_locals.append(current_extra)

            previous = current
            previous_extra = current_extra

        self.frame_count = len(self.full_locals)

    def sample(self, frame: float) -> Dict[int, Transform]:
        if self.frame_count == 0:
            return {}

        if self.frame_count == 1:
            return self.full_locals[0]

        if frame <= 0.0:
            return self.full_locals[0]

        if frame >= self.frame_count - 1:
            return self.full_locals[-1]

        index = int(math.floor(frame))
        amount = frame - index

        a = self.full_locals[index]
        b = self.full_locals[index + 1]

        result: Dict[int, Transform] = {}

        for bone_id in self.bone_ids:
            default = self.bind_local[bone_id]

            transform_a = a.get(bone_id, default)
            transform_b = b.get(bone_id, default)

            position = lerp_vector3(transform_a[0], transform_b[0], amount)
            rotation = lerp_vector3(transform_a[1], transform_b[1], amount)

            result[bone_id] = (position, rotation)

        return result

    def sample_unmapped(self, frame: float) -> Dict[int, Transform]:
        if self.frame_count == 0 or not self.extra_locals:
            return {}

        if self.frame_count == 1:
            return self.extra_locals[0]

        if frame <= 0.0:
            return self.extra_locals[0]

        if frame >= self.frame_count - 1:
            return self.extra_locals[-1]

        index = int(math.floor(frame))
        amount = frame - index

        a = self.extra_locals[index]
        b = self.extra_locals[index + 1]

        default: Transform = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        result: Dict[int, Transform] = {}

        for bone_id in self.unmapped_names:
            transform_a = a.get(bone_id, default)
            transform_b = b.get(bone_id, default)

            position = lerp_vector3(transform_a[0], transform_b[0], amount)
            rotation = lerp_vector3(transform_a[1], transform_b[1], amount)

            result[bone_id] = (position, rotation)

        return result
        
def quat_to_euler(q):
    x, y, z, w = q

    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

    wx = w * x
    wy = w * y
    wz = w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m10 = 2.0 * (xy + wz)
    m20 = 2.0 * (xz - wy)

    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)

    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    cy = math.sqrt(m00 * m00 + m10 * m10)

    if cy > 0.000001:
        rx = math.atan2(m21, m22)
        ry = math.atan2(-m20, cy)
        rz = math.atan2(m10, m00)
    else:
        rx = math.atan2(-m12, m11)
        ry = math.atan2(-m20, cy)
        rz = 0.0

    return (rx, ry, rz)


def quat_slerp(a, b, amount):
    ax, ay, az, aw = a
    bx, by, bz, bw = b

    dot = ax * bx + ay * by + az * bz + aw * bw

    if dot < 0.0:
        dot = -dot
        bx, by, bz, bw = -bx, -by, -bz, -bw

    if dot > 0.9995:
        rx = ax + (bx - ax) * amount
        ry = ay + (by - ay) * amount
        rz = az + (bz - az) * amount
        rw = aw + (bw - aw) * amount
    else:
        theta = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta = math.sin(theta)

        wa = math.sin((1.0 - amount) * theta) / sin_theta
        wb = math.sin(amount * theta) / sin_theta

        rx = ax * wa + bx * wb
        ry = ay * wa + by * wb
        rz = az * wa + bz * wb
        rw = aw * wa + bw * wb

    length = math.sqrt(rx * rx + ry * ry + rz * rz + rw * rw)

    if length < 0.00000001:
        return (0.0, 0.0, 0.0, 1.0)

    return (rx / length, ry / length, rz / length, rw / length)