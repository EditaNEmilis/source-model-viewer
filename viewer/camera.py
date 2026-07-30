import math

from OpenGL.GLU import gluLookAt


def _normalize(value):
    x, y, z = value
    length = math.sqrt(x * x + y * y + z * z)

    if length < 0.00000001:
        return (0.0, 0.0, 0.0)

    return (x / length, y / length, z / length)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


class Camera:
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = [0.0, 0.0, 0.0]
        self.distance = 12.0
        self.yaw = math.radians(35.0)
        self.pitch = math.radians(20.0)
        self.fov = 45.0
        self.near = 0.1
        self.far = 10000.0
        self.radius = 1.0

        self.rotate_speed = 0.006
        self.pan_speed = 0.001
        self.zoom_speed = 0.1
        self.zoom_drag_speed = 0.002

    def eye(self):
        cos_pitch = math.cos(self.pitch)

        x = self.target[0] + self.distance * cos_pitch * math.cos(self.yaw)
        y = self.target[1] + self.distance * cos_pitch * math.sin(self.yaw)
        z = self.target[2] + self.distance * math.sin(self.pitch)

        return (x, y, z)

    def rotate(self, dx, dy):
        self.yaw += dx * self.rotate_speed
        self.pitch -= dy * self.rotate_speed

        limit = math.radians(89.0)

        if self.pitch > limit:
            self.pitch = limit

        if self.pitch < -limit:
            self.pitch = -limit

    def pan_vectors(self):
        eye = self.eye()
        forward = _normalize(_subtract(self.target, eye))
        world_up = (0.0, 0.0, 1.0)

        right = _normalize(_cross(forward, world_up))

        if right == (0.0, 0.0, 0.0):
            right = (1.0, 0.0, 0.0)

        up = _normalize(_cross(right, forward))

        return right, up

    def pan(self, dx, dy):
        right, up = self.pan_vectors()

        scale = max(0.0005, self.distance * self.pan_speed)

        self.target[0] += -right[0] * dx * scale + up[0] * dy * scale
        self.target[1] += -right[1] * dx * scale + up[1] * dy * scale
        self.target[2] += -right[2] * dx * scale + up[2] * dy * scale

        self._update_clip()

    def zoom(self, wheel_delta):
        if wheel_delta > 0:
            factor = 1.0 - self.zoom_speed
        else:
            factor = 1.0 + self.zoom_speed

        self.distance *= factor
        self.distance = max(0.05, self.distance)

        self._update_clip()

    def zoom_drag(self, dy):
        value = dy * self.zoom_drag_speed
        value = max(-2.0, min(2.0, value))

        factor = math.exp(value)

        self.distance *= factor
        self.distance = max(0.05, self.distance)

        self._update_clip()

    def fit(self, min_bound, max_bound):
        if not min_bound or not max_bound:
            self.reset()
            return

        center = [
            (min_bound[0] + max_bound[0]) * 0.5,
            (min_bound[1] + max_bound[1]) * 0.5,
            (min_bound[2] + max_bound[2]) * 0.5,
        ]

        size = [
            max_bound[0] - min_bound[0],
            max_bound[1] - min_bound[1],
            max_bound[2] - min_bound[2],
        ]

        radius = math.sqrt(size[0] * size[0] + size[1] * size[1] + size[2] * size[2]) * 0.5

        if radius < 0.0001:
            radius = 1.0

        self.target = center
        self.radius = radius

        fov_radians = math.radians(self.fov)
        self.distance = (radius / math.tan(fov_radians * 0.5)) * 1.35

        self.yaw = math.radians(35.0)
        self.pitch = math.radians(20.0)

        self._update_clip()

    def _update_clip(self):
        self.near = max(0.01, self.distance * 0.001)
        self.far = max(100.0, self.distance * 100.0 + self.radius * 100.0)

    def apply(self):
        eye = self.eye()

        gluLookAt(
            eye[0],
            eye[1],
            eye[2],
            self.target[0],
            self.target[1],
            self.target[2],
            0.0,
            0.0,
            1.0,
        )