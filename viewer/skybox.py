"""Viewport skybox loading from viewer/assets/skybox/.

Each subfolder of the skybox directory is one selectable skybox:

* Cubemap set: six faces that share a filename prefix and end in
  ``rt``, ``lf``, ``bk``, ``ft``, ``up`` or ``dn`` (with or without an
  underscore before the suffix), for example
  ``sky_day01/sky_day01_01bk.vtf``. Faces may be ``.vtf`` files or
  common image files (``.png``, ``.jpg``, ``.bmp``, ``.webp``, ...).
* Panorama: a single image file (``.png``, ``.jpg``, ``.vtf``, ...),
  treated as an equirectangular panorama and converted to cube faces
  when loaded.

Loose single image files directly in the skybox directory are also
offered as panorama entries named after the file.

Viewer coordinates are Z-up, X/Y are ground plane axes. Source suffix
mapping: ``rt`` is -X, ``lf`` is +X, ``ft`` is +Y, ``bk`` is -Y,
``up`` is +Z, ``dn`` is -Z. (Verified against sky_day01: with this
mapping the shared edges of adjacent faces match exactly, while
``rt`` on +X leaves visible seams.)
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

SKYBOX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "skybox"
)

IMAGE_EXTENSIONS = (
    ".vtf", ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".webp",
)

# Draw/load order for the six faces.
FACE_ORDER = ("ft", "bk", "rt", "lf", "up", "dn")

# (forward, up hint) per face suffix, in viewer coordinates.
_FACE_AXES = {
    "ft": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "bk": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "rt": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "lf": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "up": ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    "dn": ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
}


class SkyboxError(ValueError):
    pass


def face_basis(suffix):
    """(forward, right, true_up) unit vectors for one face."""
    forward = np.array(_FACE_AXES[suffix][0], dtype=np.float64)
    up_hint = np.array(_FACE_AXES[suffix][1], dtype=np.float64)
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    up_hint = up_hint - forward * float(np.dot(up_hint, forward))
    up_hint = up_hint / max(np.linalg.norm(up_hint), 1e-9)
    right = np.cross(forward, up_hint)
    right = right / max(np.linalg.norm(right), 1e-9)
    # Re-derive up so the basis is orthonormal: cross(right, forward)
    # points the same way as up_hint for the faces above.
    true_up = np.cross(right, forward)
    return forward, right, true_up


@dataclass
class SkyboxEntry:
    name: str
    kind: str  # "cubemap" or "panorama"
    folder: str
    faces: Dict[str, str] = field(default_factory=dict)
    panorama: Optional[str] = None

    def label(self):
        return f"{self.name} ({self.kind})"


def _is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def _cubemap_groups(filenames):
    """Group filenames by cubemap prefix: {prefix: {suffix: filename}}.

    Source files put the face suffix at the end with or without a
    separator, e.g. ``sky_day01_01bk.vtf`` and ``sky_day01_bk.vtf``.
    """
    groups = {}
    for filename in filenames:
        stem = os.path.splitext(filename)[0].lower()
        for suffix in FACE_ORDER:
            if stem.endswith("_" + suffix):
                prefix = stem[: -(len(suffix) + 1)]
            elif stem.endswith(suffix) and len(stem) > len(suffix):
                prefix = stem[: -len(suffix)].rstrip("_")
            else:
                continue
            groups.setdefault(prefix, {})[suffix] = filename
            break
    return {
        prefix: faces
        for prefix, faces in groups.items()
        if all(suffix in faces for suffix in FACE_ORDER)
    }


def discover_skyboxes(base_dir=None):
    """List usable skyboxes in ``base_dir`` (default: assets/skybox)."""
    base = base_dir or SKYBOX_DIR
    entries = []
    if not os.path.isdir(base):
        return entries

    for folder_name in sorted(os.listdir(base)):
        folder = os.path.join(base, folder_name)
        if not os.path.isdir(folder):
            continue
        filenames = sorted(f for f in os.listdir(folder) if _is_image(f))
        if not filenames:
            continue
        groups = _cubemap_groups(filenames)
        if groups:
            prefix, faces = sorted(groups.items())[0]
            entries.append(
                SkyboxEntry(
                    name=folder_name,
                    kind="cubemap",
                    folder=folder,
                    faces={s: os.path.join(folder, faces[s]) for s in FACE_ORDER},
                )
            )
        elif len(filenames) == 1:
            entries.append(
                SkyboxEntry(
                    name=folder_name,
                    kind="panorama",
                    folder=folder,
                    panorama=os.path.join(folder, filenames[0]),
                )
            )

    # Loose panorama files directly in the skybox directory.
    for filename in sorted(os.listdir(base)):
        path = os.path.join(base, filename)
        if os.path.isfile(path) and _is_image(filename):
            entries.append(
                SkyboxEntry(
                    name=os.path.splitext(filename)[0],
                    kind="panorama",
                    folder=base,
                    panorama=path,
                )
            )
    return entries


def find_skybox(name, base_dir=None):
    """Entry by name (folder name or loose file stem), or None."""
    if not name:
        return None
    for entry in discover_skyboxes(base_dir):
        if entry.name == name:
            return entry
    return None


def load_image_rgba(path):
    """Load one image file (.vtf or a common format) as RGBA uint8."""
    if path.lower().endswith(".vtf"):
        from viewer.vtf_parser import parse_vtf

        _info, rgba = parse_vtf(path)
        return np.ascontiguousarray(rgba, dtype=np.uint8)

    from PySide6.QtGui import QImage

    image = QImage(path)
    if image.isNull():
        raise SkyboxError(f"Could not read image file: {path}")
    image = image.convertToFormat(QImage.Format_RGBA8888)
    width, height = image.width(), image.height()
    bytes_per_line = image.bytesPerLine()
    raw = bytes(image.constBits())
    if bytes_per_line == width * 4:
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    else:
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(height, bytes_per_line)
        rgba = rgba[:, : width * 4].reshape(height, width, 4)
    return np.ascontiguousarray(rgba, dtype=np.uint8)


def equirect_to_cube(rgba, face_size=512):
    """Convert an equirectangular panorama to six cube faces.

    Returns a list of RGBA arrays in FACE_ORDER. Bilinear sampling,
    with wrapping on the horizontal axis and clamping at the poles.
    """
    height, width = rgba.shape[0], rgba.shape[1]
    source = rgba.astype(np.float32)
    faces = []

    yy, xx = np.mgrid[0:face_size, 0:face_size]
    s = (xx.astype(np.float64) + 0.5) / face_size * 2.0 - 1.0
    t = 1.0 - (yy.astype(np.float64) + 0.5) / face_size * 2.0

    for suffix in FACE_ORDER:
        forward, right, true_up = face_basis(suffix)
        dirs = (
            forward[None, None, :]
            + right[None, None, :] * s[:, :, None]
            + true_up[None, None, :] * t[:, :, None]
        )
        dirs = dirs / np.maximum(
            np.linalg.norm(dirs, axis=2, keepdims=True), 1e-12
        )

        longitude = np.arctan2(dirs[:, :, 0], dirs[:, :, 1])
        u = longitude / (2.0 * np.pi) + 0.5
        v = np.arccos(np.clip(dirs[:, :, 2], -1.0, 1.0)) / np.pi

        px = u * width - 0.5
        py = v * height - 0.5
        x0 = np.floor(px).astype(np.int64)
        y0 = np.floor(py).astype(np.int64)
        fx = (px - x0)[:, :, None].astype(np.float32)
        fy = (py - y0)[:, :, None].astype(np.float32)
        x1 = x0 + 1
        y1 = y0 + 1
        y0c = np.clip(y0, 0, height - 1)
        y1c = np.clip(y1, 0, height - 1)
        x0w = np.mod(x0, width)
        x1w = np.mod(x1, width)

        top = source[y0c, x0w] * (1.0 - fx) + source[y0c, x1w] * fx
        bottom = source[y1c, x0w] * (1.0 - fx) + source[y1c, x1w] * fx
        face = top * (1.0 - fy) + bottom * fy
        faces.append(np.clip(face, 0, 255).astype(np.uint8))
    return faces


def load_skybox_faces(entry):
    """Load a skybox entry as six RGBA arrays in FACE_ORDER."""
    if entry.kind == "cubemap":
        return [load_image_rgba(entry.faces[suffix]) for suffix in FACE_ORDER]
    if entry.kind == "panorama":
        rgba = load_image_rgba(entry.panorama)
        face_size = int(max(128, min(1024, rgba.shape[1] // 4)))
        return equirect_to_cube(rgba, face_size=face_size)
    raise SkyboxError(f"Unknown skybox kind: {entry.kind!r}")


def load_skybox(name, base_dir=None):
    """Load a skybox by name; raises SkyboxError when not found."""
    entry = find_skybox(name, base_dir)
    if entry is None:
        raise SkyboxError(f"Skybox not found: {name!r}")
    return load_skybox_faces(entry)
