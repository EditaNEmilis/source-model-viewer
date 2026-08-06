import math
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


NODE_PATTERN = re.compile(r'^\s*(-?\d+)\s+"(.*)"\s+(-?\d+)\s*$')


@dataclass
class SmdBone:
    bone_id: int
    name: str
    parent_id: int


@dataclass
class SmdFrame:
    time: int
    transforms: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]


@dataclass
class SmdVertex:
    parent_bone: int
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    uv: Tuple[float, float]
    links: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class SmdTriangle:
    material: str
    indices: Tuple[int, int, int]


@dataclass
class VertexTarget:
    time: int
    overrides: Dict[int, Tuple[Tuple[float, float, float], Tuple[float, float, float]]]
    name: str = ""
    corrected: bool = False


@dataclass
class SmdModel:
    version: int = 1
    bones: List[SmdBone] = field(default_factory=list)
    frames: List[SmdFrame] = field(default_factory=list)
    vertices: List[SmdVertex] = field(default_factory=list)
    triangles: List[SmdTriangle] = field(default_factory=list)
    materials: Set[str] = field(default_factory=set)
    vertex_targets: List[VertexTarget] = field(default_factory=list)
    frame_names: Dict[int, str] = field(default_factory=dict)
    min_bound: Optional[Tuple[float, float, float]] = None
    max_bound: Optional[Tuple[float, float, float]] = None
    has_geometry: bool = False
    has_animation: bool = False
    uv_pre_flipped: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def _read_text_lines(path: str) -> List[str]:
    with open(path, "rb") as handle:
        data = handle.read()

    if not data:
        return []

    if data[:3] == b"\xef\xbb\xbf":
        text = data.decode("utf-8-sig", errors="ignore")

    elif data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16", errors="ignore")

    else:
        sample = data[:4096]

        if b"\x00" in sample:
            try:
                text = data.decode("utf-16", errors="ignore")
            except UnicodeError:
                text = data.decode("utf-8", errors="ignore")
        else:
            text = data.decode("utf-8", errors="ignore")

    return text.splitlines()


def _split_comment(line: str) -> Tuple[str, str]:
    result = []
    in_quotes = False
    i = 0
    comment = ""

    while i < len(line):
        character = line[i]

        if character == '"':
            in_quotes = not in_quotes
            result.append(character)
        elif not in_quotes:
            if character in "#;":
                comment = line[i + 1:]
                break

            if character == "/" and i + 1 < len(line) and line[i + 1] == "/":
                comment = line[i + 2:]
                break

            result.append(character)
        else:
            result.append(character)

        i += 1

    return "".join(result).strip(), comment.strip()


def _normalize_normal(normal: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = normal
    length = math.sqrt(x * x + y * y + z * z)

    if length < 0.00000001:
        return (0.0, 0.0, 1.0)

    return (x / length, y / length, z / length)


def _parse_time(parts: List[str]) -> int:
    if len(parts) <= 1:
        return 0

    try:
        return int(parts[1])
    except ValueError:
        try:
            return int(float(parts[1]))
        except ValueError:
            return 0


def _parse_optional_links(parts: List[str], start_index: int) -> List[Tuple[int, float]]:
    links = []

    if start_index >= len(parts):
        return links

    try:
        link_count = int(parts[start_index])
    except ValueError:
        return links

    if link_count < 0 or link_count > 1024:
        return links

    index = start_index + 1

    for _ in range(link_count):
        if index + 1 >= len(parts):
            break

        try:
            bone_id = int(parts[index])
            weight = float(parts[index + 1])
        except ValueError:
            break

        links.append((bone_id, weight))
        index += 2

    return links


def _parse_vertex_line(line: str) -> Optional[SmdVertex]:
    parts = line.split()

    if len(parts) < 9:
        return None

    try:
        parent_bone = int(parts[0])

        position = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
        )

        normal = (
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
        )

        uv = (
            float(parts[7]),
            float(parts[8]),
        )

    except ValueError:
        return None

    links = _parse_optional_links(parts, 9)

    normal = _normalize_normal(normal)

    return SmdVertex(
        parent_bone=parent_bone,
        position=position,
        normal=normal,
        uv=uv,
        links=links,
    )


def _parse_vertex_animation_line(
    line: str,
) -> Optional[Tuple[int, Tuple[float, float, float], Tuple[float, float, float]]]:
    parts = line.split()

    if len(parts) < 7:
        return None

    try:
        vertex_id = int(parts[0])

        position = (
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
        )

        normal = (
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
        )

    except ValueError:
        return None

    normal = _normalize_normal(normal)

    return vertex_id, position, normal


def parse_smd(path: str) -> SmdModel:
    model = SmdModel()

    min_bound = [math.inf, math.inf, math.inf]
    max_bound = [-math.inf, -math.inf, -math.inf]

    state = None
    current_material = None
    pending_vertices = []
    current_frame = None
    current_target = None

    section_names = {"nodes", "skeleton", "triangles", "vertexanimation"}

    lines = _read_text_lines(path)

    for raw_line in lines:
        line, comment = _split_comment(raw_line)

        if not line:
            continue

        lowered = line.lower()

        if state is None and lowered.startswith("version"):
            parts = line.split()

            if len(parts) > 1:
                try:
                    model.version = int(parts[1])
                except ValueError:
                    model.version = 1

            continue

        # Some exporters omit end before starting the next block.
        if lowered in section_names and lowered != state:
            if state == "skeleton" and current_frame is not None:
                model.frames.append(current_frame)
                current_frame = None

            if state == "vertexanimation" and current_target is not None:
                model.vertex_targets.append(current_target)
                current_target = None

            state = lowered

            if state == "skeleton":
                current_frame = None
            elif state == "triangles":
                current_material = None
                pending_vertices = []
            elif state == "vertexanimation":
                current_target = None

            continue

        if state is None:
            continue

        if lowered == "end":
            if state == "skeleton" and current_frame is not None:
                model.frames.append(current_frame)
                current_frame = None

            if state == "vertexanimation" and current_target is not None:
                model.vertex_targets.append(current_target)
                current_target = None

            state = None
            current_material = None
            pending_vertices = []
            continue

        if state == "nodes":
            match = NODE_PATTERN.match(line)

            if match:
                bone = SmdBone(
                    bone_id=int(match.group(1)),
                    name=match.group(2),
                    parent_id=int(match.group(3)),
                )
            else:
                parts = line.split()

                if len(parts) < 3:
                    continue

                try:
                    bone_id = int(parts[0])
                    parent_id = int(parts[-1])
                    name = " ".join(parts[1:-1]).strip('"')
                except ValueError:
                    continue

                bone = SmdBone(
                    bone_id=bone_id,
                    name=name,
                    parent_id=parent_id,
                )

            model.bones.append(bone)

        elif state == "skeleton":
            if lowered.startswith("time"):
                if current_frame is not None:
                    model.frames.append(current_frame)

                time_value = _parse_time(line.split())
                current_frame = SmdFrame(time=time_value, transforms={})

                if comment:
                    model.frame_names[time_value] = comment

            else:
                parts = line.split()

                if len(parts) < 7:
                    continue

                try:
                    bone_id = int(parts[0])

                    position = (
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )

                    rotation = (
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                    )

                except ValueError:
                    continue

                if current_frame is None:
                    current_frame = SmdFrame(time=0, transforms={})

                current_frame.transforms[bone_id] = (position, rotation)

        elif state == "triangles":
            if current_material is None:
                current_material = line.strip().strip('"')
                model.materials.add(current_material)
                pending_vertices = []

            else:
                vertex = _parse_vertex_line(line)

                if vertex is None:
                    current_material = None
                    pending_vertices = []
                    continue

                pending_vertices.append(vertex)

                if len(pending_vertices) == 3:
                    base_index = len(model.vertices)

                    model.vertices.extend(pending_vertices)

                    model.triangles.append(
                        SmdTriangle(
                            material=current_material,
                            indices=(base_index, base_index + 1, base_index + 2),
                        )
                    )

                    for vertex in pending_vertices:
                        for i in range(3):
                            if vertex.position[i] < min_bound[i]:
                                min_bound[i] = vertex.position[i]

                            if vertex.position[i] > max_bound[i]:
                                max_bound[i] = vertex.position[i]

                    pending_vertices = []
                    current_material = None

        elif state == "vertexanimation":
            if lowered.startswith("time"):
                if current_target is not None:
                    model.vertex_targets.append(current_target)

                time_value = _parse_time(line.split())

                name = comment
                if not name:
                    name = model.frame_names.get(time_value, "")

                current_target = VertexTarget(
                    time=time_value,
                    overrides={},
                    name=name,
                )

            else:
                if current_target is None:
                    current_target = VertexTarget(time=0, overrides={})

                parsed = _parse_vertex_animation_line(line)

                if parsed is None:
                    continue

                vertex_id, position, normal = parsed
                current_target.overrides[vertex_id] = (position, normal)

    if state == "skeleton" and current_frame is not None:
        model.frames.append(current_frame)

    if state == "vertexanimation" and current_target is not None:
        model.vertex_targets.append(current_target)

    # Fill in names from skeleton time comments if the vertexanimation
    # time lines did not carry their own comments.
    for target in model.vertex_targets:
        if not target.name:
            target.name = model.frame_names.get(target.time, "")

    if model.triangles:
        model.has_geometry = True
        model.min_bound = (min_bound[0], min_bound[1], min_bound[2])
        model.max_bound = (max_bound[0], max_bound[1], max_bound[2])

    model.has_animation = bool(model.vertex_targets)

    return model