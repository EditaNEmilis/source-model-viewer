import colorsys
import os
import zlib
import numpy as np
from OpenGL.GL import (
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_GENERATE_MIPMAP,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_LINES,
    GL_MODELVIEW,
    GL_MODULATE,
    GL_NORMALIZE,
    GL_POSITION,
    GL_PROJECTION,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindTexture,
    glClear,
    glClearColor,
    glColor3f,
    glColorMaterial,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glGenerateMipmap,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMatrixMode,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glTexImage2D,
    glTexParameteri,
    glTexEnvi,
    glTranslatef,
    glVertex3f,
    glViewport,
)
from OpenGL.GLU import gluPerspective
from viewer.camera import Camera
from viewer.mesh_buffers import MeshBuffers
from viewer.pose import VertexAnimator
from viewer.skeleton import SkeletonRig, SkeletalAnimation
from viewer.skinning import Skinning
from viewer.vtf_parser import parse_vtf, VtfError


class Renderer:
    def __init__(self):
        self.background_color = (0.10, 0.10, 0.11, 1.0)
        self.width = 1
        self.height = 1

        self.camera = Camera()
        self.animator = VertexAnimator()
        self.skinning = Skinning()
        self.mesh = MeshBuffers()
        self.model = None
        self.rig = None
        self.material_colors = {}
        self.model_position = [0.0, 0.0, 0.0]
        self.model_rotation = [0.0, 0.0, 0.0]
        self.model_center = [0.0, 0.0, 0.0]

        self._static_positions = None
        self._static_normals = None
        self._gpu_positions = None
        self._gpu_normals = None
        self._buffers_dirty = False
        self.backface_culling = False

        self.skeletal_animation_source = None
        self.skeletal_animation = None
        self.driver_bone_name_hint = ""
        self.driver_bone_ref_id = -1
        self.driver_track_id = -1
        self.driver_enabled_value = False
        self.driver_progress_axis_override = None
        self.driver_progress_axis_value = "y"
        self.proximity_skin = False

        self._animation_clips = []
        self._current_clip_index = -1
        self._current_clip_name = ""
        self._clip_metadata = {}

        self.texture_cache = {}
        self.material_dirs = []
        self._material_batches = []

    def initialize(self):
        glClearColor(*self.background_color)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_NORMALIZE)
        glDisable(GL_CULL_FACE)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.25, 0.25, 0.25, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.85, 0.85, 0.85, 1.0))

    def resize(self, width, height):
        self.width = max(1, width)
        self.height = max(1, height)
        glViewport(0, 0, self.width, self.height)

    def paint(self):
        if self.width <= 0 or self.height <= 0:
            return
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width / float(self.height)
        gluPerspective(self.camera.fov, aspect, self.camera.near, self.camera.far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 0.4, 1.0, 0.0))
        self.camera.apply()
        self._draw_grid()
        self._draw_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def set_model(self, model):
        self.model = model
        self.reset_model_transform()
        self._static_positions = None
        self._static_normals = None
        self.rig = None
        self.mesh.release()
        self._unload_textures()
        self._material_batches = []
        vertex_weights = []

        if model and model.has_geometry:
            self._static_positions = np.array(
                [vertex.position for vertex in model.vertices], dtype=np.float32
            )
            self._static_normals = np.array(
                [vertex.normal for vertex in model.vertices], dtype=np.float32
            )
            texcoords = np.array(
                [vertex.uv for vertex in model.vertices], dtype=np.float32
            )
            texcoords[:, 1] = 1.0 - texcoords[:, 1]

            self.model_center = [
                (model.min_bound[0] + model.max_bound[0]) * 0.5,
                (model.min_bound[1] + model.max_bound[1]) * 0.5,
                (model.min_bound[2] + model.max_bound[2]) * 0.5,
            ]
            self.reset_camera()

            colors = self._build_colors()
            indices = self._build_material_batches()
            if indices is None or len(indices) == 0:
                indices = self._build_indices()

            if len(indices) > 0:
                self.mesh.build(
                    self._static_positions,
                    self._static_normals,
                    colors,
                    indices,
                    texcoords=texcoords,
                )
        else:
            self.model_center = [0.0, 0.0, 0.0]
            self.camera.reset()

        if model and model.bones:
            reference_transforms = {}
            if model.frames:
                reference_transforms = model.frames[0].transforms
            self.rig = SkeletonRig(model.bones, reference_transforms)
            vertex_weights = self._build_vertex_weights(model)
            self.skinning.set_rig(self.rig, vertex_weights)

        self.rebuild_animation_frames()
        self._rebuild_skeletal()

        self._gpu_positions = self._static_positions
        self._gpu_normals = self._static_normals
        self._buffers_dirty = False

        self.set_skeletal_frame(self.skinning.frame)

    def set_animation_targets(self, targets):
        self.animator.targets = targets if targets else []
        self.rebuild_animation_frames()
        if self.skeletal_animation is not None:
            if self.has_driver_bone() and self.animator.frame_count() > 0:
                self.driver_enabled_value = True
            self.set_skeletal_frame(self.skinning.frame)

    # ------------------------------------------------------------------
    # Texture / material management
    # ------------------------------------------------------------------

    def add_material_directory(self, dir_path):
        dir_path = os.path.normpath(dir_path)
        if dir_path not in self.material_dirs:
            self.material_dirs.append(dir_path)
            self._unload_textures()

    def _unload_textures(self):
        for tex_id in self.texture_cache.values():
            if tex_id:
                glDeleteTextures(1, [tex_id])
        self.texture_cache = {}

    def _find_vtf_file(self, material_name):
        name = material_name.strip().strip("/")

        dot = name.rfind(".")
        if dot > 0:
            ext = name[dot + 1:].lower()
            if ext in ("vtf", "vmt", "bmp", "tga", "png", "jpg", "jpeg"):
                name = name[:dot]

        name = name.replace("\\", "/")

        candidates = [name + ".vtf"]
        if name.lower().startswith("materials/"):
            candidates.append(name[10:] + ".vtf")
        else:
            candidates.append("materials/" + name + ".vtf")

        for search_dir in self.material_dirs:
            for candidate in candidates:
                full_path = os.path.join(search_dir, candidate)
                if os.path.isfile(full_path):
                    return full_path

        # Recursive fallback for bare material names
        target_filename = os.path.basename(name).lower() + ".vtf"
        for search_dir in self.material_dirs:
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.lower() == target_filename:
                        return os.path.join(root, f)

        return None

    def _load_material_texture(self, material_name):
        if material_name in self.texture_cache:
            return self.texture_cache[material_name]

        vtf_path = self._find_vtf_file(material_name)

        if vtf_path is None:
            self.texture_cache[material_name] = None
            return None

        try:
            info, rgba = parse_vtf(vtf_path)
        except (VtfError, OSError) as e:
            self.texture_cache[material_name] = None
            return None

        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

        height, width = rgba.shape[0], rgba.shape[1]
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, rgba.tobytes()
        )

        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)
        self.texture_cache[material_name] = tex_id
        return tex_id

    def _build_material_batches(self):
        self._material_batches = []
        if self.model is None or not self.model.triangles:
            return None

        batches = {}
        batch_order = []
        for triangle in self.model.triangles:
            mat = triangle.material
            if mat not in batches:
                batches[mat] = []
                batch_order.append(mat)
            batches[mat].extend(triangle.indices)

        all_indices = []
        for mat in batch_order:
            start = len(all_indices)
            all_indices.extend(batches[mat])
            self._material_batches.append((mat, start, len(batches[mat])))

        return np.array(all_indices, dtype=np.uint32)

    # ------------------------------------------------------------------
    # Animation modes and controls
    # ------------------------------------------------------------------

    def set_animation_mode(self, mode):
        self.animator.set_mode(mode)
        self._refresh_pose()

    def set_match_ids_by_position(self, enabled):
        self.animator.set_match_ids(enabled)
        self.rebuild_animation_frames()

    def set_progress(self, progress):
        self.animator.set_progress(progress)
        self._refresh_pose()

    def set_intensity(self, intensity):
        self.animator.set_intensity(intensity)
        self._refresh_pose()

    # ------------------------------------------------------------------
    # Skeletal animation
    # ------------------------------------------------------------------

    def set_skeletal_animation_model(self, model):
        self.skeletal_animation_source = model
        self._rebuild_skeletal()
        self.set_skeletal_frame(0.0)

    def clear_skeletal_animation(self):
        self.skeletal_animation_source = None
        self.skeletal_animation = None
        self.skinning.set_animation(None)
        self.skinning.enabled = False
        self.skinning.skin_matrices = None
        self.driver_enabled_value = False
        self.driver_bone_ref_id = -1
        self.driver_track_id = -1
        self._refresh_pose()

    def set_skeletal_enabled(self, enabled):
        self.skinning.enabled = bool(
            enabled
            and self.skeletal_animation is not None
            and self.rig is not None
        )
        self.skinning.set_frame(self.skinning.frame)
        self._refresh_pose()

    def set_driver_enabled(self, enabled):
        self.driver_enabled_value = bool(
            enabled
            and self.skeletal_animation is not None
            and self.has_driver_bone()
        )
        self.set_skeletal_frame(self.skinning.frame)

    def set_driver_bone_name(self, name):
        self.driver_bone_name_hint = name if name else ""
        self._refresh_driver_bone()
        if self.skeletal_animation is not None:
            self.set_skeletal_frame(self.skinning.frame)

    def set_driver_progress_axis(self, axis):
        if axis in ("y", "z"):
            self.driver_progress_axis_override = axis
        elif axis == "auto":
            self.driver_progress_axis_override = None
        self._refresh_driver_bone()
        if self.skeletal_animation is not None:
            self.set_skeletal_frame(self.skinning.frame)

    def set_proximity_skin(self, enabled):
        self.proximity_skin = bool(enabled)
        self._rebuild_weights()
        self._refresh_pose()

    def _rebuild_weights(self):
        if self.model is None or self.rig is None:
            return
        weights = self._build_vertex_weights(self.model)
        self.skinning.set_rig(self.rig, weights)

    def _build_vertex_weights(self, model):
        parsed = []
        for vertex in model.vertices:
            vertex_weights = []
            if vertex.links:
                total = 0.0
                for bone_id, weight in vertex.links:
                    vertex_weights.append((bone_id, weight))
                    total += weight
                remaining = 1.0 - total
                if remaining > 0.000001:
                    vertex_weights.append((vertex.parent_bone, remaining))
            else:
                vertex_weights.append((vertex.parent_bone, 1.0))
            parsed.append(vertex_weights)
        if self.proximity_skin and self.rig is not None and self._weights_degenerate(parsed):
            return self._proximity_weights(model)
        return parsed

    def _weights_degenerate(self, parsed):
        if not parsed or self.rig is None:
            return False
        from collections import Counter
        dominant = Counter()
        for vertex_weights in parsed:
            if not vertex_weights:
                continue
            bone_id = max(vertex_weights, key=lambda entry: entry[1])[0]
            dominant[bone_id] += 1
        used = sum(1 for count in dominant.values() if count > 0)
        total = len(parsed)
        top = max(dominant.values()) if dominant else 0
        if used < max(2, len(self.rig.bone_ids) // 2):
            return True
        if total > 0 and top > 0.7 * total:
            return True
        return False

    def _proximity_weights(self, model):
        origins = {}
        for bone_id in self.rig.bone_ids:
            matrix = self.rig.bind_world.get(bone_id)
            if matrix is None:
                origins[bone_id] = (0.0, 0.0, 0.0)
            else:
                origins[bone_id] = (matrix[3], matrix[7], matrix[11])
        bone_ids = list(origins.keys())
        if not bone_ids:
            return [[(0, 1.0)] for _ in model.vertices]
        origin_array = np.array(
            [origins[bone_id] for bone_id in bone_ids],
            dtype=np.float32,
        )
        weights = []
        for vertex in model.vertices:
            point = np.array(vertex.position, dtype=np.float32)
            delta = origin_array - point
            distance_sq = (delta * delta).sum(axis=1)
            best = int(distance_sq.argmin())
            weights.append([(bone_ids[best], 1.0)])
        return weights

    def skeletal_weight_stats(self):
        skinning = self.skinning
        if skinning.weight_bone_ids is None or self.rig is None:
            return None
        ids = skinning.weight_bone_ids
        vals = skinning.weight_values
        if vals.shape[1] > 1:
            dominant = ids[np.arange(ids.shape[0]), np.argmax(vals, axis=1)]
        else:
            dominant = ids[:, 0]
        inverse = {index: bone_id for bone_id, index in skinning.bone_index_map.items()}
        per_bone = {}
        unbound = 0
        for bone_index in np.unique(dominant):
            bone_index = int(bone_index)
            count = int((dominant == bone_index).sum())
            if bone_index == 0:
                unbound = count
            else:
                per_bone[inverse.get(bone_index, -1)] = count
        top = sorted(per_bone.items(), key=lambda item: -item[1])[:6]
        top_named = [
            (self.rig.id_to_name.get(bone_id, str(bone_id)), count)
            for bone_id, count in top
        ]
        return {
            "vertices": int(ids.shape[0]),
            "bones_with_vertices": len(per_bone),
            "rig_bones": len(self.rig.bone_ids),
            "unbound_vertices": unbound,
            "top_bones": top_named,
        }

    def skeletal_rig_bone_count(self):
        if self.rig is None:
            return 0
        return len(self.rig.bone_ids)

    def skeletal_anim_bone_count(self):
        if self.skeletal_animation is None:
            return 0
        return self.skeletal_animation.anim_bone_count

    def skeletal_mapped_count(self):
        if self.skeletal_animation is None:
            return 0
        return self.skeletal_animation.name_mapped_count

    def skeletal_driven_count(self):
        if self.skeletal_animation is None:
            return 0
        return len(self.skeletal_animation.driven_ref_ids)

    def skeletal_unmapped_names(self):
        if self.skeletal_animation is None:
            return []
        return list(self.skeletal_animation.unmapped_names.values())

    def skeletal_enabled(self):
        return self.skinning.enabled

    def driver_enabled(self):
        return self.driver_enabled_value

    def has_skeletal_animation(self):
        return self.skeletal_animation is not None

    def has_skeletal_source(self):
        return self.skeletal_animation_source is not None

    def has_rig(self):
        return self.rig is not None

    def has_driver_bone(self):
        return self.driver_bone_ref_id != -1 or self.driver_track_id != -1

    def skeletal_frame_count(self):
        if self.skeletal_animation is None:
            return 0
        return self.skeletal_animation.frame_count

    def skeletal_current_frame(self):
        return self.skinning.frame

    def set_backface_culling(self, enabled):
        self.backface_culling = bool(enabled)

    def set_skeletal_frame(self, frame):
        count = self.skeletal_frame_count()
        if count == 0:
            self.skinning.set_frame(0.0)
            self._refresh_pose()
            return
        frame = max(0.0, min(float(frame), count - 1))
        if self.driver_enabled_value and self.has_driver_bone() and self.skeletal_animation:
            local_transforms = self.skeletal_animation.sample(frame)
            driver_transform = None
            if self.driver_bone_ref_id != -1:
                driver_transform = local_transforms.get(self.driver_bone_ref_id)
            elif self.driver_track_id != -1:
                unmapped = self.skeletal_animation.sample_unmapped(frame)
                driver_transform = unmapped.get(self.driver_track_id)
            if driver_transform is not None:
                driver_position = driver_transform[0]
                self.animator.set_intensity(driver_position[0])
                if self.driver_progress_axis_value == "z":
                    progress_value = driver_position[2]
                else:
                    progress_value = driver_position[1]
                self.animator.set_progress(progress_value)
        self.skinning.set_frame(frame)
        self._refresh_pose()

    def driver_bone_name(self):
        if self.driver_bone_ref_id != -1 and self.rig is not None:
            return self.rig.id_to_name.get(self.driver_bone_ref_id, "")
        if self.driver_track_id != -1 and self.skeletal_animation is not None:
            return self.skeletal_animation.unmapped_names.get(self.driver_track_id, "")
        return ""

    def driver_progress_axis(self):
        return self.driver_progress_axis_value

    # ------------------------------------------------------------------
    # Vertex animation queries
    # ------------------------------------------------------------------

    def animation_mode(self):
        return self.animator.mode

    def match_ids_by_position(self):
        return self.animator.match_ids

    def animation_ignored_count(self):
        return self.animator.ignored_count

    def animation_max_vertex_id(self):
        return self.animator.max_vertex_id

    def animation_reference_vertex_count(self):
        if self.model and self.model.has_geometry:
            return len(self.model.vertices)
        return 0

    def animation_first_target_count(self):
        return self.animator.first_target_count

    def animation_matched_count(self):
        return self.animator.matched_count

    def animation_matched_vertex_count(self):
        return self.animator.matched_vertex_count

    def progress(self):
        return self.animator.progress

    def intensity(self):
        return self.animator.intensity

    def rebuild_animation_frames(self):
        self.animator.build_frames(
            self.animator.targets,
            self._static_positions,
            self._static_normals,
            self._model_bounds(),
        )
        self.animator.set_progress(self.animator.progress)
        self._refresh_pose()

    def frame_count(self):
        return self.animator.frame_count()

    def current_frame(self):
        return self.animator.current_frame

    def set_frame(self, frame):
        self.animator.set_frame(frame)
        self._refresh_pose()

    def animation_name(self, index=None):
        return self.animator.animation_name(index)

    # ------------------------------------------------------------------
    # Camera and view
    # ------------------------------------------------------------------

    def reset_view(self):
        self.reset_camera()

    def reset_camera(self):
        min_bound, max_bound = self._display_bounds()
        if min_bound and max_bound:
            self.camera.fit(min_bound, max_bound)
        else:
            self.camera.reset()

    def reset_model_transform(self):
        self.model_position = [0.0, 0.0, 0.0]
        self.model_rotation = [0.0, 0.0, 0.0]

    def reset_all(self):
        self.reset_model_transform()
        self.reset_camera()
        self.set_progress(0.0)
        self.set_intensity(1.0)
        self.set_skeletal_frame(0.0)

    def move_model(self, dx, dy):
        if not self.model or not self.model.has_geometry:
            return
        right, up = self.camera.pan_vectors()
        scale = max(0.0005, self.camera.distance * self.camera.pan_speed)
        self.model_position[0] += right[0] * dx * scale - up[0] * dy * scale
        self.model_position[1] += right[1] * dx * scale - up[1] * dy * scale
        self.model_position[2] += right[2] * dx * scale - up[2] * dy * scale

    # ------------------------------------------------------------------
    # DMX animation clips
    # ------------------------------------------------------------------

    def set_animation_clips(self, clips):
        self._animation_clips = clips if clips else []
        self._clip_metadata = {name: model.metadata for name, model in clips}
        self._current_clip_name = ""
        if self._animation_clips:
            self._current_clip_index = 0
            self._current_clip_name = self._animation_clips[0][0]
            self.set_skeletal_animation_model(self._animation_clips[0][1])
        else:
            self.clear_skeletal_animation()

    def set_animation_clip(self, index):
        if 0 <= index < len(self._animation_clips):
            name, model = self._animation_clips[index]
            self._current_clip_index = index
            self._current_clip_name = name
            self.set_skeletal_animation_model(model)
            self.set_skeletal_frame(0.0)

    def clip_names(self):
        return [name for name, _ in self._animation_clips]

    def current_clip_name(self):
        return self._current_clip_name

    def clip_metadata(self, index):
        if 0 <= index < len(self._animation_clips):
            name, _ = self._animation_clips[index]
            return self._clip_metadata.get(name, {})
        return {}

    def set_skeletal_progress(self, progress):
        count = self.skeletal_frame_count()
        if count > 0:
            frame = progress * (count - 1)
            self.set_skeletal_frame(frame)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_bounds(self):
        if self.model and self.model.has_geometry:
            return (self.model.min_bound, self.model.max_bound)
        return None

    def _build_colors(self):
        vertex_count = len(self.model.vertices)
        colors = np.empty((vertex_count, 3), dtype=np.float32)
        colors[:] = (0.72, 0.72, 0.74)
        for triangle in self.model.triangles:
            color = self._material_color(triangle.material)
            for index in triangle.indices:
                if 0 <= index < vertex_count:
                    colors[index] = color
        return colors

    def _build_indices(self):
        vertex_count = len(self.model.vertices)
        indices = []
        for triangle in self.model.triangles:
            a, b, c = triangle.indices
            if 0 <= a < vertex_count and 0 <= b < vertex_count and 0 <= c < vertex_count:
                indices.extend((a, b, c))
        return np.array(indices, dtype=np.uint32)

    def _rebuild_skeletal(self):
        self.skeletal_animation = None
        self.driver_bone_ref_id = -1
        self.driver_track_id = -1
        if (
            self.skeletal_animation_source is not None
            and self.rig is not None
            and self.skeletal_animation_source.frames
        ):
            self.skeletal_animation = SkeletalAnimation(
                self.skeletal_animation_source,
                self.rig,
            )
            self._refresh_driver_bone()
            self.skinning.enabled = self.skeletal_animation.frame_count > 0
            self.driver_enabled_value = (
                self.has_driver_bone() and self.animator.frame_count() > 0
            )
        else:
            self.skinning.enabled = False
            self.driver_enabled_value = False
        self.skinning.set_animation(self.skeletal_animation)

    def _refresh_driver_bone(self):
        self.driver_bone_ref_id = -1
        self.driver_track_id = -1
        self.driver_progress_axis_value = self.driver_progress_axis_override or "y"
        if self.rig is not None:
            lower_name_to_id = {
                name.lower(): bone_id
                for name, bone_id in self.rig.name_to_id.items()
            }
            if self.driver_bone_name_hint:
                hint = self.driver_bone_name_hint.lower()
                if hint in lower_name_to_id:
                    self.driver_bone_ref_id = lower_name_to_id[hint]
            if self.driver_bone_ref_id == -1:
                for name, bone_id in lower_name_to_id.items():
                    if (
                        "vertexanimdriver" in name
                        or "vca_driver" in name
                        or "flexdriver" in name
                        or "driver" in name
                    ):
                        self.driver_bone_ref_id = bone_id
                        break
        if self.driver_bone_ref_id == -1 and self.skeletal_animation is not None:
            lower_unmapped = {
                name.lower(): bone_id
                for bone_id, name in self.skeletal_animation.unmapped_names.items()
            }
            if self.driver_bone_name_hint:
                hint = self.driver_bone_name_hint.lower()
                if hint in lower_unmapped:
                    self.driver_track_id = lower_unmapped[hint]
            if self.driver_track_id == -1:
                for name, bone_id in lower_unmapped.items():
                    if (
                        "vertexanimdriver" in name
                        or "vca_driver" in name
                        or "flexdriver" in name
                        or "driver" in name
                    ):
                        self.driver_track_id = bone_id
                        break
        if self.driver_progress_axis_override:
            self.driver_progress_axis_value = self.driver_progress_axis_override
        else:
            self.driver_progress_axis_value = self._detect_driver_progress_axis()

    def _detect_driver_progress_axis(self):
        if (
            self.skeletal_animation is None
            or self.skeletal_animation.frame_count == 0
            or not self.has_driver_bone()
        ):
            return "y"
        min_y = float("inf")
        max_y = float("-inf")
        min_z = float("inf")
        max_z = float("-inf")
        for index in range(self.skeletal_animation.frame_count):
            if self.driver_bone_ref_id != -1:
                transform = self.skeletal_animation.full_locals[index].get(
                    self.driver_bone_ref_id
                )
            else:
                transform = self.skeletal_animation.extra_locals[index].get(
                    self.driver_track_id
                )
            if transform is None:
                continue
            position = transform[0]
            min_y = min(min_y, position[1])
            max_y = max(max_y, position[1])
            min_z = min(min_z, position[2])
            max_z = max(max_z, position[2])
        range_y = max_y - min_y
        range_z = max_z - min_z
        if range_z > range_y + 0.000001:
            return "z"
        return "y"

    def _refresh_pose(self):
        if self._static_positions is None:
            self._gpu_positions = None
            self._gpu_normals = None
            self._buffers_dirty = False
            return
        positions, normals = self.animator.compute_pose(
            self._static_positions,
            self._static_normals,
        )
        if self.skinning.enabled and self.skinning.animation is not None:
            positions, normals = self.skinning.apply(positions, normals)
        self._gpu_positions = positions
        self._gpu_normals = normals
        self._buffers_dirty = True

    def _display_bounds(self):
        if not self.model or not self.model.has_geometry:
            return None, None
        min_bound = self.model.min_bound
        max_bound = self.model.max_bound
        offset = self.model_position
        return (
            (
                min_bound[0] + offset[0],
                min_bound[1] + offset[1],
                min_bound[2] + offset[2],
            ),
            (
                max_bound[0] + offset[0],
                max_bound[1] + offset[1],
                max_bound[2] + offset[2],
            ),
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_grid(self):
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        base_x = self.model_center[0]
        base_y = self.model_center[1]
        base_z = self.model_center[2]
        extent = max(10.0, self.camera.radius * 2.0)
        step = extent / 10.0
        glBegin(GL_LINES)
        glColor3f(0.22, 0.22, 0.24)
        for i in range(-10, 11):
            offset = i * step
            glVertex3f(base_x + offset, base_y - extent, base_z)
            glVertex3f(base_x + offset, base_y + extent, base_z)
            glVertex3f(base_x - extent, base_y + offset, base_z)
            glVertex3f(base_x + extent, base_y + offset, base_z)
        glColor3f(0.75, 0.25, 0.25)
        glVertex3f(base_x, base_y, base_z)
        glVertex3f(base_x + extent, base_y, base_z)
        glColor3f(0.25, 0.75, 0.25)
        glVertex3f(base_x, base_y, base_z)
        glVertex3f(base_x, base_y + extent, base_z)
        glColor3f(0.30, 0.45, 0.95)
        glVertex3f(base_x, base_y, base_z)
        glVertex3f(base_x, base_y, base_z + extent)
        glEnd()

    def _draw_model(self):
        if (
            not self.model
            or not self.model.has_geometry
            or not self.mesh.is_valid()
        ):
            return

        if self._buffers_dirty:
            self.mesh.update(self._gpu_positions, self._gpu_normals)
            self._buffers_dirty = False

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        if self.backface_culling:
            glEnable(GL_CULL_FACE)
        else:
            glDisable(GL_CULL_FACE)

        glPushMatrix()
        glTranslatef(
            self.model_position[0],
            self.model_position[1],
            self.model_position[2],
        )
        glRotatef(self.model_rotation[0], 1.0, 0.0, 0.0)
        glRotatef(self.model_rotation[1], 0.0, 1.0, 0.0)
        glRotatef(self.model_rotation[2], 0.0, 0.0, 1.0)

        if self._material_batches and self.mesh.has_texcoords:
            for mat_name, offset, count in self._material_batches:
                tex_id = self._load_material_texture(mat_name)
                if tex_id:
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, tex_id)
                    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                    self.mesh.draw_range(offset, count, with_texcoords=True, with_colors=False)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    glDisable(GL_TEXTURE_2D)
                else:
                    self.mesh.draw_range(offset, count, with_texcoords=False, with_colors=True)
        else:
            self.mesh.draw(with_texcoords=False)

        glPopMatrix()
        glDisable(GL_LIGHTING)

    def _material_color(self, material):
        if not material:
            return (0.72, 0.72, 0.74)
        if material in self.material_colors:
            return self.material_colors[material]
        checksum = zlib.crc32(material.encode("utf-8", errors="ignore")) & 0xFFFFFFFF
        hue = (checksum % 1024) / 1024.0
        color = colorsys.hsv_to_rgb(hue, 0.30, 0.82)
        self.material_colors[material] = color
        return color