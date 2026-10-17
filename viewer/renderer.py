import colorsys
import os
import zlib
import numpy as np
from OpenGL.GL import (
    GL_ALPHA_TEST,
    GL_AMBIENT,
    GL_AMBIENT_AND_DIFFUSE,
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_BUFFER_BIT,
    GL_COLOR_MATERIAL,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT_AND_BACK,
    GL_GREATER,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE,
    GL_FILL,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_LINES,
    GL_POINTS,
    GL_MODELVIEW,
    GL_MODULATE,
    GL_NORMALIZE,
    GL_NUM_EXTENSIONS,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_POSITION,
    GL_PROJECTION,
    GL_RGBA,
    GL_REPEAT,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE2,
    GL_TEXTURE_ENV,
    GL_TEXTURE_ENV_MODE,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MAX_LEVEL,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    GL_COMPILE_STATUS,
    GL_LINK_STATUS,
    glAlphaFunc,
    glActiveTexture,
    glAttachShader,
    glBegin,
    glBindTexture,
    glBlendFunc,
    glClear,
    glClearColor,
    glClientActiveTexture,
    glColor3f,
    glColor4f,
    glColorMask,
    glColorMaterial,
    glCompileShader,
    glCompressedTexImage2D,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glDeleteTextures,
    glDisable,
    glEnable,
    glEnd,
    glGenTextures,
    glGenerateMipmap,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetString,
    glGetIntegerv,
    glGetStringi,
    glGetUniformLocation,
    glLightfv,
    glLineWidth,
    glLinkProgram,
    glLoadIdentity,
    glMatrixMode,
    glPointSize,
    glPolygonMode,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glShaderSource,
    glTexImage2D,
    glTexParameteri,
    glTexEnvi,
    glTranslatef,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUseProgram,
    glVertex3f,
    glViewport,
)
from OpenGL.GL.ARB.texture_compression_bptc import GL_COMPRESSED_RGBA_BPTC_UNORM_ARB
from OpenGL.GL.ARB.texture_compression_rgtc import (
    GL_COMPRESSED_RED_RGTC1,
    GL_COMPRESSED_RG_RGTC2,
)
from OpenGL.GL.EXT.texture_compression_s3tc import (
    GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
    GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
)
from OpenGL.GLU import gluPerspective
from viewer.camera import Camera
from viewer.mesh_buffers import MeshBuffers
from viewer.pose import VertexAnimator
from viewer.skeleton import SkeletonRig, SkeletalAnimation, evaluate_world_matrices
from viewer.skinning import Skinning
from viewer.vmat_parser import (
    VmatError,
    parse_vmat_c,
    pick_color_texture,
    pick_normal_texture,
    pick_occlusion_texture,
)
from viewer.vtex_parser import VtexError, parse_vtex_c
from viewer.vtf_parser import parse_vtf, VtfError
from viewer.vmt_parser import parse_vmt

# BC compression kind -> (OpenGL internal format, required extension).
_COMPRESSED_FORMATS = {
    "dxt1": (GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, "GL_EXT_texture_compression_s3tc"),
    "dxt5": (GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, "GL_EXT_texture_compression_s3tc"),
    "bc4": (GL_COMPRESSED_RED_RGTC1, "GL_ARB_texture_compression_rgtc"),
    "bc5": (GL_COMPRESSED_RG_RGTC2, "GL_ARB_texture_compression_rgtc"),
    "bc7": (GL_COMPRESSED_RGBA_BPTC_UNORM_ARB, "GL_ARB_texture_compression_bptc"),
}

_GL_EXTENSION_FALLBACKS = {
    "GL_ARB_texture_compression_rgtc": "GL_EXT_texture_compression_rgtc",
}

# Fixed-function style lighting for Source 2 materials with a normal map.
_MODEL_VERTEX_SHADER = """
#version 120
varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vViewPos;

void main() {
    vUv = gl_MultiTexCoord0.xy;
    vNormal = normalize(gl_NormalMatrix * gl_Normal);
    vec4 viewPos = gl_ModelViewMatrix * gl_Vertex;
    vViewPos = viewPos.xyz;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
"""

_MODEL_FRAGMENT_SHADER = """
#version 120
uniform sampler2D colorMap;
uniform sampler2D normalMap;
uniform sampler2D aoMap;
uniform vec3 tint;
uniform vec3 lightDir;
uniform float ambientStrength;
uniform float diffuseStrength;
uniform float alphaThreshold;
uniform float aoStrength;
uniform int hasNormal;
uniform int hasAo;
uniform int aoChannel;
uniform int unlit;
uniform int alphaTest;

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vViewPos;

void main() {
    vec4 base = texture2D(colorMap, vUv);
    if (alphaTest == 1 && base.a < alphaThreshold) {
        discard;
    }

    vec3 n = normalize(vNormal);
    if (hasNormal == 1) {
        vec2 nxy = texture2D(normalMap, vUv).rg * 2.0 - 1.0;
        vec3 dp1 = dFdx(vViewPos);
        vec3 dp2 = dFdy(vViewPos);
        vec2 duv1 = dFdx(vUv);
        vec2 duv2 = dFdy(vUv);
        vec3 dp2perp = cross(dp2, n);
        vec3 dp1perp = cross(n, dp1);
        vec3 tangent = dp2perp * duv1.x + dp1perp * duv2.x;
        vec3 bitangent = dp2perp * duv1.y + dp1perp * duv2.y;
        float invmax = inversesqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-8));
        vec3 mapped = vec3(nxy, sqrt(max(0.0, 1.0 - dot(nxy, nxy))));
        n = normalize(mat3(tangent * invmax, bitangent * invmax, n) * mapped);
    }

    vec3 lit;
    if (unlit == 1) {
        lit = base.rgb;
    } else {
        float ao = 1.0;
        if (hasAo == 1) {
            vec4 maskSample = texture2D(aoMap, vUv);
            float occlusion = (aoChannel == 1) ? maskSample.g : maskSample.r;
            ao = mix(1.0, occlusion, aoStrength);
        }
        float ndl = max(dot(n, normalize(lightDir)), 0.0);
        lit = base.rgb * (ambientStrength * ao + diffuseStrength * ndl);
    }

    gl_FragColor = vec4(lit * tint, base.a);
}
"""

_LIGHT_DIR = (0.4, 0.4, 1.0)
_MODEL_UNIFORMS = (
    "colorMap", "normalMap", "aoMap", "tint", "lightDir",
    "ambientStrength", "diffuseStrength", "alphaThreshold", "aoStrength",
    "hasNormal", "hasAo", "aoChannel", "unlit", "alphaTest",
)


def _vtex_wrap_modes(flags):
    """Map VTexFlags clamp suggestions to GL wrap modes."""
    wrap_s = GL_CLAMP_TO_EDGE if flags & 0x1 else GL_REPEAT
    wrap_t = GL_CLAMP_TO_EDGE if flags & 0x2 else GL_REPEAT
    return wrap_s, wrap_t


def _compile_shader(stage, source):
    shader = glCreateShader(stage)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        raise RuntimeError(log.decode("utf-8", "replace") if isinstance(log, bytes) else str(log))
    return shader


def _compile_program(vertex_source, fragment_source):
    vertex = _compile_shader(GL_VERTEX_SHADER, vertex_source)
    try:
        fragment = _compile_shader(GL_FRAGMENT_SHADER, fragment_source)
    except Exception:
        glDeleteShader(vertex)
        raise

    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)
    glDeleteShader(vertex)
    glDeleteShader(fragment)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program)
        glDeleteProgram(program)
        raise RuntimeError(log.decode("utf-8", "replace") if isinstance(log, bytes) else str(log))
    return program


class Renderer:
    def __init__(self):
        self.background_color = (0.10, 0.10, 0.11, 1.0)
        self.width = 1
        self.height = 1
        self.show_grid = True
        self.show_axes = True
        self.view_mode = "textured"
        self.wireframe_overlay = False
        self.show_skeleton = False

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

        self.texture_cache = {}
        self.material_props = {}
        self.material_maps = {}
        self.material_dirs = []
        self.model_material_dirs = []
        self._material_batches = []
        self._gl_extensions = None
        self._model_shader = None
        self._model_uniforms = {}

    def initialize(self):
        glActiveTexture(GL_TEXTURE0)
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
        # Embedding toolkits may leave any unit active.
        glActiveTexture(GL_TEXTURE0)
        glClientActiveTexture(GL_TEXTURE0)
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width / float(self.height)
        gluPerspective(self.camera.fov, aspect, self.camera.near, self.camera.far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 0.4, 1.0, 0.0))
        self.camera.apply()
        if self.show_grid:
            self._draw_grid()
        if self.show_axes:
            self._draw_axes()
        self._draw_model()
        if self.show_skeleton:
            self._draw_skeleton()

    def set_background_color(self, r, g, b, a=1.0):
        self.background_color = (float(r), float(g), float(b), float(a))

    def set_show_grid(self, enabled):
        self.show_grid = bool(enabled)

    def set_show_axes(self, enabled):
        self.show_axes = bool(enabled)

    def set_model_view_mode(self, mode):
        if mode in ("textured", "solid", "uv_checker"):
            self.view_mode = mode

    def set_wireframe(self, enabled):
        self.wireframe_overlay = bool(enabled)

    def set_show_skeleton(self, enabled):
        self.show_skeleton = bool(enabled)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def set_model(self, model):
        self.model = model
        self.model_material_dirs = []
        if model and hasattr(model, "metadata") and "material_dirs" in model.metadata:
            self.model_material_dirs = [
                d.replace("\\", "/").strip("/")
                for d in model.metadata["material_dirs"]
                if d
            ]

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
            if model.uv_pre_flipped:
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

        self.embedded_textures = dict(getattr(model, "embedded_textures", {}) or {})

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
        for maps in self.material_maps.values():
            for key in ("normal", "ao"):
                tex_id = maps.get(key)
                if tex_id:
                    glDeleteTextures(1, [tex_id])
        self.texture_cache = {}
        self.material_props = {}
        self.material_maps = {}
        checker_id = getattr(self, "_checker_tex", None)
        if checker_id:
            try:
                glDeleteTextures(1, [checker_id])
            except Exception:
                pass
            self._checker_tex = None

    def _resolve_path(self, name, ext):
        if not name:
            return None

        clean_name = name.strip().replace("\\", "/").strip('"/')

        lowered = clean_name.lower()
        for suffix in (
            ".vmat_c", ".vtex_c", ".vmat", ".vtex",
            ".vtf", ".vmt", ".bmp", ".tga", ".png", ".jpg", ".jpeg",
        ):
            if lowered.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break

        if clean_name.lower().startswith("materials/"):
            clean_name = clean_name[10:]

        relative_candidates = []
        for cddir in self.model_material_dirs:
            cddir_clean = cddir.strip("/")
            if cddir_clean:
                relative_candidates.append(cddir_clean + "/" + clean_name + ext)

        relative_candidates.append(clean_name + ext)
        relative_candidates.append("materials/" + clean_name + ext)

        for candidate in relative_candidates:
            for search_dir in self.material_dirs:
                search_clean = os.path.normpath(search_dir)
                full_path = os.path.join(search_clean, candidate)
                if os.path.isfile(full_path):
                    return full_path

                if not candidate.lower().startswith("materials/"):
                    mat_full = os.path.join(search_clean, "materials", candidate)
                    if os.path.isfile(mat_full):
                        return mat_full

        target_name = os.path.basename(clean_name).lower() + ext
        for search_dir in self.material_dirs:
            for cddir in self.model_material_dirs:
                cddir_clean = cddir.strip("/")
                test_dirs = [
                    os.path.join(search_dir, cddir_clean),
                    os.path.join(search_dir, "materials", cddir_clean),
                ]
                for test_dir in test_dirs:
                    if os.path.isdir(test_dir):
                        try:
                            for entry in os.listdir(test_dir):
                                if entry.lower() == target_name:
                                    return os.path.join(test_dir, entry)
                        except OSError:
                            pass

        return None

    def _find_vtf_file(self, material_name):
        # 1. Direct VTF check
        path = self._resolve_path(material_name, ".vtf")
        if path:
            return path

        # 2. Check matching VMT
        vmt_path = self._resolve_path(material_name, ".vmt")
        if vmt_path is None:
            return None

        try:
            shader, params = parse_vmt(vmt_path)
        except Exception:
            return None

        # Check eye iris texture FIRST so pupils/eyes render instead of blank white scleras
        keys_to_check = [
            "$iris", "iris",
            "$basetexture", "basetexture",
            "$basetexture2", "basetexture2",
            "$maintexture", "maintexture",
            "$texture2", "texture2",
        ]

        for key in keys_to_check:
            base = params.get(key)
            if base:
                clean_base = base.replace("\\", "/").strip('"/').lstrip("/")
                resolved = self._resolve_path(clean_base, ".vtf")
                if resolved:
                    return resolved

        return None

    def _parse_vmt_color(self, val_str):
        if not val_str:
            return (1.0, 1.0, 1.0, 1.0)
        clean = val_str.strip('"{}[ ]')
        parts = clean.split()
        if len(parts) >= 3:
            try:
                nums = [float(p) for p in parts[:3]]
                # If values are in 0-255 range, normalize to 0.0-1.0
                if any(n > 1.0 for n in nums):
                    nums = [n / 255.0 for n in nums]
                return (nums[0], nums[1], nums[2], 1.0)
            except ValueError:
                pass
        return (1.0, 1.0, 1.0, 1.0)

    def _load_material_texture(self, material_name):
        if material_name in self.texture_cache:
            return self.texture_cache[material_name]

        props = {
            "nocull": False,
            "nodraw": False,
            "selfillum": False,
            "translucent": False,
            "alphatest": False,
            "color": (1.0, 1.0, 1.0, 1.0),
        }

        # Embedded textures (GoldSrc MDL): decoded at parse time, no
        # filesystem lookup needed.
        embedded = self.embedded_textures.get(material_name)

        if embedded is not None:
            flags = 0
            if self.model is not None:
                flags = (self.model.metadata.get("texture_flags", {}) or {}).get(
                    material_name, 0
                )

            props["selfillum"] = bool(flags & 4)    # STUDIO_NF_FULLBRIGHT
            props["alphatest"] = bool(flags & 64)   # STUDIO_NF_MASKED
            self.material_props[material_name] = props

            tex_id = self._upload_rgba_texture(embedded)
            self.texture_cache[material_name] = tex_id
            return tex_id

        # Source 2 models resolve textures through .vmat_c -> .vtex_c.
        if self._is_source2_model():
            tex_id = self._load_source2_material_texture(material_name, props)
            if tex_id is not None or props["nodraw"]:
                self.material_props[material_name] = props
                self.texture_cache[material_name] = tex_id
                return tex_id

        vmt_path = self._resolve_path(material_name, ".vmt")
        if vmt_path and os.path.isfile(vmt_path):
            try:
                _, params = parse_vmt(vmt_path)
                props["nocull"] = params.get("$nocull") in ("1", "true") or params.get("nocull") in ("1", "true")
                props["nodraw"] = params.get("$no_draw") in ("1", "true") or params.get("no_draw") in ("1", "true")
                props["selfillum"] = params.get("$selfillum") in ("1", "true") or params.get("selfillum") in ("1", "true")
                props["translucent"] = params.get("$translucent") in ("1", "true") or params.get("translucent") in ("1", "true")
                props["alphatest"] = params.get("$alphatest") in ("1", "true") or params.get("alphatest") in ("1", "true")

                color_val = params.get("$color") or params.get("$color2") or params.get("color") or params.get("color2")
                if color_val:
                    props["color"] = self._parse_vmt_color(color_val)
            except Exception:
                pass

        self.material_props[material_name] = props

        if props["nodraw"]:
            self.texture_cache[material_name] = None
            return None

        vtf_path = self._find_vtf_file(material_name)
        if vtf_path is None:
            self.texture_cache[material_name] = None
            return None

        try:
            info, rgba = parse_vtf(vtf_path)
        except (VtfError, OSError):
            self.texture_cache[material_name] = None
            return None

        # Mask alpha to 255 for opaque materials with phong reflection masks
        is_transparent = props["translucent"] or props["alphatest"]
        if not is_transparent and rgba.shape[2] == 4:
            rgba[:, :, 3] = 255

        tex_id = self._upload_rgba_texture(rgba)
        self.texture_cache[material_name] = tex_id
        return tex_id

    def _upload_rgba_texture(self, rgba, wrap_s=GL_CLAMP_TO_EDGE, wrap_t=GL_CLAMP_TO_EDGE):
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)

        height, width = rgba.shape[0], rgba.shape[1]
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, rgba.tobytes()
        )

        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id

    # ------------------------------------------------------------------
    # Source 2 materials and textures
    # ------------------------------------------------------------------

    def _is_source2_model(self):
        metadata = getattr(self.model, "metadata", None) or {}
        return metadata.get("source") == "vmdl_c"

    def _find_source2_material(self, material_name):
        reference = None
        if self.model is not None:
            paths = self.model.metadata.get("material_paths") or {}
            reference = paths.get(material_name)

        path = self._resolve_path(reference, ".vmat_c") if reference else None
        if path is None:
            path = self._resolve_path(material_name, ".vmat_c")
        return path

    def _resolve_source2_texture(self, reference):
        return self._resolve_path(reference, ".vtex_c")

    def _load_source2_material_texture(self, material_name, props):
        vmat_path = self._find_source2_material(material_name)
        if vmat_path is None:
            return None

        try:
            material = parse_vmat_c(vmat_path)
        except (VmatError, OSError):
            return None

        tint = material.vector_params.get("g_vColorTint")
        if tint and len(tint) >= 3:
            props["color"] = tuple(
                min(1.0, max(0.0, float(component))) for component in tint[:3]
            ) + (1.0,)
        if material.int_params.get("F_ALPHA_TEST"):
            props["alphatest"] = True

        reference = pick_color_texture(material)
        if not reference:
            return None

        texture_path = self._resolve_source2_texture(reference)
        if texture_path is None:
            return None

        try:
            texture = parse_vtex_c(texture_path)
        except (VtexError, OSError):
            return None

        normal_id = self._load_source2_map(pick_normal_texture(material))
        ao_reference, ao_channel = pick_occlusion_texture(material)
        ao_id = self._load_source2_map(ao_reference)
        self.material_maps[material_name] = {
            "normal": normal_id,
            "ao": ao_id,
            "ao_channel": ao_channel,
        }

        if texture.rgba is not None:
            wrap_s, wrap_t = _vtex_wrap_modes(texture.flags)
            return self._upload_rgba_texture(texture.rgba, wrap_s, wrap_t)
        return self._upload_compressed_texture(texture)

    def _load_source2_map(self, reference):
        if not reference:
            return None
        texture_path = self._resolve_source2_texture(reference)
        if texture_path is None:
            return None
        try:
            texture = parse_vtex_c(texture_path)
        except (VtexError, OSError):
            return None
        if texture.rgba is not None:
            wrap_s, wrap_t = _vtex_wrap_modes(texture.flags)
            return self._upload_rgba_texture(texture.rgba, wrap_s, wrap_t)
        return self._upload_compressed_texture(texture)

    def _ensure_model_shader(self):
        if self._model_shader is not None:
            return self._model_shader or None

        try:
            program = _compile_program(_MODEL_VERTEX_SHADER, _MODEL_FRAGMENT_SHADER)
        except Exception:
            self._model_shader = False
            return None

        self._model_shader = program
        self._model_uniforms = {
            name: glGetUniformLocation(program, name) for name in _MODEL_UNIFORMS
        }
        return program

    def _gl_extension_supported(self, extension):
        if self._gl_extensions is None:
            self._gl_extensions = self._query_gl_extensions()
        if not self._gl_extensions:
            # Extension list unavailable (core profile on some drivers):
            # let the driver accept or reject the upload.
            return True
        return extension in self._gl_extensions

    @staticmethod
    def _query_gl_extensions():
        extensions = set()
        try:
            raw = glGetString(GL_EXTENSIONS)
            if raw:
                text = raw.decode("ascii", "replace") if isinstance(raw, bytes) else str(raw)
                extensions.update(text.split())
        except Exception:
            pass
        if extensions:
            return extensions

        try:
            count = int(glGetIntegerv(GL_NUM_EXTENSIONS))
            for index in range(min(count, 8192)):
                name = glGetStringi(GL_EXTENSIONS, index)
                if name:
                    extensions.add(
                        name.decode("ascii", "replace")
                        if isinstance(name, bytes) else str(name)
                    )
        except Exception:
            pass
        return extensions

    def _upload_compressed_texture(self, texture):
        if not texture.compression or not texture.compressed_mips:
            return None

        spec = _COMPRESSED_FORMATS.get(texture.compression)
        if spec is None:
            return None
        internal_format, extension = spec
        if not self._gl_extension_supported(extension):
            fallback = _GL_EXTENSION_FALLBACKS.get(extension)
            if fallback is None or not self._gl_extension_supported(fallback):
                return None

        mips = texture.compressed_mips
        has_mipmaps = len(mips) > 1
        wrap_s, wrap_t = _vtex_wrap_modes(texture.flags)

        tex_id = None
        try:
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_MIN_FILTER,
                GL_LINEAR_MIPMAP_LINEAR if has_mipmaps else GL_LINEAR,
            )
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, len(mips) - 1)

            for level, width, height, payload in mips:
                glCompressedTexImage2D(
                    GL_TEXTURE_2D, level, internal_format, width, height, 0,
                    np.frombuffer(payload, dtype=np.uint8),
                )
            glBindTexture(GL_TEXTURE_2D, 0)
        except Exception:
            glBindTexture(GL_TEXTURE_2D, 0)
            if tex_id:
                try:
                    glDeleteTextures(1, [tex_id])
                except Exception:
                    pass
            return None
        return tex_id

    def _draw_shaded(self, offset, count, tex_id, material_name, props):
        """Draw one material batch with the normal-mapped shader."""
        program = self._ensure_model_shader()
        if not program:
            return False

        maps = self.material_maps.get(material_name) or {}
        normal_id = maps.get("normal")
        ao_id = maps.get("ao")
        uniforms = self._model_uniforms

        glUseProgram(program)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glUniform1i(uniforms["colorMap"], 0)

        if normal_id:
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, normal_id)
            glUniform1i(uniforms["normalMap"], 1)

        if ao_id:
            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, ao_id)
            glUniform1i(uniforms["aoMap"], 2)

        glUniform1i(uniforms["hasNormal"], 1 if normal_id else 0)
        glUniform1i(uniforms["hasAo"], 1 if ao_id else 0)
        glUniform1i(uniforms["aoChannel"], int(maps.get("ao_channel", 0)))
        glUniform1i(uniforms["unlit"], 1 if props.get("selfillum") else 0)
        glUniform1i(uniforms["alphaTest"], 1 if props.get("alphatest") else 0)
        glUniform1f(uniforms["alphaThreshold"], 0.1)
        glUniform1f(uniforms["aoStrength"], 0.75)
        glUniform1f(uniforms["ambientStrength"], 0.25)
        glUniform1f(uniforms["diffuseStrength"], 0.85)
        glUniform3f(uniforms["lightDir"], _LIGHT_DIR[0], _LIGHT_DIR[1], _LIGHT_DIR[2])

        color = props.get("color", (1.0, 1.0, 1.0, 1.0))
        glUniform3f(uniforms["tint"], float(color[0]), float(color[1]), float(color[2]))

        self.mesh.draw_range(offset, count, with_texcoords=True, with_colors=False)

        glUseProgram(0)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        return True

    def _build_material_batches(self):
        self._material_batches = []
        if self.model is None or not self.model.triangles:
            return None

        vertex_count = len(self.model.vertices)
        batches = {}
        batch_order = []
        for triangle in self.model.triangles:
            a, b, c = triangle.indices
            if not (
                0 <= a < vertex_count
                and 0 <= b < vertex_count
                and 0 <= c < vertex_count
            ):
                continue
            mat = triangle.material
            if mat not in batches:
                batches[mat] = []
                batch_order.append(mat)
            batches[mat].extend((a, b, c))

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
        # set_rig clears the cached skin matrices. Reapply the current
        # frame so the viewport keeps showing the posed model instead of
        # snapping back to the unposed shape until playback resumes.
        self.skinning.set_frame(self.skinning.frame)

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
        self._animation_clips = list(clips) if clips else []
        self._current_clip_index = -1
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

    def current_clip_index(self):
        return self._current_clip_index

    def clip_metadata(self, index):
        if 0 <= index < len(self._animation_clips):
            _, model = self._animation_clips[index]
            return model.metadata
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
        glEnd()

    def _draw_axes(self):
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        base_x = self.model_center[0]
        base_y = self.model_center[1]
        base_z = self.model_center[2]
        extent = max(10.0, self.camera.radius * 2.0)
        glBegin(GL_LINES)
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
        glLineWidth(1.0)

    def _get_checker_texture(self):
        tex_id = getattr(self, "_checker_tex", None)
        if tex_id:
            return tex_id
        size = 128
        tile = 16
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        for y in range(size):
            for x in range(size):
                on = ((x // tile) + (y // tile)) % 2 == 0
                if on:
                    rgba[y, x] = (200, 200, 200, 255)
                else:
                    rgba[y, x] = (80, 80, 90, 255)
        tex_id = self._upload_rgba_texture(rgba)
        self._checker_tex = tex_id
        return tex_id

    def _draw_skeleton(self):
        if self.rig is None:
            return
        try:
            if self.skeletal_animation is not None:
                locals_map = self.skeletal_animation.sample(self.skinning.frame)
            else:
                locals_map = dict(self.rig.bind_local)
            worlds = evaluate_world_matrices(self.rig, locals_map)
        except Exception:
            return
        points = {}
        for bone_id, mat in worlds.items():
            points[bone_id] = (mat[3], mat[7], mat[11])
        if not points:
            return
        glDisable(GL_LIGHTING)
        glDisable(GL_TEXTURE_2D)
        # Skeleton is an overlay: ignore depth so bones stay visible
        # through the mesh instead of hiding inside it.
        glDisable(GL_DEPTH_TEST)
        glPushMatrix()
        glTranslatef(
            self.model_position[0],
            self.model_position[1],
            self.model_position[2],
        )
        glRotatef(self.model_rotation[0], 1.0, 0.0, 0.0)
        glRotatef(self.model_rotation[1], 0.0, 1.0, 0.0)
        glRotatef(self.model_rotation[2], 0.0, 0.0, 1.0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glColor3f(1.0, 0.8, 0.2)
        for bone_id in self.rig.bone_ids:
            parent_id = self.rig.parent.get(bone_id, -1)
            if parent_id in points and bone_id in points:
                glVertex3f(*points[parent_id])
                glVertex3f(*points[bone_id])
        glEnd()
        glPointSize(5.0)
        glBegin(GL_POINTS)
        glColor3f(1.0, 0.4, 0.2)
        for bone_id in self.rig.bone_ids:
            if bone_id in points:
                glVertex3f(*points[bone_id])
        glEnd()
        glPointSize(1.0)
        glLineWidth(1.0)
        glPopMatrix()
        glEnable(GL_DEPTH_TEST)

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

        # Enable alpha test and blending for eyelashes, hair cards, and decals
        glEnable(GL_ALPHA_TEST)
        glAlphaFunc(GL_GREATER, 0.1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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

        use_checker = self.view_mode == "uv_checker"
        use_solid = self.view_mode == "solid"
        checker_id = None
        if use_checker and self.mesh.has_texcoords:
            try:
                checker_id = self._get_checker_texture()
            except Exception:
                checker_id = None

        if self._material_batches and self.mesh.has_texcoords:
            blend = not self._is_source2_model()

            for mat_name, offset, count in self._material_batches:
                props = self.material_props.get(mat_name, {})

                if props.get("nodraw"):
                    continue

                # Source 2 color maps can carry a mask in their alpha
                # channel, so only blend the materials that ask for it.
                if blend:
                    glEnable(GL_BLEND)
                elif props.get("translucent") or props.get("alphatest"):
                    glEnable(GL_BLEND)
                else:
                    glDisable(GL_BLEND)

                # Likewise, alpha test would discard most Source 2
                # surfaces whose alpha is a mask instead of coverage.
                if blend or props.get("alphatest"):
                    glEnable(GL_ALPHA_TEST)
                else:
                    glDisable(GL_ALPHA_TEST)

                # Keep the mask alpha out of the framebuffer alpha so the
                # widget stays opaque when it is composited.
                if self._is_source2_model() and not (
                    props.get("translucent") or props.get("alphatest")
                ):
                    glColorMask(True, True, True, False)
                else:
                    glColorMask(True, True, True, True)

                # Handle $nocull per material
                if props.get("nocull"):
                    glDisable(GL_CULL_FACE)
                elif self.backface_culling:
                    glEnable(GL_CULL_FACE)
                else:
                    glDisable(GL_CULL_FACE)

                # Handle $selfillum (glowing eyes / displays)
                if props.get("selfillum"):
                    glDisable(GL_LIGHTING)
                else:
                    glEnable(GL_LIGHTING)

                if use_solid:
                    glDisable(GL_TEXTURE_2D)
                    self.mesh.draw_range(offset, count, with_texcoords=False, with_colors=True)
                    continue

                if use_checker and checker_id:
                    glEnable(GL_LIGHTING)
                    glColor4f(1.0, 1.0, 1.0, 1.0)
                    glEnable(GL_TEXTURE_2D)
                    glBindTexture(GL_TEXTURE_2D, checker_id)
                    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)
                    self.mesh.draw_range(offset, count, with_texcoords=True, with_colors=False)
                    glBindTexture(GL_TEXTURE_2D, 0)
                    glDisable(GL_TEXTURE_2D)
                    continue

                tex_id = self._load_material_texture(mat_name)

                # Apply color tint
                color = props.get("color", (1.0, 1.0, 1.0, 1.0))
                glColor4f(*color)

                if tex_id and self._is_source2_model():
                    if self._draw_shaded(offset, count, tex_id, mat_name, props):
                        continue

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
            glDisable(GL_TEXTURE_2D)
            glEnable(GL_LIGHTING)
            self.mesh.draw(with_texcoords=False, with_colors=True)

        if self.wireframe_overlay:
            glDisable(GL_LIGHTING)
            glDisable(GL_TEXTURE_2D)
            glColor4f(0.1, 0.1, 0.1, 1.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(1.0)
            self.mesh.draw(with_texcoords=False, with_colors=False)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glPopMatrix()
        glColorMask(True, True, True, True)
        glDisable(GL_ALPHA_TEST)
        glDisable(GL_BLEND)
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