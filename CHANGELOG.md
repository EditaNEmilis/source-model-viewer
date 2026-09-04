# Changelog

Changelog of each versions.

## [0.6.0] | MDL and Preferences

### [0.6.1]

#### Fixed
- Compiled MDL textures rendered vertically mirrored. VVD texture coordinates
  are stored in the render-ready V convention, the parser now flags them
  pre-flipped instead of letting the renderer flip them a second time.
- Bodyparts after the first silently produced zero triangles. 
  `mstudiomodel_t.vertexindex` is a byte offset into the VVD vertex pool
  (48 bytes per vertex), not a vertex index. Multi-bodypart models now load
  all their bodygroups.
- VVD fixup reconstruction now only concatenates LOD 0 fixup runs instead of
  appending every LOD's vertices into the reconstructed pool.

### Added
- Compiled Source engine model support (`.mdl`).
- Binary vertex buffer decoder (`.vvd`) supporting LOD 0 vertex reconstruction and fixup tables.
- Hardware strip and index decoder (`.vtx`) supporting triangle lists (TRILIST) and strips (TRISTRIP) across dx90, dx80, and generic vtx formats.
- Bone hierarchy and bind pose extraction directly from `mstudiobone_t` binary structures.
- Automatic material folder detection: scans up from `models/...` paths to automatically mount the game or workshop `materials` directory.
- Model `cdmaterials` search path resolution for automatic texture locating.
- Settings dialog (`File > Settings...`, `Ctrl+,`) with persistent storage (`QSettings`).
- Configurable preferences: Auto-mount materials toggle, default materials folder, background color picker, grid display, default FOV, default FPS, and default backface culling.
- VMT shader parameter support: `$nocull`, `$no_draw`, `$selfillum`, `$color` / `$color2`, `$translucent`, and `$alphatest`.
- Eye shader texture resolution: prioritizes `$iris` for character eyes to prevent blank scleras.
- Skin replacement table translation via `skinreference_index`.

### [0.6.2]

#### Added
- GoldSrc MDL (HLMDL) support: Half-Life 1 models (IDST version 10) load
  through a version dispatch in the compiled model parser, sharing the
  existing SmdModel pipeline, skinning, and validation tooling.
- Embedded texture decoding: 8-bit palettized textures are converted to
  RGBA at parse time and uploaded directly, with per-texture palette and
  no filesystem lookup.
- External texture companion files: models storing textures in a
  `<name>T.mdl` side file resolve automatically.
- Skin family 0 translation for mesh-to-texture assignment.
- STUDIO_NF_MASKED binary transparency (palette entry 255) via the
  existing alpha test path.
- STUDIO_NF_FULLBRIGHT mapped to the unlit render path.

#### Changed
- Renderer texture upload unified into a shared RGBA helper used by both
  the VTF and embedded texture paths.

#### Notes
- Chrome (sphere map) textures sample flat; runtime sphere-mapped
  coordinates are not computed.
- Sequence animation data (compressed animvalues) is not loaded yet;
  models show their bind pose. Planned as the follow-up feature.

### Changed
- File open dialog now accepts `.mdl` alongside `.smd`, `.vta`, and `.dmx`.
- Completely removed runtime recursive filesystem scanning in texture resolution to eliminate loading freezes.
- Alpha channel masking: textures with Phong specular masks are rendered opaque unless `$translucent` or `$alphatest` is explicitly enabled.
- Help dialogs updated with MDL pipeline details, Settings documentation, and 0.6.0 version info.

### Fixed
- Stacking sub-models: bodyparts now pick visible meshes rather than rendering all alternate options on top of each other.
- VTX `MeshHeader_t` struct stride aligned to 12 bytes to prevent index and material corruption across multi-mesh models.
- Prevented double `materials/materials/` path nesting when resolving VMT relative texture paths.
- Background color now applies immediately and persists across application restarts.

### [0.6.3]

#### Added
- HL2 beta model support (MDL version 37, 2001-2003 era builds): vertices
  load directly from the MDL as 64-byte records (blend weights, position,
  normal, UV), bones from the 196-byte struct including the poseToBone
  matrix, meshes from 68-byte entries, and triangles decode from the
  companion version 6 VTX strips. Bodyparts use the 16-byte entry layout
  with sequential 280-byte model slots.
- Compiled MDL skeletal animation loading: parses local `mstudioanimdesc_t` sequences and
  decodes per-bone RLE data (quaternion64/quaternion48 rotations, half-float positions,
  delta rotation/position channels) into playable clips.
- External `.ani` animation block container support, including animblock table discovery.
- `$includemodel` chain merging: opening a model now also loads animation clips from its
  included models when they sit next to it, with cycle protection.
- Animation-only MDL loads: missing `.vvd` / `.vtx` companions no longer abort parsing,
  bones and sequences still load.

#### Fixed
- Compiled models built with extended VTX headers rendered corrupted meshes:
  strip group, strip, and mesh header strides were guessed from the MDL
  version instead of read from the data. All three are now located by scoring
  candidate chains against the MDL topology, trying relative and absolute
  offset variants at every level of the bodypart/model/LOD hierarchy.
- VVD vertex pools built from interleaved fixup tables came out truncated.
  Newer compilers emit fixup runs of every LOD in file order; when those runs
  cover the LOD vertex count the pool is rebuilt by concatenating them in
  file order, otherwise vertices fall back to original pool order. Fixes
  missing body geometry on such models.
- Multi-bodygroup models derailed VTX chain discovery when a bodypart variant
  slot contained no geometry; empty variants are now skipped while locating
  the strip chain.
- Strip group vertex ids are decoded through version-aware record layouts
  (9 bytes with id at +4 on current files, 15 bytes with id at +12 on beta
  ones), fixing scrambled vertex references.

## [0.5.0] | VTF Support

### [0.5.1]

#### Added
- VMT (Valve Material) fallback. When a material has no .vtf of its own, the viewer reads the
  matching .vmt and follows its `$basetexture` reference (or `$iris` for Eye shader
  materials, `$maintexture` as a last resort) to the real texture.
- New vmt_parser.py module with a small KeyValues tokenizer that skips nested blocks
  like Proxies and lowercases parameter names.

#### Fixed
- Out-of-range triangle indices could reach the GL index buffer on the textured draw
  path. `_build_material_batches` now validates indices the same way `_build_indices` does.
- DMX models with flipVCoordinates set had their V coordinate flipped twice. The model
  now carries a `uv_pre_flipped` flag and the renderer skips its own flip in that case.
- VTF 7.3+ files with a thumbnail decoded from a shifted offset. The low-res skip now
  only applies to 7.2 and older, since 7.3+ stores the thumbnail as its own resource
  and the high-res offset already points past it.
- Replaced the fragile *locals()* check in `open_dmx_model` with an explicit message
  variable initialized to *None*.
- The clip dropdown now syncs with the renderer's current clip by matching user data,
  so decorated labels like "ref (30 frames, 1.00s)" resolve correctly. `open_sequence`
  stores the clip name as user data too for consistency.

### Added
- VTF (Valve Texture Format) texture loading from a user-selected materials directory.
- Supports DXT1, DXT3, DXT5, BGRA8888, BGRX8888, BGR888, RGB888,
BGR565, BGRA4444, BGRA5551, RGBA8888, ABGR8888, ARGB8888, I8, IA88,
A8, and UV88 formats.
- Per-material texture binding with automatic VTF file resolution
  (direct path, materials/prefix, and recursive filename search).
- Proper mipmap generation via `glGenerateMipmap` for trilinear filtering.
- White vertex color for textured draws to prevent hash-color tinting.
- Materials folder selector (File > Set Materials Folder, Ctrl+Shift+M).

### Changed
- MeshBuffers extended with texcoord buffer and per-range drawing
  for material-batched rendering.
- Renderer splits draw calls by material group for correct texture binding.
- UV coordinates passed through the full pipeline from SMD/DMX parsers
  to the GPU.

### Fixed
- Texture cache poisoning when materials folder is set after initial render
  (cache cleared on directory add).
- VTF format enum alignment with Source SDK *imageformat.h*.
  (ATI1N/ATI2N corrected to slots 38/37) ([i](https://github.com/ValveSoftware/source-sdk-2013/blob/88fa198fba3fb85d46d4c95018254693fdc3af0a/src/public/bitmap/imageformat.h)).

## [0.4.0] | DMX, Clips and Metadata

### [0.4.1]

#### Fixed
- Removed duplicate `_build_vertex_weights` method in renderer.py that silently
  overrode the proximity skin degenerate check.
- Moved `self.skinning.set_rig()` call inside the `if model and model.bones:`
  block in `set_model` to avoid calling it with None rig.

#### Changed
- SMD `NODE_PATTERN` regex now captures full bone names with `(.*)` instead of
  single character `(.)`.

### Added
- DMX multi-clip animation support with clip selector and metadata (duration, frame count, FPS).
- Sequence scrub bar for skeletal animations (works for SMD and DMX sequences).
- Driver bone support for vertex animation (X position = intensity, Y/Z = progress).
- Backface Culling toggle (View menu) for faster rendering on closed meshes.
- Proximity Skin Fallback toggle (View menu) for meshes with degenerate skin weights.
- Styled Help dialogs (Viewer Controls, Animation Guide, Supported Formats, About) with a dark theme.
- Animation Validation dialog now shows bone mapping and vertex weight statistics.

### Changed
- Renderer split into `pose.py`, `skinning.py`, `mesh_buffers.py` and orchestration in `renderer.py`.
- VBO draw pass replaces per-triangle immediate mode. Much faster on dense meshes.
- Skeletal skinning now uses the reference rig inverse bind (fixes tearing with DMX + SMD sequences).
- Flex info TXT parser now also reads `driverbone` and `progressaxis`.
- Controls: improved model move mode with `M` key, consistent pan/rotate/zoom.

### Fixed
- Skeletal animation tearing caused by mismatched inverse bind (now using reference bind).
- Backface culling crash due to missing `set_backface_culling` method.
- DMX binary v3 loading (null byte after header, byte-length string tables).
- DMX attribute arrays with type codes starting at 15 (children, jointList, etc.).
- UnboundLocalError in `validate_animation` due to indentation issues.
- Model move mode now works reliably with `Alt`/`Ctrl` modifiers and `M` toggle.

### Removed
- Dead code from the old renderer (inline posing, skinning).

## [0.3.0] | Lotta features and more

### Added
- VTA vertex animation support with shape-key and sequence modes.
- Intensity and Progress sliders for flex shape control (tx/ty simulation).
- Flex info TXT loading (target names, FPS, default intensity/progress).
- Skeletal animation from SMD sequences (Skel checkbox).
- Driver bone detection and control (vertexAnimDriver).
- Auto Match VTA IDs (View menu) to map VTA vertices by position.
- Skeleton rig and skinning (linear blend skinning with numpy).
- DMX (Data Model eXchange) parser for KeyValues2 and binary (versions 1-5).
- DMX mesh loading with shape keys (delta states).
- DMX animation list (first clip only at this stage).
- Validate Animation dialog to inspect VTA and skeletal mapping.
- numpy dependency for vectorized posing and skinning math.

### Changed
- Posed skeleton now uses proper AngleMatrix convention and reference inverse bind.
- VTA basis is taken from the first target, allowing sparse later targets.
- SMD parser now handles UTF-8/UTF-16 and Crowbar-style comments.

### Fixed
- VTA vertex ID matching on split vertices (grouping by position).
- SMD parsing of vertex links and optional weights.
- DMX binary parsing for version 4 (short/int index mix).
- DMX reference model building (jointList, baseStates, transforms).
- DMX vertex data extraction (positions, normals, textureCoords, joint weights).

## [0.2.0] | A fine prototype

### Added
- SMD (StudioModel Data) parsing of nodes, skeleton, triangles, and vertexanimation blocks.
- Triangle rendering with per-material colours (CRC32 hash to HSV).
- Camera with orbit (left drag), pan (middle drag), and zoom (wheel/right drag).
- Grid with coloured axes.
- Model offset (Ctrl+drag, Alt+drag, or M toggle for model move mode).
- Reset functions: camera (Ctrl+R), model position (Ctrl+M), all (Ctrl+Shift+R).
- Controls help dialog.

### Changed
- Camera controls reworked (left=rotate, middle=pan, right=zoom, shift/ctrl modifiers).
- Viewport now handles mouse events, keyboard shortcuts, and device pixel ratio.
- Renderer uses fixed-function pipeline with lighting and material colours.

### Fixed
- Duplicate toolbar buttons (removed, kept menu bar).
- Mouse right-click context menu disabled.
- Window focus for keyboard shortcuts.

## [0.1.0] | If it works, it works

### Added
- Initial project scaffolding (`main.py`, `viewer/__init__.py`, `viewer/main_window.py`, `viewer/viewport.py`, and `viewer/renderer.py`).
- Basic OpenGL widget with clear colour and depth test.
- Menu bar with File (Open, Exit), View (Reset View), Help (About).
- Open file dialog (placeholder).
- Status bar.
- Requirements file (`PySide6`, `PyOpenGL`).