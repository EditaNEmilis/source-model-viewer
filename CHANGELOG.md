# Changelog

Changelog of each versions.

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
- DMX parser for KeyValues2 and binary (versions 1-5).
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
- SMD parsing of nodes, skeleton, triangles, and vertexanimation blocks.
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