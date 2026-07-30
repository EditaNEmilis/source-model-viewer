import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from viewer.flex_info import parse_flex_info
from viewer.dmx_parser import is_dmx_file, load_dmx
from viewer.smd_parser import parse_smd
from viewer.viewport import Viewport

from viewer.help_dialogs import (
    about_html,
    animation_html,
    controls_html,
    formats_html,
    show_help_dialog,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Source Model Viewer")
        self.resize(1024, 768)

        self.animation_targets = []
        self.flex_info = None
        self.flex_info_override = False

        self.viewport = Viewport(self)
        self.viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        controls = QFrame()
        controls.setFrameShape(QFrame.StyledPanel)
        controls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(6, 4, 6, 4)
        controls_layout.setSpacing(4)

        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.setRange(0, 1000)
        self.progress_slider.setSingleStep(10)
        self.progress_slider.setPageStep(100)

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setEnabled(False)
        self.intensity_slider.setRange(0, 1000)
        self.intensity_slider.setValue(1000)
        self.intensity_slider.setSingleStep(10)
        self.intensity_slider.setPageStep(100)

        self.shape_mode_check = QCheckBox("Shape Key Mode")
        self.shape_mode_check.setChecked(True)
        self.shape_mode_check.setEnabled(False)

        self.skeletal_check = QCheckBox("Skel")
        self.skeletal_check.setToolTip("Apply skeletal animation from the loaded sequence SMD")
        self.skeletal_check.setEnabled(False)

        self.driver_check = QCheckBox("Driver")
        self.driver_check.setToolTip("Use the driver bone to control intensity and progress")
        self.driver_check.setEnabled(False)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        self.fps_spin.setSuffix(" FPS")
        self.fps_spin.setEnabled(False)

        self.frame_label = QLabel("No animation")
        self.frame_label.setMinimumWidth(140)
        self.frame_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        controls_layout.addWidget(self.play_button, 0, 0, 2, 1)

        controls_layout.addWidget(QLabel("Progress"), 0, 1)
        controls_layout.addWidget(self.progress_slider, 0, 2)

        controls_layout.addWidget(self.shape_mode_check, 0, 3)
        controls_layout.addWidget(self.skeletal_check, 0, 4)
        controls_layout.addWidget(self.driver_check, 0, 5)
        controls_layout.addWidget(self.fps_spin, 0, 6)

        controls_layout.addWidget(QLabel("Intensity"), 1, 1)
        controls_layout.addWidget(self.intensity_slider, 1, 2)

        controls_layout.addWidget(self.frame_label, 1, 3, 1, 4)

        controls_layout.setColumnStretch(2, 1)

        layout.addWidget(self.viewport, 1)
        layout.addWidget(controls, 0)

        self.setCentralWidget(central)

        self.menuBar().setNativeMenuBar(False)
        self.menuBar().setFocusPolicy(Qt.NoFocus)

        self._create_actions()
        self._create_menus()

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.advance_frame)
        self.play_timer.setInterval(int(1000 / self.fps_spin.value()))

        self.play_button.clicked.connect(self.toggle_playback)
        self.progress_slider.valueChanged.connect(self.progress_changed)
        self.intensity_slider.valueChanged.connect(self.intensity_changed)
        self.shape_mode_check.toggled.connect(self.shape_mode_changed)
        self.skeletal_check.toggled.connect(self.skeletal_changed)
        self.driver_check.toggled.connect(self.driver_changed)
        self.fps_spin.valueChanged.connect(self.fps_changed)

        self.statusBar().showMessage("Source Model Viewer in Python")

    def showEvent(self, event):
        super().showEvent(event)
        self.viewport.setFocus()

    def _create_actions(self):
        self.open_action = QAction("&Open...", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.setStatusTip("Open a reference SMD or vertex animation file")
        self.open_action.triggered.connect(self.open_model)

        self.open_sequence_action = QAction("Open &Sequence SMD...", self)
        self.open_sequence_action.setStatusTip("Open a skeletal animation SMD file")
        self.open_sequence_action.triggered.connect(self.open_sequence)

        self.open_flex_info_action = QAction("Open Flex Info &TXT...", self)
        self.open_flex_info_action.setShortcut("Ctrl+Shift+T")
        self.open_flex_info_action.setStatusTip("Open an optional flex info TXT file")
        self.open_flex_info_action.triggered.connect(self.open_flex_info)

        self.clear_animation_action = QAction("&Clear Animation", self)
        self.clear_animation_action.setStatusTip("Clear the current vertex animation")
        self.clear_animation_action.triggered.connect(self.clear_animation)

        self.clear_sequence_action = QAction("Clear Sequence", self)
        self.clear_sequence_action.setStatusTip("Clear the current skeletal sequence")
        self.clear_sequence_action.triggered.connect(self.clear_sequence)

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.exit_action.setStatusTip("Exit the program")
        self.exit_action.triggered.connect(self.close)

        self.reset_camera_action = QAction("&Reset Camera", self)
        self.reset_camera_action.setShortcut("Ctrl+R")
        self.reset_camera_action.setStatusTip("Reset camera position")
        self.reset_camera_action.triggered.connect(self.reset_camera)

        self.reset_model_action = QAction("Reset &Model Position", self)
        self.reset_model_action.setShortcut("Ctrl+M")
        self.reset_model_action.setStatusTip("Reset model position")
        self.reset_model_action.triggered.connect(self.reset_model_position)

        self.reset_all_action = QAction("Reset &All", self)
        self.reset_all_action.setShortcut("Ctrl+Shift+R")
        self.reset_all_action.setStatusTip("Reset camera and model position")
        self.reset_all_action.triggered.connect(self.reset_all)

        self.auto_match_action = QAction("Auto Match VTA IDs", self)
        self.auto_match_action.setCheckable(True)
        self.auto_match_action.setChecked(False)
        self.auto_match_action.setStatusTip("Try to match VTA vertex IDs to reference vertices by position")
        self.auto_match_action.triggered.connect(self.toggle_auto_match)

        self.validate_animation_action = QAction("&Validate Animation...", self)
        self.validate_animation_action.setStatusTip("Show animation validation details")
        self.validate_animation_action.triggered.connect(self.validate_animation)

        self.controls_action = QAction("&Viewer Controls...", self)
        self.controls_action.setShortcut("F1")
        self.controls_action.setStatusTip("Mouse and keyboard controls")
        self.controls_action.triggered.connect(self.show_controls)

        self.animation_guide_action = QAction("&Animation Guide...", self)
        self.animation_guide_action.setStatusTip("How VTA and DMX animation works here")
        self.animation_guide_action.triggered.connect(self.show_animation_guide)

        self.formats_action = QAction("Supported &Formats...", self)
        self.formats_action.setStatusTip("SMD, VTA, and DMX format notes")
        self.formats_action.triggered.connect(self.show_formats)

        self.about_action = QAction("&About...", self)
        self.about_action.setStatusTip("About this viewer")
        self.about_action.triggered.connect(self.show_about)
        
        self.culling_action = QAction("Backface Culling", self)
        self.culling_action.setCheckable(True)
        self.culling_action.setChecked(False)
        self.culling_action.setStatusTip("Skip back facing triangles (faster, disable if the model looks hollow)")
        self.culling_action.triggered.connect(self.toggle_culling)
        
        self.proximity_skin_action = QAction("Proximity Skin Fallback", self)
        self.proximity_skin_action.setCheckable(True)
        self.proximity_skin_action.setChecked(False)
        self.proximity_skin_action.setStatusTip(
            "When the mesh has no usable skin weights, bind each vertex to its nearest bone so the sequence deforms it"
        )
        self.proximity_skin_action.triggered.connect(self.toggle_proximity_skin)

    def _create_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_sequence_action)
        file_menu.addAction(self.open_flex_info_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_animation_action)
        file_menu.addAction(self.clear_sequence_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.reset_camera_action)
        view_menu.addAction(self.reset_model_action)
        view_menu.addAction(self.reset_all_action)
        view_menu.addSeparator()
        view_menu.addAction(self.auto_match_action)
        view_menu.addAction(self.culling_action)
        view_menu.addAction(self.proximity_skin_action)
        view_menu.addAction(self.validate_animation_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.controls_action)
        help_menu.addAction(self.animation_guide_action)
        help_menu.addAction(self.formats_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)

    def _parse_details(self, parsed):
        details = (
            f"Triangles: {len(parsed.triangles)}\n"
            f"Vertex targets: {len(parsed.vertex_targets)}\n"
            f"Bones: {len(parsed.bones)}\n"
            f"Skeleton frames: {len(parsed.frames)}"
        )

        named_targets = [
            target.name
            for target in parsed.vertex_targets[:5]
            if target.name
        ]

        if named_targets:
            details += "\nFirst target names: " + ", ".join(named_targets)

        return details

    def open_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Model",
            "",
            "Source Model Data (*.smd *.vta *.dmx);;Vertex Animation (*.vta);;DMX Model (*.dmx);;All Files (*)",
        )

        if not file_path:
            return

        if is_dmx_file(file_path):
            self.open_dmx_model(file_path)
            return

        try:
            parsed = parse_smd(file_path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load model file.\n{error}",
            )
            return

        if parsed.has_geometry:
            if parsed.has_animation:
                self.animation_targets = parsed.vertex_targets

            self.viewport.set_model(parsed)
            self.viewport.set_animation_targets(self.animation_targets)

            self.stop_playback()
            self.update_animation_ui()

            if parsed.has_animation:
                self.statusBar().showMessage(
                    f"Loaded {len(parsed.triangles)} triangles and embedded vertex animation from {file_path}"
                )
            elif self.animation_targets:
                self.statusBar().showMessage(
                    f"Loaded {len(parsed.triangles)} triangles and applied current vertex animation from {file_path}"
                )
            else:
                self.statusBar().showMessage(
                    f"Loaded {len(parsed.triangles)} triangles from {file_path}"
                )

            self._show_animation_warnings()

        elif parsed.has_animation:
            self.animation_targets = parsed.vertex_targets
            self.viewport.set_animation_targets(self.animation_targets)

            self.stop_playback()
            self.update_animation_ui()

            if self.viewport.frame_count() > 0:
                self.statusBar().showMessage(
                    f"Loaded {len(self.animation_targets)} vertex animation targets from {file_path}"
                )
            else:
                self.statusBar().showMessage(
                    f"Loaded {len(self.animation_targets)} vertex animation targets from {file_path}. Load a reference SMD."
                )

            self._show_animation_warnings()

        else:
            lower_path = file_path.lower()
            details = self._parse_details(parsed)

            if lower_path.endswith(".vta"):
                self.statusBar().showMessage(
                    f"No vertex animation block found in {file_path}"
                )
                QMessageBox.warning(
                    self,
                    "No Vertex Animation",
                    "This VTA file did not parse as vertex animation.\n\n" + details,
                )

            elif parsed.frames:
                self.statusBar().showMessage(
                    f"Skeletal animation data in {file_path}. Load a reference mesh."
                )
                QMessageBox.information(
                    self,
                    "Skeletal Animation",
                    "This file contains skeletal animation data only.\n"
                    "Load a reference SMD mesh to display it.\n\n" + details,
                )

            else:
                self.statusBar().showMessage(
                    f"No renderable triangles or vertex animation in {file_path}"
                )
                QMessageBox.warning(
                    self,
                    "No Model Data",
                    "This file does not contain triangles or vertex animation.\n\n" + details,
                )

            return

        if self.animation_targets:
            self._try_auto_flex_info(file_path)

            if self.flex_info is not None:
                self.apply_flex_info(self.flex_info, self.flex_info_override)
                
    def open_dmx_model(self, file_path):
        try:
            reference_model, animation_model = load_dmx(file_path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load DMX file.\n{error}",
            )
            return

        loaded_something = False

        if reference_model is not None and reference_model.has_geometry:
            if reference_model.has_animation:
                self.animation_targets = reference_model.vertex_targets

            self.viewport.set_model(reference_model)
            self.viewport.set_animation_targets(self.animation_targets)

            loaded_something = True

            message = f"Loaded {len(reference_model.triangles)} triangles from DMX {file_path}"

            if reference_model.has_animation:
                message += f" and {len(reference_model.vertex_targets) - 1} shape keys"

            self.statusBar().showMessage(message)

        elif animation_model is not None:
            self.viewport.set_skeletal_animation_model(animation_model)
            loaded_something = True
            self.statusBar().showMessage(
                f"Loaded DMX animation from {file_path}"
            )

        if not loaded_something:
            QMessageBox.warning(
                self,
                "No DMX Data",
                "This DMX file does not contain a mesh or an animation.",
            )
            return

        if animation_model is not None and reference_model is not None and reference_model.has_geometry:
            self.viewport.set_skeletal_animation_model(animation_model)

        self.stop_playback()
        self.update_animation_ui()
        self._show_animation_warnings()

    def open_sequence(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Sequence SMD",
            "",
            "Source Model Data (*.smd);;All Files (*)",
        )

        if not file_path:
            return
            
        if is_dmx_file(file_path):
            try:
                _, animation_model = load_dmx(file_path)
            except Exception as error:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Could not load DMX file.\n{error}",
                )
                return

            if animation_model is None:
                QMessageBox.warning(
                    self,
                    "No Sequence Data",
                    "This DMX file does not contain an animation list.",
                )
                return

            self.viewport.set_skeletal_animation_model(animation_model)

            self._try_auto_flex_info(file_path)

            if self.flex_info is not None:
                self.apply_flex_info(self.flex_info, self.flex_info_override)

            self.stop_playback()
            self.update_animation_ui()

            self.statusBar().showMessage(
                f"Loaded DMX sequence from {file_path}"
            )
            return

        try:
            parsed = parse_smd(file_path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load sequence file.\n{error}",
            )
            return

        if not parsed.frames:
            QMessageBox.warning(
                self,
                "No Sequence Data",
                "This file does not contain skeleton frames.",
            )
            return

        self.viewport.set_skeletal_animation_model(parsed)

        self._try_auto_flex_info(file_path)

        if self.flex_info is not None:
            self.apply_flex_info(self.flex_info, self.flex_info_override)

        self.stop_playback()
        self.update_animation_ui()

        if self.viewport.has_rig():
            driver_name = self.viewport.driver_bone_name()

            if driver_name:
                self.statusBar().showMessage(
                    f"Loaded sequence from {file_path}, driver bone {driver_name}"
                )
            else:
                self.statusBar().showMessage(
                    f"Loaded sequence from {file_path}"
                )
        else:
            self.statusBar().showMessage(
                f"Loaded sequence from {file_path}. Load a reference SMD to use it."
            )

    def open_flex_info(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Flex Info TXT",
            "",
            "Text Files (*.txt);;All Files (*)",
        )

        if not file_path:
            return

        try:
            info = parse_flex_info(file_path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load flex info file.\n{error}",
            )
            return

        if (
            not info["names"]
            and info["fps"] is None
            and info["intensity"] is None
            and info["progress"] is None
            and info["driver_bone"] is None
            and info["progress_axis"] is None
        ):
            self.statusBar().showMessage(
                f"No flex info found in {file_path}"
            )
            return

        self.flex_info = info
        self.flex_info_override = True

        if self.animation_targets or self.viewport.has_skeletal_source():
            self.apply_flex_info(self.flex_info, True, f"Loaded flex info from {file_path}")
        else:
            self.statusBar().showMessage(
                f"Loaded flex info from {file_path}. Load a VTA or model to apply it."
            )

    def _try_auto_flex_info(self, file_path):
        base_path = os.path.splitext(file_path)[0]
        txt_path = base_path + ".txt"

        if not os.path.isfile(txt_path):
            return

        try:
            self.flex_info = parse_flex_info(txt_path)
            self.flex_info_override = False
        except Exception:
            self.flex_info = None
            self.flex_info_override = False

    def apply_flex_info(self, info, override=False, status_prefix=None):
        names = info.get("names") or {}
        fps = info.get("fps")
        intensity = info.get("intensity")
        progress = info.get("progress")
        driver_bone = info.get("driver_bone")
        progress_axis = info.get("progress_axis")

        applied_names = 0

        if names and self.animation_targets:
            targets = sorted(self.animation_targets, key=lambda target: target.time)

            for target in targets:
                if target.time in names:
                    if override or not target.name:
                        target.name = names[target.time]
                        applied_names += 1

            if applied_names == 0:
                keys = sorted(names)

                for i, target in enumerate(targets):
                    if i < len(keys):
                        if override or not target.name:
                            target.name = names[keys[i]]
                            applied_names += 1

            self.viewport.set_animation_targets(self.animation_targets)
            self.update_animation_ui()

        if driver_bone:
            self.viewport.set_driver_bone_name(driver_bone)

        if progress_axis:
            self.viewport.set_driver_progress_axis(progress_axis)

        if fps is not None:
            fps_value = int(round(float(fps)))
            fps_value = max(1, min(120, fps_value))
            self.fps_spin.setValue(fps_value)

        if intensity is not None and not self.viewport.driver_enabled():
            intensity_value = int(round(float(intensity) * 1000.0))
            intensity_value = max(0, min(1000, intensity_value))
            self.intensity_slider.setValue(intensity_value)

        if progress is not None and not self.viewport.driver_enabled():
            progress_value = int(round(float(progress) * 1000.0))
            progress_value = max(0, min(1000, progress_value))
            self.progress_slider.setValue(progress_value)

        if self.viewport.driver_enabled():
            self.sync_sliders_from_renderer()

        if status_prefix:
            self.statusBar().showMessage(
                f"{status_prefix}, {applied_names} names applied"
            )
        elif applied_names > 0:
            self.statusBar().showMessage(
                f"Applied {applied_names} flex info names"
            )

    def _show_animation_warnings(self):
        ignored = self.viewport.animation_ignored_count()

        if ignored > 0:
            self.statusBar().showMessage(
                f"Warning: {ignored} VTA vertex IDs did not match the reference mesh."
            )

    def clear_animation(self):
        self.animation_targets = []
        self.flex_info = None
        self.flex_info_override = False

        self.viewport.set_animation_targets([])

        self.stop_playback()
        self.update_animation_ui()

        self.statusBar().showMessage("Animation cleared")

    def clear_sequence(self):
        self.viewport.clear_skeletal_animation()

        self.stop_playback()
        self.update_animation_ui()

        self.statusBar().showMessage("Sequence cleared")

    def reset_camera(self):
        self.viewport.reset_camera()
        self.statusBar().showMessage("Camera reset")

    def reset_model_position(self):
        self.viewport.reset_model_position()
        self.statusBar().showMessage("Model position reset")

    def reset_all(self):
        self.viewport.reset_all()
        self.stop_playback()
        self.update_animation_ui()
        self.statusBar().showMessage("Camera and model position reset")

    def toggle_auto_match(self):
        enabled = self.auto_match_action.isChecked()

        self.viewport.set_match_ids_by_position(enabled)

        self.stop_playback()
        self.update_animation_ui()
        self._show_animation_warnings()

        if enabled:
            self.statusBar().showMessage("Auto match enabled")
        else:
            self.statusBar().showMessage("Auto match disabled")
            
    def toggle_culling(self):
        self.viewport.set_backface_culling(self.culling_action.isChecked())
        
    def toggle_proximity_skin(self):
        enabled = self.proximity_skin_action.isChecked()

        self.viewport.set_proximity_skin(enabled)
        self.update_animation_ui()

        if enabled:
            self.statusBar().showMessage("Proximity skin enabled")
        else:
            self.statusBar().showMessage("Proximity skin disabled")

    def validate_animation(self):
        lines = []

        reference_count = self.viewport.animation_reference_vertex_count()
        target_count = len(self.animation_targets)
        frame_count = self.viewport.frame_count()
        first_target_count = self.viewport.animation_first_target_count()
        max_vertex_id = self.viewport.animation_max_vertex_id()
        ignored = self.viewport.animation_ignored_count()
        matched_ids = self.viewport.animation_matched_count()
        matched_vertices = self.viewport.animation_matched_vertex_count()

        intensity = self.viewport.intensity()
        progress = self.viewport.progress()

        sequence_frames = self.viewport.skeletal_frame_count()
        sequence_loaded = self.viewport.has_skeletal_animation()
        sequence_source = self.viewport.has_skeletal_source()

        if reference_count > 0:
            lines.append(f"Reference vertices: {reference_count}")
        else:
            lines.append("No reference mesh loaded.")

        lines.append(f"Loaded VTA targets: {target_count}")
        lines.append(f"Built animation frames: {frame_count}")
        lines.append(f"First target vertex count: {first_target_count}")
        lines.append(f"Max VTA vertex ID: {max_vertex_id}")
        lines.append(f"Ignored VTA overrides: {ignored}")
        lines.append(f"Position matched VTA IDs: {matched_ids}")
        lines.append(f"Affected reference vertices: {matched_vertices}")
        lines.append(f"Auto match enabled: {self.auto_match_action.isChecked()}")
        lines.append(f"Shape key mode: {self.shape_mode_check.isChecked()}")
        lines.append(f"Intensity: {intensity:.2f}")
        lines.append(f"Progress: {progress:.2f}")

        lines.append("")
        lines.append("Sequence")
        lines.append(f"Sequence source loaded: {sequence_source}")
        lines.append(f"Sequence usable: {sequence_loaded}")
        lines.append(f"Sequence frames: {sequence_frames}")
        lines.append(f"Skeletal enabled: {self.viewport.skeletal_enabled()}")
        lines.append(f"Driver bone: {self.viewport.driver_bone_name() or 'none'}")
        lines.append(f"Driver enabled: {self.viewport.driver_enabled()}")
        lines.append(f"Driver progress axis: {self.viewport.driver_progress_axis()}")

        lines.append("")
        lines.append("Bone mapping")
        lines.append(f"Reference bones: {self.viewport.skeletal_rig_bone_count()}")
        lines.append(f"Sequence bones: {self.viewport.skeletal_anim_bone_count()}")
        lines.append(f"Name matched: {self.viewport.skeletal_mapped_count()}")
        lines.append(f"Bones with motion data: {self.viewport.skeletal_driven_count()}")

        unmapped = self.viewport.skeletal_unmapped_names()

        if unmapped:
            preview = ", ".join(unmapped[:12])

            if len(unmapped) > 12:
                preview += f", ... and {len(unmapped) - 12} more"

            lines.append(f"Unmatched sequence bones: {preview}")

        stats = self.viewport.skeletal_weight_stats()

        if stats is not None:
            lines.append("")
            lines.append("Vertex weights")
            lines.append(f"Weighted vertices: {stats['vertices']}")
            lines.append(
                f"Bones that own vertices: {stats['bones_with_vertices']} of {stats['rig_bones']}"
            )
            lines.append(f"Unbound vertices (frozen): {stats['unbound_vertices']}")

            if stats["top_bones"]:
                top = ", ".join(
                    f"{name}:{count}" for name, count in stats["top_bones"]
                )
                lines.append(f"Top bones by vertex count: {top}")

            half_rig = stats["rig_bones"] // 2
            half_verts = stats["vertices"] // 2

            if stats["bones_with_vertices"] < half_rig or stats["unbound_vertices"] > half_verts:
                lines.append(
                    "Weights look degenerate. Turn on View > Proximity Skin Fallback, then press Play."
                )

        if reference_count > 0 and max_vertex_id >= 0 and max_vertex_id >= reference_count:
            lines.append("VTA IDs exceed reference vertex count.")

        if ignored > 0:
            lines.append("Some VTA vertices are not applied.")

        if target_count > 0 and frame_count == 0:
            lines.append("Load a reference SMD to build frames.")

        if reference_count > 0 and first_target_count > 0 and first_target_count < reference_count:
            lines.append(
                f"First VTA target covers {first_target_count} of {reference_count} reference vertices. "
                "This is normal if the VTA only animates part of the mesh."
            )

        QMessageBox.information(
            self,
            "Animation Validation",
            "\n".join(lines),
        )

    def toggle_playback(self):
        if self.play_timer.isActive():
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if not self.play_button.isEnabled():
            return

        self.play_button.setText("Pause")
        self.play_timer.start()

    def stop_playback(self):
        self.play_button.setText("Play")
        self.play_timer.stop()

    def _sequence_playback_active(self):
        return (
            self.viewport.has_skeletal_animation()
            and (self.skeletal_check.isChecked() or self.driver_check.isChecked())
        )

    def advance_frame(self):
        if self._sequence_playback_active():
            sequence_count = self.viewport.skeletal_frame_count()

            if sequence_count <= 1:
                self.stop_playback()
                return

            current = self.viewport.skeletal_current_frame()
            new_frame = current + 1.0

            if new_frame >= sequence_count:
                new_frame = 0.0

            self.viewport.set_skeletal_frame(new_frame)

            if self.driver_check.isChecked():
                self.sync_sliders_from_renderer()

            self.update_frame_label()
            return

        count = self.viewport.frame_count()

        if count <= 1:
            self.stop_playback()
            return

        step = max(1, int(round(1000.0 / float(count - 1))))
        value = self.progress_slider.value()

        if value + step >= 1000:
            self.progress_slider.setValue(0)
        else:
            self.progress_slider.setValue(value + step)

    def progress_changed(self, value):
        progress = value / 1000.0
        self.viewport.set_progress(progress)
        self.update_frame_label()

    def intensity_changed(self, value):
        intensity = value / 1000.0
        self.viewport.set_intensity(intensity)
        self.update_frame_label()

    def shape_mode_changed(self, checked):
        if checked:
            self.viewport.set_animation_mode("shape")
        else:
            self.viewport.set_animation_mode("sequence")

        self.update_frame_label()

    def skeletal_changed(self, checked):
        self.viewport.set_skeletal_enabled(checked)
        self.update_animation_ui()

    def driver_changed(self, checked):
        self.viewport.set_driver_enabled(checked)

        if checked:
            self.sync_sliders_from_renderer()

        self.update_animation_ui()

    def fps_changed(self, value):
        if value < 1:
            value = 1

        self.play_timer.setInterval(int(1000 / value))

    def sync_sliders_from_renderer(self):
        progress_value = int(round(self.viewport.progress() * 1000.0))
        intensity_value = int(round(self.viewport.intensity() * 1000.0))

        progress_value = max(0, min(1000, progress_value))
        intensity_value = max(0, min(1000, intensity_value))

        self.progress_slider.blockSignals(True)
        self.intensity_slider.blockSignals(True)

        self.progress_slider.setValue(progress_value)
        self.intensity_slider.setValue(intensity_value)

        self.progress_slider.blockSignals(False)
        self.intensity_slider.blockSignals(False)

        self.update_frame_label()

    def update_animation_ui(self):
        self.stop_playback()

        vta_count = self.viewport.frame_count()
        sequence_count = self.viewport.skeletal_frame_count()
        has_sequence = self.viewport.has_skeletal_animation()
        has_sequence_source = self.viewport.has_skeletal_source()
        has_driver = self.viewport.has_driver_bone()

        self.skeletal_check.setEnabled(has_sequence)
        self.driver_check.setEnabled(has_sequence and has_driver)

        self.skeletal_check.blockSignals(True)
        self.driver_check.blockSignals(True)

        self.skeletal_check.setChecked(self.viewport.skeletal_enabled())
        self.driver_check.setChecked(self.viewport.driver_enabled())

        self.skeletal_check.blockSignals(False)
        self.driver_check.blockSignals(False)

        driver_active = (
            has_sequence
            and has_driver
            and self.driver_check.isChecked()
        )

        if vta_count > 0 or driver_active:
            self.intensity_slider.setEnabled(not driver_active)
            self.shape_mode_check.setEnabled(vta_count > 0)
        else:
            self.intensity_slider.setEnabled(False)
            self.shape_mode_check.setEnabled(False)

        if (vta_count > 1 or driver_active) and not driver_active:
            self.progress_slider.setEnabled(True)
        else:
            self.progress_slider.setEnabled(False)

        if driver_active:
            self.sync_sliders_from_renderer()
        else:
            if vta_count == 0:
                self.progress_slider.setValue(0)
                self.intensity_slider.setValue(1000)
                self.viewport.set_progress(0.0)
                self.viewport.set_intensity(1.0)

        sequence_playback = self._sequence_playback_active()

        if sequence_playback:
            play_enabled = sequence_count > 1
        else:
            play_enabled = vta_count > 1

        self.play_button.setEnabled(play_enabled)
        self.fps_spin.setEnabled(play_enabled)

        if vta_count > 0 or has_sequence or has_sequence_source:
            self.update_frame_label()
        elif self.animation_targets:
            self.frame_label.setText("No reference mesh")
        else:
            self.frame_label.setText("No animation")

    def update_frame_label(self):
        vta_count = self.viewport.frame_count()
        sequence_count = self.viewport.skeletal_frame_count()

        if self._sequence_playback_active() and sequence_count > 0:
            frame = self.viewport.skeletal_current_frame()
            index = int(frame)

            if index < 0:
                index = 0

            if index >= sequence_count:
                index = sequence_count - 1

            text = f"Seq {index + 1}/{sequence_count}"

            if self.driver_check.isChecked():
                intensity = self.viewport.intensity()
                progress = self.viewport.progress()
                text += f"  I:{intensity:.2f} P:{progress:.2f}"

            self.frame_label.setText(text)
            return

        if vta_count <= 0:
            if self.animation_targets or self.viewport.has_skeletal_source():
                self.frame_label.setText("No reference mesh")
            else:
                self.frame_label.setText("No animation")
            return

        progress = self.progress_slider.value() / 1000.0
        intensity = self.intensity_slider.value() / 1000.0

        if vta_count <= 1:
            frame_float = 0.0
        else:
            frame_float = progress * (vta_count - 1)

        if self.shape_mode_check.isChecked():
            index = int(frame_float + 0.5)
        else:
            index = int(frame_float)

        if index < 0:
            index = 0

        if index >= vta_count:
            index = vta_count - 1

        name = self.viewport.animation_name(index)

        if name:
            self.frame_label.setText(
                f"{index + 1}/{vta_count}: {name}  I:{intensity:.2f}"
            )
        else:
            self.frame_label.setText(
                f"Target {index + 1}/{vta_count}  I:{intensity:.2f}"
            )

    def show_controls(self):
        show_help_dialog(self, "Viewer Controls", controls_html())

    def show_animation_guide(self):
        show_help_dialog(self, "Animation Guide", animation_html())

    def show_formats(self):
        show_help_dialog(self, "Supported Formats", formats_html())

    def show_about(self):
        show_help_dialog(self, "About", about_html(), width=480, height=440)