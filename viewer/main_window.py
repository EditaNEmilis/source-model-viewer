import os

from PySide6.QtCore import Qt, QTimer
from PySide6.QtCore import QCoreApplication, QSettings
from PySide6.QtGui import QAction, QColor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from viewer.app_icon import apply_app_icon
from viewer.flex_info import parse_flex_info
from viewer.dmx_parser import is_dmx_file, load_dmx
from viewer.mdl_parser import is_mdl_file, is_vmdl_c_file, parse_mdl
from viewer.smd_parser import parse_smd
from viewer.viewport import Viewport

from viewer.help_dialogs import (
    about_html,
    animation_html,
    controls_html,
    formats_html,
    show_help_dialog,
)

from viewer.settings_dialog import SettingsDialog

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        QCoreApplication.setOrganizationName("SourceModelViewer")
        QCoreApplication.setApplicationName("SourceModelViewer")

        self.setWindowTitle("Source Model Viewer")
        apply_app_icon(self)
        self.resize(1024, 768)
        self._geometry_restored = False

        self.animation_targets = []
        self.flex_info = None
        self.flex_info_override = False
        self._bodygroup_model = None
        self._bodygroup_selection = []

        self.setAcceptDrops(True)

        self.viewport = Viewport(self)
        self.viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        controls = QFrame()
        controls.setFrameShape(QFrame.StyledPanel)
        controls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.controls_frame = controls

        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(6, 4, 6, 4)
        controls_layout.setSpacing(4)

        self.play_button = QPushButton("Play")
        self.play_button.setEnabled(False)
        self.play_button.setToolTip("Play or pause the active animation (Space)")

        self.first_frame_button = QPushButton("|<")
        self.prev_frame_button = QPushButton("<")
        self.next_frame_button = QPushButton(">")
        self.last_frame_button = QPushButton(">|")
        for step_button in (
            self.first_frame_button,
            self.prev_frame_button,
            self.next_frame_button,
            self.last_frame_button,
        ):
            step_button.setFixedWidth(30)
            step_button.setEnabled(False)
        self.first_frame_button.setToolTip("Jump to the first frame (Home)")
        self.prev_frame_button.setToolTip("Step one frame back (Left)")
        self.next_frame_button.setToolTip("Step one frame forward (Right)")
        self.last_frame_button.setToolTip("Jump to the last frame (End)")
        for button in (
            self.play_button,
            self.first_frame_button,
            self.prev_frame_button,
            self.next_frame_button,
            self.last_frame_button,
        ):
            button.setAutoDefault(False)

        self.loop_check = QCheckBox("Loop")
        self.loop_check.setChecked(True)
        self.loop_check.setToolTip("Wrap playback at the end (on by default)")

        self.clip_info_label = QLabel("")
        self.clip_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.clip_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.clip_info_label.setToolTip("Selected clip: frames, fps, duration, looping")

        self.model_info_label = QLabel("")
        self.model_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.model_info_label.setToolTip("Loaded model: triangles, bones, materials, clips")

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

        self.clip_combo = QComboBox()
        self.clip_combo.setEnabled(False)
        self.clip_combo.setToolTip("Select animation clip")
        self.clip_combo.currentIndexChanged.connect(self.clip_changed)

        self.blend_combo = QComboBox()
        self.blend_combo.setToolTip("Select a blend direction for this sequence")
        self.blend_combo.currentIndexChanged.connect(self.blend_changed)

        self.sequence_slider = QSlider(Qt.Horizontal)
        self.sequence_slider.setEnabled(False)
        self.sequence_slider.setRange(0, 1000)
        self.sequence_slider.setSingleStep(10)
        self.sequence_slider.setPageStep(100)
        self.sequence_slider.valueChanged.connect(self.sequence_progress_changed)

        transport = QWidget()
        transport_layout = QHBoxLayout(transport)
        transport_layout.setContentsMargins(0, 0, 0, 0)
        transport_layout.setSpacing(2)
        transport_layout.addWidget(self.first_frame_button)
        transport_layout.addWidget(self.prev_frame_button)
        transport_layout.addWidget(self.play_button)
        transport_layout.addWidget(self.next_frame_button)
        transport_layout.addWidget(self.last_frame_button)

        info_strip = QWidget()
        info_layout = QHBoxLayout(info_strip)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(12)
        info_layout.addWidget(self.model_info_label, 1)
        info_layout.addWidget(self.clip_info_label, 0)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        # Row 0 / 1: playback transport, scrubbing and toggles.
        controls_layout.addWidget(transport, 0, 0, 2, 1)

        controls_layout.addWidget(QLabel("Progress"), 0, 1)
        controls_layout.addWidget(self.progress_slider, 0, 2)

        controls_layout.addWidget(self.shape_mode_check, 0, 3)
        controls_layout.addWidget(self.skeletal_check, 0, 4)
        controls_layout.addWidget(self.driver_check, 0, 5)
        controls_layout.addWidget(self.fps_spin, 0, 6)

        controls_layout.addWidget(QLabel("Intensity"), 1, 1)
        controls_layout.addWidget(self.intensity_slider, 1, 2)
        controls_layout.addWidget(self.loop_check, 1, 3)
        controls_layout.addWidget(self.frame_label, 1, 4, 1, 3)

        # Row 2: clip selection and sequence scrubbing.
        controls_layout.addWidget(QLabel("Clip"), 2, 0)
        controls_layout.addWidget(self.clip_combo, 2, 1, 1, 3)
        controls_layout.addWidget(QLabel("Sequence"), 2, 4)
        controls_layout.addWidget(self.sequence_slider, 2, 5, 1, 2)

        # Row 3: blend selector for multi-blend sequences.
        self.blend_bar = QWidget()
        blend_layout = QHBoxLayout(self.blend_bar)
        blend_layout.setContentsMargins(0, 0, 0, 0)
        blend_layout.setSpacing(6)
        blend_layout.addWidget(QLabel("Blend"))
        blend_layout.addWidget(self.blend_combo)
        blend_layout.addStretch(1)
        self.blend_bar.setVisible(False)
        controls_layout.addWidget(self.blend_bar, 3, 0, 1, 7)

        # Row 4: bodygroup selectors, one combo per bodypart. Hidden for
        # formats without bodygroup metadata. The scroll area keeps long
        # selector rows (beta models have many bodyparts) from widening
        # the window.
        self.bodygroup_bar = QWidget()
        self.bodygroup_layout = QHBoxLayout(self.bodygroup_bar)
        self.bodygroup_layout.setContentsMargins(0, 0, 0, 0)
        self.bodygroup_layout.setSpacing(6)
        self.bodygroup_scroll = QScrollArea()
        self.bodygroup_scroll.setFrameShape(QFrame.NoFrame)
        self.bodygroup_scroll.setWidgetResizable(False)
        self.bodygroup_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.bodygroup_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.bodygroup_scroll.setFixedHeight(34)
        self.bodygroup_scroll.setWidget(self.bodygroup_bar)
        self.bodygroup_scroll.setVisible(False)
        controls_layout.addWidget(self.bodygroup_scroll, 4, 0, 1, 7)

        # Row 5 / 6: persistent readouts, separated from the controls.
        controls_layout.addWidget(separator, 5, 0, 1, 7)
        controls_layout.addWidget(info_strip, 6, 0, 1, 7)

        controls_layout.setColumnStretch(2, 1)
        controls_layout.setColumnStretch(5, 1)

        layout.addWidget(self.viewport, 1)
        layout.addWidget(controls, 0)

        self.setCentralWidget(central)

        self.menuBar().setNativeMenuBar(False)
        self.menuBar().setFocusPolicy(Qt.NoFocus)

        self._create_actions()
        self._create_menus()

        self._clip_models = []
        self._blend_models = []

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
        self.first_frame_button.clicked.connect(self.goto_first_frame)
        self.prev_frame_button.clicked.connect(self.step_frame_back)
        self.next_frame_button.clicked.connect(self.step_frame_forward)
        self.last_frame_button.clicked.connect(self.goto_last_frame)

        self.statusBar().showMessage("Source Model Viewer in Python")

    def keyPressEvent(self, event):
        # Key events that a child widget accepts (sliders, spin boxes,
        # checkboxes, line edits) never reach here, so their native handling
        # stays intact.
        key = event.key()
        if key == Qt.Key_Space and self.play_button.isEnabled():
            self.toggle_playback()
            event.accept()
            return
        if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Home, Qt.Key_End):
            if key == Qt.Key_Left:
                self.step_frame(-1)
            elif key == Qt.Key_Right:
                self.step_frame(1)
            elif key == Qt.Key_Home:
                self.set_frame_boundary(True)
            else:
                self.set_frame_boundary(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._geometry_restored:
            self._geometry_restored = True
            settings = QSettings("SourceModelViewer", "Settings")
            geometry = settings.value("window/geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
                screen = QGuiApplication.primaryScreen()
                if screen is not None:
                    available = screen.availableGeometry()
                    width = min(self.width(), available.width())
                    height = min(self.height(), available.height())
                    x = max(available.left(), min(self.x(), available.right() - width))
                    y = max(available.top(), min(self.y(), available.bottom() - height))
                    self.setGeometry(x, y, width, height)
        self.apply_settings()
        self.viewport.setFocus()

    def closeEvent(self, event):
        settings = QSettings("SourceModelViewer", "Settings")
        settings.setValue("window/geometry", self.saveGeometry())
        settings.sync()
        super().closeEvent(event)

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

        self.settings_action = QAction("&Settings...", self)
        self.settings_action.setShortcut("Ctrl+,")
        self.settings_action.setStatusTip("Configure viewer preferences")
        self.settings_action.triggered.connect(self.open_settings)

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

        self.show_grid_action = QAction("Show Ground Grid", self)
        self.show_grid_action.setCheckable(True)
        self.show_grid_action.setChecked(True)
        self.show_grid_action.setStatusTip("Toggle the ground grid")
        self.show_grid_action.triggered.connect(self.toggle_grid)

        self.show_axes_action = QAction("Show 3D Axes", self)
        self.show_axes_action.setCheckable(True)
        self.show_axes_action.setChecked(True)
        self.show_axes_action.setStatusTip("Toggle the red green blue axis lines")
        self.show_axes_action.triggered.connect(self.toggle_axes)

        self.show_controls_action = QAction("Show Playback Controls", self)
        self.show_controls_action.setCheckable(True)
        self.show_controls_action.setChecked(True)
        self.show_controls_action.setStatusTip("Show or hide the bottom playback controls")
        self.show_controls_action.triggered.connect(self.toggle_show_controls)

        self.wireframe_action = QAction("Wireframe Overlay", self)
        self.wireframe_action.setCheckable(True)
        self.wireframe_action.setChecked(False)
        self.wireframe_action.setStatusTip("Draw a wireframe overlay on the model")
        self.wireframe_action.triggered.connect(self.toggle_wireframe)

        self.skeleton_action = QAction("Show Skeleton", self)
        self.skeleton_action.setCheckable(True)
        self.skeleton_action.setChecked(False)
        self.skeleton_action.setStatusTip("Draw bones and joints in the 3D view")
        self.skeleton_action.triggered.connect(self.toggle_skeleton)

        self.solid_view_action = QAction("Solid Color View", self)
        self.solid_view_action.setCheckable(True)
        self.solid_view_action.setChecked(False)
        self.solid_view_action.setStatusTip("Ignore textures and show flat material colors")
        self.solid_view_action.triggered.connect(self.toggle_solid_view)

        self.uv_checker_action = QAction("UV Checker View", self)
        self.uv_checker_action.setCheckable(True)
        self.uv_checker_action.setChecked(False)
        self.uv_checker_action.setStatusTip("Show a checker pattern for UV inspection")
        self.uv_checker_action.triggered.connect(self.toggle_uv_checker)

        self.materials_folder_action = QAction("Set &Materials Folder...", self)
        self.materials_folder_action.setShortcut("Ctrl+Shift+M")
        self.materials_folder_action.setStatusTip(
            "Add a folder to search for VTF textures"
        )
        self.materials_folder_action.triggered.connect(self.set_materials_folder)

        self.materials_action = QAction("&Materials...", self)
        self.materials_action.setStatusTip(
            "List the model materials, their textures and any that are missing"
        )
        self.materials_action.triggered.connect(self.show_materials)

    def _create_menus(self):
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_sequence_action)
        file_menu.addAction(self.open_flex_info_action)
        file_menu.addAction(self.materials_folder_action)
        file_menu.addSeparator()
        file_menu.addAction(self.clear_animation_action)
        file_menu.addAction(self.clear_sequence_action)
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        view_menu = self.menuBar().addMenu("&View")
        view_menu.addAction(self.reset_camera_action)
        view_menu.addAction(self.reset_model_action)
        view_menu.addAction(self.reset_all_action)
        view_menu.addSeparator()
        view_menu.addAction(self.show_grid_action)
        view_menu.addAction(self.show_axes_action)
        view_menu.addAction(self.show_controls_action)
        view_menu.addSeparator()
        view_menu.addAction(self.materials_action)
        view_menu.addSeparator()
        view_menu.addAction(self.solid_view_action)
        view_menu.addAction(self.uv_checker_action)
        view_menu.addAction(self.wireframe_action)
        view_menu.addAction(self.skeleton_action)
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

    def _clip_display_label(self, index, clip_name, clip_model, duplicate_name):
        metadata = getattr(clip_model, "metadata", {}) or {}
        frames = len(clip_model.frames) if clip_model.frames else 0
        try:
            duration = float(metadata.get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0

        try:
            sequence_group = int(metadata.get("sequence_group", 0) or 0)
        except (TypeError, ValueError):
            sequence_group = 0
        try:
            blend_count = max(1, int(metadata.get("blend_count", 1) or 1))
            blend_index = max(0, int(metadata.get("blend_index", 0) or 0))
        except (TypeError, ValueError):
            blend_count = 1
            blend_index = 0

        qualifiers = []
        if sequence_group > 0:
            qualifiers.append(f"group {sequence_group}")
        if blend_count > 1:
            qualifiers.append(f"blend {blend_index + 1}/{blend_count}")
        sequence_blends = metadata.get("sequence_blends") or []
        if len(sequence_blends) > 1:
            qualifiers.append(f"{len(sequence_blends)} blends")
        if metadata.get("fallback"):
            qualifiers.append("bind pose")
        if duplicate_name:
            qualifiers.append(f"clip {index + 1}")

        label = clip_name
        if qualifiers:
            label += " [" + ", ".join(qualifiers) + "]"
        return f"{label} ({frames} frames, {duration:.2f}s)"

    def _set_animation_clips(self, clips):
        clips = list(clips) if clips else []
        self._clip_models = clips
        self.viewport.set_animation_clips(clips)

        name_counts = {}
        for clip_name, _ in clips:
            key = clip_name.casefold()
            name_counts[key] = name_counts.get(key, 0) + 1

        self.clip_combo.blockSignals(True)
        self.clip_combo.clear()
        for index, (clip_name, clip_model) in enumerate(clips):
            duplicate_name = name_counts.get(clip_name.casefold(), 0) > 1
            label = self._clip_display_label(
                index, clip_name, clip_model, duplicate_name
            )
            self.clip_combo.addItem(label, index)
        if clips:
            self.clip_combo.setCurrentIndex(0)
        self.clip_combo.setEnabled(bool(clips))
        self.clip_combo.blockSignals(False)

        self.sequence_slider.blockSignals(True)
        self.sequence_slider.setValue(0)
        self.sequence_slider.blockSignals(False)

        self._update_blend_selector()

        self.stop_playback()
        self.update_animation_ui()

    def _display_compiled_model(self, parsed, file_path):
        # Shared post-parse handling for compiled models (Source 1 MDL
        # and Source 2 .vmdl_c): mount materials, push to the viewport,
        # wire animation clips, and report the load in the status bar.
        auto_mat = self._auto_add_materials_root(file_path)

        self._set_animation_clips([])
        self.viewport.set_model(parsed)
        self._set_bodygroups(parsed)

        targets = getattr(parsed, "vertex_targets", []) or []
        self.animation_targets = targets
        self.viewport.set_animation_targets(targets)

        clips = parsed.metadata.get("animation_clips") or []
        self._set_animation_clips(clips)
        self._maybe_autoplay()

        msg = f"Loaded {len(parsed.triangles)} triangles from {file_path}"
        if auto_mat:
            msg += f" (Mounted materials: {auto_mat})"
        if clips:
            msg += f" and {len(clips)} animation clips"
        if len(targets) > 1:
            msg += f" and {len(targets) - 1} shape keys"
        anim_error = parsed.metadata.get("animation_error")
        if anim_error:
            msg += f" (animation error: {anim_error})"
        morph_error = parsed.metadata.get("morph_error")
        if morph_error:
            msg += f" (morphs: {morph_error})"
        if parsed.metadata.get("vmdl_truncated_tail"):
            msg += " (partial: trailing Source 2 scene data skipped)"
        self.statusBar().showMessage(msg)

    def open_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Model",
            "",
            "Source Model Files (*.smd *.vta *.dmx *.mdl *.vmdl_c);;Source Compiled Model (*.mdl);;Source 2 Compiled Model (*.vmdl_c);;StudioModel Data (*.smd);;Vertex Animation (*.vta);;Data Model eXchange (*.dmx);;All Files (*)",
        )

        if not file_path:
            return

        self.load_model_file(file_path)

    def load_model_file(self, file_path):
        if is_dmx_file(file_path):
            self.open_dmx_model(file_path)
            return

        if file_path.lower().endswith(".vmdl_c") or is_vmdl_c_file(file_path):
            from viewer.vmdl_parser import parse_vmdl_c

            try:
                parsed = parse_vmdl_c(file_path)
            except Exception as error:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Could not load Source 2 model.\n{error}",
                )
                return

            self._display_compiled_model(parsed, file_path)
            return

        if is_mdl_file(file_path) or file_path.lower().endswith(".mdl"):
            try:
                parsed = parse_mdl(file_path)
            except Exception as error:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Could not load compiled MDL model.\n{error}",
                )
                return

            self._display_compiled_model(parsed, file_path)
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
            self._set_animation_clips([])
            if parsed.has_animation:
                self.animation_targets = parsed.vertex_targets

            self.viewport.set_model(parsed)
            self.viewport.set_animation_targets(self.animation_targets)
            self._set_bodygroups(None)

            self.stop_playback()
            self.update_animation_ui()
            self._maybe_autoplay()

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
            self._maybe_autoplay()

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

        settings = QSettings("SourceModelViewer", "Settings")
        auto_mount_enabled = settings.value("materials/auto_mount", True, type=bool)

        if auto_mount_enabled:
            self._auto_add_materials_root(file_path)

    @staticmethod
    def _model_paths_from_mime(mime_data):
        if mime_data is None or not mime_data.hasUrls():
            return []
        paths = []
        for url in mime_data.urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        return paths

    @staticmethod
    def _is_model_path(path):
        return path.lower().endswith((".smd", ".vta", ".dmx", ".mdl", ".vmdl_c"))

    def dragEnterEvent(self, event):
        paths = self._model_paths_from_mime(event.mimeData())
        if any(self._is_model_path(path) for path in paths):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData() and event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [
            path
            for path in self._model_paths_from_mime(event.mimeData())
            if self._is_model_path(path)
        ]
        if not paths:
            event.ignore()
            return

        event.acceptProposedAction()

        if len(paths) > 1:
            self.statusBar().showMessage(
                f"Loading {os.path.basename(paths[0])}; "
                f"{len(paths) - 1} other file(s) ignored"
            )

        self.load_model_file(paths[0])

    def open_dmx_model(self, file_path):
        try:
            reference_model, animation_clips = load_dmx(file_path)
        except Exception as error:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Could not load DMX file.\n{error}",
            )
            return

        loaded_something = False
        message = None

        if reference_model is not None and reference_model.has_geometry:
            self._set_animation_clips([])
            if reference_model.has_animation:
                self.animation_targets = reference_model.vertex_targets

            self.viewport.set_model(reference_model)
            self.viewport.set_animation_targets(self.animation_targets)
            self._set_bodygroups(None)

            loaded_something = True

            message = f"Loaded {len(reference_model.triangles)} triangles from DMX {file_path}"

            if reference_model.has_animation:
                message += f" and {len(reference_model.vertex_targets) - 1} shape keys"

            self.statusBar().showMessage(message)

        # Handle animation clips
        if (
            reference_model is not None and reference_model.has_geometry
        ) or animation_clips:
            self._set_animation_clips(animation_clips)
        if animation_clips:
            loaded_something = True
            if message is None:
                message = f"Loaded {len(animation_clips)} animation clips from DMX"
            else:
                message += f" and {len(animation_clips)} animation clips"
            self.statusBar().showMessage(message)

        if not loaded_something:
            QMessageBox.warning(
                self,
                "No DMX Data",
                "This DMX file does not contain a mesh or an animation.",
            )
            return

        self._maybe_autoplay()
        self._show_animation_warnings()

    def open_sequence(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Sequence SMD",
            "",
            "StudioModel Data (*.smd);;All Files (*)",
        )

        if not file_path:
            return

        if is_dmx_file(file_path):
            try:
                _, animation_clips = load_dmx(file_path)
            except Exception as error:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    f"Could not load DMX file.\n{error}",
                )
                return

            if not animation_clips:
                QMessageBox.warning(
                    self,
                    "No Sequence Data",
                    "This DMX file does not contain any animation clips.",
                )
                return

            self._set_animation_clips(animation_clips)

            self._try_auto_flex_info(file_path)

            if self.flex_info is not None:
                self.apply_flex_info(self.flex_info, self.flex_info_override)

            self._maybe_autoplay()
            self.statusBar().showMessage(
                f"Loaded {len(animation_clips)} DMX clips from {file_path}"
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

        self._set_animation_clips([])
        self.viewport.set_skeletal_animation_model(parsed)

        self._try_auto_flex_info(file_path)

        if self.flex_info is not None:
            self.apply_flex_info(self.flex_info, self.flex_info_override)

        self.stop_playback()
        self.update_animation_ui()
        self._maybe_autoplay()

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

    def set_materials_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Materials Directory",
            "",
        )
        if dir_path:
            self.viewport.renderer.add_material_directory(dir_path)
            self.viewport.update()
            self.statusBar().showMessage(f"Materials folder: {dir_path}")

    def _auto_add_materials_root(self, file_path):
        norm = os.path.normpath(file_path)
        parts = norm.split(os.sep)

        # Check for standard <root>/models/... directory layout
        for i in range(len(parts) - 1, 0, -1):
            if parts[i].lower() == "models":
                root = os.sep.join(parts[:i])
                candidate = os.path.join(root, "materials")
                if os.path.isdir(candidate):
                    self.viewport.renderer.add_material_directory(candidate)
                    return candidate

        # Check adjacent materials/ folder
        direct_candidate = os.path.join(os.path.dirname(file_path), "materials")
        if os.path.isdir(direct_candidate):
            self.viewport.renderer.add_material_directory(direct_candidate)
            return direct_candidate

        # Walk up from the model's folder: a materials/ dir next to it
        # or a few levels above (e.g. temp/materials for a model in
        # temp/model) is the model's material set.
        parent = os.path.dirname(os.path.abspath(file_path))
        for _ in range(6):
            parent = os.path.dirname(parent)
            candidate = os.path.join(parent, "materials")
            if os.path.isdir(candidate):
                self.viewport.renderer.add_material_directory(candidate)
                return candidate

        return None

    def show_materials(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Materials")
        dialog.resize(780, 460)
        layout = QVBoxLayout(dialog)

        summary = QLabel("")
        summary.setWordWrap(True)
        layout.addWidget(summary)

        table = QTableWidget(0, 3, dialog)
        table.setHorizontalHeaderLabels(["Material", "Texture", "Status"])
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(table, 1)

        def fill():
            rows = self.viewport.renderer.material_report()
            table.setRowCount(len(rows))

            for row, entry in enumerate(rows):
                name_item = QTableWidgetItem(entry["name"])
                texture_item = QTableWidgetItem(entry["texture"] or "")
                status_item = QTableWidgetItem(entry["status"])

                if entry["detail"]:
                    name_item.setToolTip(entry["detail"])
                if entry["texture"]:
                    texture_item.setToolTip(entry["texture"])
                if entry["status"] == "missing":
                    status_item.setForeground(QColor("#e06c75"))
                elif entry["status"] in ("ok", "embedded"):
                    status_item.setForeground(QColor("#98c379"))

                table.setItem(row, 0, name_item)
                table.setItem(row, 1, texture_item)
                table.setItem(row, 2, status_item)

            missing = sum(1 for entry in rows if entry["status"] == "missing")
            if rows:
                summary.setText(
                    f"{len(rows)} materials, {missing} missing. "
                    "Hover a material for its VMT or VMAT path."
                )
            else:
                summary.setText("No materials loaded.")
            table.resizeRowsToContents()

        fill()

        buttons = QHBoxLayout()
        folder_button = QPushButton("Set Materials Folder...")
        close_button = QPushButton("Close")

        def choose_folder():
            dir_path = QFileDialog.getExistingDirectory(
                dialog, "Select Materials Directory", ""
            )
            if dir_path:
                self.viewport.renderer.add_material_directory(dir_path)
                self.viewport.update()
                fill()
                self.statusBar().showMessage(f"Materials folder: {dir_path}")

        folder_button.clicked.connect(choose_folder)
        close_button.clicked.connect(dialog.accept)
        buttons.addWidget(folder_button)
        buttons.addStretch(1)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        dialog.exec()

    def _clear_bodygroup_widgets(self):
        while self.bodygroup_layout.count():
            item = self.bodygroup_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _set_bodygroups(self, model):
        self._clear_bodygroup_widgets()
        self._bodygroup_model = model
        self._bodygroup_selection = []

        metadata = (getattr(model, "metadata", None) or {}) if model else {}
        groups = metadata.get("bodygroups") or []
        if not groups:
            self.bodygroup_scroll.setVisible(False)
            return

        saved = metadata.get("bodygroup_selection") or []
        added = False

        for group_index, group in enumerate(groups):
            options = group.get("options") or []
            name = group.get("name") or f"Body {group_index + 1}"

            if not options:
                self._bodygroup_selection.append(0)
                continue

            label = QLabel(name)
            label.setToolTip(f"Bodygroup: {name}")

            combo = QComboBox()
            combo.setMaximumWidth(190)
            for option_index, option_name in enumerate(options):
                combo.addItem(option_name or f"Option {option_index + 1}", option_index)
            combo.setToolTip("Options: " + ", ".join(options))

            current = saved[group_index] if group_index < len(saved) else 0
            try:
                current = int(current)
            except (TypeError, ValueError):
                current = 0
            current = max(0, min(current, max(0, len(options) - 1)))
            combo.setCurrentIndex(current)
            self._bodygroup_selection.append(current)
            combo.currentIndexChanged.connect(
                lambda choice, index=group_index: self.bodygroup_changed(index, choice)
            )

            self.bodygroup_layout.addWidget(label)
            self.bodygroup_layout.addWidget(combo)
            added = True

        if added:
            self.bodygroup_layout.activate()
            self.bodygroup_bar.adjustSize()
        self.bodygroup_scroll.setVisible(added)

    def bodygroup_changed(self, group_index, choice):
        if self._bodygroup_model is None:
            return
        if not 0 <= group_index < len(self._bodygroup_selection):
            return

        self._bodygroup_selection[group_index] = int(choice)

        from viewer.smd_parser import compose_bodygroup_model

        composed = compose_bodygroup_model(
            self._bodygroup_model, self._bodygroup_selection
        )
        if composed is self._bodygroup_model:
            return

        self.viewport.set_model(composed, preserve_view=True)
        self.viewport.update()
        self.update_model_info()
        self.statusBar().showMessage(
            f"Bodygroup {group_index + 1} set to option {int(choice) + 1}"
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
        self._set_animation_clips([])
        self.sequence_slider.setEnabled(False)
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

    def toggle_grid(self):
        self.viewport.set_show_grid(self.show_grid_action.isChecked())

    def toggle_axes(self):
        self.viewport.set_show_axes(self.show_axes_action.isChecked())

    def toggle_show_controls(self):
        visible = self.show_controls_action.isChecked()
        self.controls_frame.setVisible(visible)
        settings = QSettings("SourceModelViewer", "Settings")
        settings.setValue("window/show_controls", visible)

    def toggle_wireframe(self):
        self.viewport.set_wireframe(self.wireframe_action.isChecked())

    def toggle_skeleton(self):
        self.viewport.set_show_skeleton(self.skeleton_action.isChecked())

    def toggle_solid_view(self):
        if self.solid_view_action.isChecked():
            self.uv_checker_action.blockSignals(True)
            self.uv_checker_action.setChecked(False)
            self.uv_checker_action.blockSignals(False)
            self.viewport.set_model_view_mode("solid")
        else:
            self.viewport.set_model_view_mode("textured")

    def toggle_uv_checker(self):
        if self.uv_checker_action.isChecked():
            self.solid_view_action.blockSignals(True)
            self.solid_view_action.setChecked(False)
            self.solid_view_action.blockSignals(False)
            self.viewport.set_model_view_mode("uv_checker")
        else:
            self.viewport.set_model_view_mode("textured")

    def _apply_view_mode_actions(self, mode):
        self.solid_view_action.blockSignals(True)
        self.uv_checker_action.blockSignals(True)
        self.solid_view_action.setChecked(mode == "solid")
        self.uv_checker_action.setChecked(mode == "uv_checker")
        self.solid_view_action.blockSignals(False)
        self.uv_checker_action.blockSignals(False)

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

    def _step_uses_sequence(self):
        if self._sequence_playback_active():
            return True
        if self.viewport.frame_count() > 1:
            return False
        return self.viewport.has_skeletal_animation()

    def _sync_sequence_slider(self, frame, count):
        progress = frame / (count - 1) if count > 1 else 0.0
        self.sequence_slider.blockSignals(True)
        self.sequence_slider.setValue(int(round(progress * 1000)))
        self.sequence_slider.blockSignals(False)

    def step_frame(self, delta):
        self.stop_playback()
        if self._step_uses_sequence():
            count = self.viewport.skeletal_frame_count()
            if count <= 0:
                return
            current = self.viewport.skeletal_current_frame()
            new_frame = max(0.0, min(float(current) + delta, count - 1))
            self.viewport.set_skeletal_frame(new_frame)
            self._sync_sequence_slider(new_frame, count)
        else:
            count = self.viewport.frame_count()
            if count <= 1:
                return
            step = max(1, int(round(1000.0 / float(count - 1))))
            value = self.progress_slider.value()
            self.progress_slider.setValue(max(0, min(1000, value + delta * step)))
        self.update_frame_label()

    def set_frame_boundary(self, first):
        self.stop_playback()
        if self._step_uses_sequence():
            count = self.viewport.skeletal_frame_count()
            if count <= 0:
                return
            new_frame = 0.0 if first else float(count - 1)
            self.viewport.set_skeletal_frame(new_frame)
            self._sync_sequence_slider(new_frame, count)
        else:
            if self.viewport.frame_count() <= 1:
                return
            self.progress_slider.setValue(0 if first else 1000)
        self.update_frame_label()

    def step_frame_back(self):
        self.step_frame(-1)

    def step_frame_forward(self):
        self.step_frame(1)

    def goto_first_frame(self):
        self.set_frame_boundary(True)

    def goto_last_frame(self):
        self.set_frame_boundary(False)

    def update_clip_info(self):
        index = self.clip_combo.currentIndex()
        if index < 0 or index >= len(self._clip_models):
            self.clip_info_label.setText("")
            return
        _, clip_model = self._clip_models[index]
        metadata = getattr(clip_model, "metadata", {}) or {}
        frames = len(clip_model.frames) if clip_model.frames else 0
        parts = [f"{frames}f"]
        fps = metadata.get("frame_rate")
        try:
            if fps is not None:
                parts.append(f"{float(fps):g}fps")
        except (TypeError, ValueError):
            pass
        duration = metadata.get("duration")
        try:
            if duration is not None:
                parts.append(f"{float(duration):.2f}s")
        except (TypeError, ValueError):
            pass
        if "looping" in metadata:
            parts.append("loop" if metadata.get("looping") else "once")
        self.clip_info_label.setText("  ".join(parts))

    def update_model_info(self):
        model = self.viewport.renderer.model
        if model is None:
            self.model_info_label.setText("")
            return
        triangles = len(getattr(model, "triangles", []) or [])
        bones = len(getattr(model, "bones", []) or [])
        materials = len(getattr(model, "materials", []) or [])
        clips = len(self._clip_models)
        self.model_info_label.setText(
            f"{triangles} triangles  |  {bones} bones  |  "
            f"{materials} materials  |  {clips} clips"
        )

    def _maybe_autoplay(self):
        settings = QSettings("SourceModelViewer", "Settings")
        if settings.value("defaults/autoplay", False, type=bool) and self.play_button.isEnabled():
            self.start_playback()

    def advance_frame(self):
        if self._sequence_playback_active():
            count = self.viewport.skeletal_frame_count()
            if count <= 1:
                self.stop_playback()
                return
            current = self.viewport.skeletal_current_frame()
            new_frame = current + 1.0
            if new_frame >= count:
                if not self.loop_check.isChecked():
                    self.viewport.set_skeletal_frame(count - 1)
                    self._sync_sequence_slider(count - 1, count)
                    if self.driver_check.isChecked():
                        self.sync_sliders_from_renderer()
                    self.update_frame_label()
                    self.stop_playback()
                    return
                new_frame = 0.0
            self.viewport.set_skeletal_frame(new_frame)

            # Update sequence slider
            self._sync_sequence_slider(new_frame, count)

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
            if not self.loop_check.isChecked():
                self.progress_slider.setValue(1000)
                self.stop_playback()
                return
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
        step_enabled = (vta_count > 1) or (has_sequence and sequence_count > 1)
        for step_button in (
            self.first_frame_button,
            self.prev_frame_button,
            self.next_frame_button,
            self.last_frame_button,
        ):
            step_button.setEnabled(step_enabled)
        self.loop_check.setEnabled(play_enabled)

        # Update clip combo and sequence slider
        clips = self.viewport.clip_names()
        if clips:
            self.clip_combo.setEnabled(True)
            current_index = self.viewport.current_clip_index()
            if 0 <= current_index < self.clip_combo.count():
                if self.clip_combo.currentIndex() != current_index:
                    self.clip_combo.setCurrentIndex(current_index)
        else:
            self.clip_combo.setEnabled(False)

        self.sequence_slider.setEnabled(has_sequence)

        if vta_count > 0 or has_sequence or has_sequence_source:
            self.update_frame_label()
        elif self.animation_targets:
            self.frame_label.setText("No reference mesh")
        else:
            self.frame_label.setText("No animation")

        self.update_clip_info()
        self.update_model_info()

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

    def clip_changed(self, index):
        if index >= 0 and self.viewport:
            self.viewport.set_animation_clip(index)
            self._update_blend_selector()
            self.sequence_slider.setValue(0)
            self.stop_playback()
            self.update_animation_ui()
            self._maybe_autoplay()

    def _update_blend_selector(self):
        self.blend_combo.blockSignals(True)
        self.blend_combo.clear()
        self._blend_models = []

        model = None
        index = self.clip_combo.currentIndex()
        if 0 <= index < len(self._clip_models):
            model = self._clip_models[index][1]

        metadata = (getattr(model, "metadata", None) or {}) if model else {}
        blends = metadata.get("sequence_blends") or []
        self._blend_models = blends

        for blend_index, blend in enumerate(blends):
            frames = len(getattr(blend.get("model"), "frames", []) or [])
            self.blend_combo.addItem(
                f"{blend_index + 1} of {len(blends)} ({frames}f)", blend_index
            )
        self.blend_combo.setCurrentIndex(0)
        self.blend_combo.blockSignals(False)
        self.blend_bar.setVisible(len(blends) > 1)

    def blend_changed(self, index):
        if not 0 <= index < len(self._blend_models):
            return

        model = self._blend_models[index].get("model")
        if model is None:
            return

        self.viewport.set_skeletal_animation_model(model, preserve_frame=True)
        self.update_frame_label()
        self.statusBar().showMessage(
            f"Blend {index + 1} of {len(self._blend_models)}"
        )

    def sequence_progress_changed(self, value):
        progress = value / 1000.0
        self.viewport.set_skeletal_progress(progress)
        self.update_frame_label()

    def show_controls(self):
        show_help_dialog(self, "Viewer Controls", controls_html())

    def show_animation_guide(self):
        show_help_dialog(self, "Animation Guide", animation_html())

    def show_formats(self):
        show_help_dialog(self, "Supported Formats", formats_html())

    def show_about(self):
        show_help_dialog(self, "About", about_html(), width=480, height=440)

    def apply_settings(self):
        settings = QSettings("SourceModelViewer", "Settings")

        # Viewport settings
        bg_raw = settings.value("viewport/bg_color", "#1a1a1c")
        bg_hex = str(bg_raw) if bg_raw else "#1a1a1c"
        qcolor = QColor(bg_hex)
        if not qcolor.isValid():
            qcolor = QColor("#1a1a1c")

        self.viewport.renderer.set_background_color(
            qcolor.redF(), qcolor.greenF(), qcolor.blueF(), 1.0
        )

        skybox_name = settings.value("viewport/skybox", None, type=str)
        if skybox_name is None:
            from viewer.skybox import discover_skyboxes

            entries = discover_skyboxes()
            skybox_name = entries[0].name if entries else ""
        self.viewport.set_skybox(skybox_name)
        skybox_error = getattr(self.viewport.renderer, "skybox_error", "")
        if skybox_name and skybox_error:
            self.statusBar().showMessage(
                f"Skybox {skybox_name!r} could not be loaded: {skybox_error}"
            )

        show_grid = settings.value("viewport/show_grid", True, type=bool)
        self.viewport.renderer.set_show_grid(show_grid)
        self.show_grid_action.setChecked(show_grid)

        show_axes = settings.value("viewport/show_axes", True, type=bool)
        self.viewport.renderer.set_show_axes(show_axes)
        self.show_axes_action.setChecked(show_axes)

        fov = settings.value("viewport/fov", 45, type=int)
        self.viewport.renderer.camera.fov = fov

        # Default materials folder if set
        default_mat = settings.value("materials/default_dir", "", type=str)
        if default_mat and os.path.isdir(default_mat):
            self.viewport.renderer.add_material_directory(default_mat)

        # Defaults
        default_fps = settings.value("defaults/fps", 30, type=int)
        self.fps_spin.setValue(default_fps)

        culling = settings.value("defaults/culling", False, type=bool)
        self.culling_action.setChecked(culling)
        self.viewport.set_backface_culling(culling)

        proximity = settings.value("defaults/proximity_skin", False, type=bool)
        self.proximity_skin_action.setChecked(proximity)
        self.viewport.set_proximity_skin(proximity)

        view_mode = str(settings.value("model/view_mode", "textured", type=str))
        if view_mode not in ("textured", "solid", "uv_checker"):
            view_mode = "textured"
        self.viewport.set_model_view_mode(view_mode)
        self._apply_view_mode_actions(view_mode)

        wireframe = settings.value("model/wireframe", False, type=bool)
        self.wireframe_action.setChecked(wireframe)
        self.viewport.set_wireframe(wireframe)

        show_skeleton = settings.value("model/show_skeleton", False, type=bool)
        self.skeleton_action.setChecked(show_skeleton)
        self.viewport.set_show_skeleton(show_skeleton)

        # Playback defaults
        loop_default = settings.value("defaults/loop", True, type=bool)
        self.loop_check.setChecked(loop_default)

        shape_mode_default = settings.value("defaults/shape_key_mode", True, type=bool)
        self.shape_mode_check.blockSignals(True)
        self.shape_mode_check.setChecked(shape_mode_default)
        self.shape_mode_check.blockSignals(False)
        self.viewport.set_animation_mode("shape" if shape_mode_default else "sequence")

        # Window and appearance
        theme = settings.value("window/theme", 0, type=int)
        apply_app_icon(self, theme=theme)

        show_controls = settings.value("window/show_controls", True, type=bool)
        self.show_controls_action.setChecked(show_controls)
        self.controls_frame.setVisible(show_controls)

        # Texture filtering was saved by the dialog but never applied before
        filter_mode = settings.value("materials/filter_mode", 0, type=int)
        self.viewport.renderer.set_texture_filter_mode(filter_mode)

        self.viewport.update()

    def open_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.apply_settings()
            # apply_settings() may already have shown a skybox error;
            # do not overwrite it with the generic message.
            skybox_error = getattr(self.viewport.renderer, "skybox_error", "")
            if not skybox_error:
                self.statusBar().showMessage("Settings saved")