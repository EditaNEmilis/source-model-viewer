import os
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


SETTINGS_STYLE = """
    QDialog {
        background-color: #17181d;
        color: #d6d8de;
    }
    QTabWidget::pane {
        border: 1px solid #2a2c34;
        background-color: #17181d;
    }
    QTabBar::tab {
        background-color: #20222a;
        color: #9da3b4;
        padding: 8px 16px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #2a2c34;
        color: #f2f3f7;
    }
    QGroupBox {
        font-weight: bold;
        border: 1px solid #2a2c34;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 10px;
        color: #a9b1e8;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }
    QLabel {
        color: #d6d8de;
    }
    QLineEdit, QSpinBox, QComboBox {
        background-color: #23252d;
        color: #d6d8de;
        border: 1px solid #3a3d47;
        border-radius: 4px;
        padding: 4px 8px;
    }
    QPushButton {
        background-color: #2a2c34;
        color: #d6d8de;
        border: 1px solid #3a3d47;
        border-radius: 4px;
        padding: 6px 16px;
    }
    QPushButton:hover {
        background-color: #343742;
    }
    QCheckBox {
        color: #d6d8de;
    }
"""


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 420)
        self.setStyleSheet(SETTINGS_STYLE)

        self.settings = QSettings("SourceModelViewer", "Settings")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        tabs = QTabWidget()

        # Tab 1: Materials & Textures
        mat_tab = QWidget()
        mat_layout = QVBoxLayout(mat_tab)

        mat_group = QGroupBox("Material Resolution")
        mat_form = QFormLayout(mat_group)

        self.auto_mount_check = QCheckBox("Auto-mount materials folder from model path")
        self.auto_mount_check.setChecked(
            self.settings.value("materials/auto_mount", True, type=bool)
        )
        mat_form.addRow(self.auto_mount_check)

        self.default_mat_edit = QLineEdit()
        self.default_mat_edit.setText(
            self.settings.value("materials/default_dir", "", type=str)
        )
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_default_mat)

        mat_path_row = QHBoxLayout()
        mat_path_row.addWidget(self.default_mat_edit, 1)
        mat_path_row.addWidget(browse_btn, 0)
        mat_form.addRow("Default Materials Folder:", mat_path_row)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Trilinear (Smooth)", "Bilinear", "Nearest / Point (Pixelated)"])
        self.filter_combo.setCurrentIndex(
            self.settings.value("materials/filter_mode", 0, type=int)
        )
        mat_form.addRow("Texture Filtering:", self.filter_combo)

        mat_layout.addWidget(mat_group)
        mat_layout.addStretch()
        tabs.addTab(mat_tab, "Materials")

        # Tab 2: Viewport & Appearance
        view_tab = QWidget()
        view_layout = QVBoxLayout(view_tab)

        view_group = QGroupBox("Viewport Display")
        view_form = QFormLayout(view_group)

        self.show_grid_check = QCheckBox("Show Ground Grid")
        self.show_grid_check.setChecked(
            self.settings.value("viewport/show_grid", True, type=bool)
        )
        view_form.addRow(self.show_grid_check)

        self.bg_color_btn = QPushButton()
        self.bg_color = QColor(
            self.settings.value("viewport/bg_color", "#1a1a1c", type=str)
        )
        self._update_color_btn()
        self.bg_color_btn.clicked.connect(self._pick_bg_color)
        view_form.addRow("Background Color:", self.bg_color_btn)

        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(10, 120)
        self.fov_spin.setValue(self.settings.value("viewport/fov", 45, type=int))
        self.fov_spin.setSuffix("°")
        view_form.addRow("Camera FOV:", self.fov_spin)

        view_layout.addWidget(view_group)
        view_layout.addStretch()
        tabs.addTab(view_tab, "Viewport")

        # Tab 3: Defaults & Behavior
        anim_tab = QWidget()
        anim_layout = QVBoxLayout(anim_tab)

        anim_group = QGroupBox("Defaults on Model Load")
        anim_form = QFormLayout(anim_group)

        self.default_fps_spin = QSpinBox()
        self.default_fps_spin.setRange(1, 120)
        self.default_fps_spin.setValue(
            self.settings.value("defaults/fps", 30, type=int)
        )
        self.default_fps_spin.setSuffix(" FPS")
        anim_form.addRow("Default FPS:", self.default_fps_spin)

        self.culling_default_check = QCheckBox("Enable Backface Culling by default")
        self.culling_default_check.setChecked(
            self.settings.value("defaults/culling", False, type=bool)
        )
        anim_form.addRow(self.culling_default_check)

        self.proximity_default_check = QCheckBox("Enable Proximity Skinning by default")
        self.proximity_default_check.setChecked(
            self.settings.value("defaults/proximity_skin", False, type=bool)
        )
        anim_form.addRow(self.proximity_default_check)

        anim_layout.addWidget(anim_group)
        anim_layout.addStretch()
        tabs.addTab(anim_tab, "Defaults")

        layout.addWidget(tabs, 1)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self._save_and_accept)

        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _browse_default_mat(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Default Materials Folder", self.default_mat_edit.text()
        )
        if dir_path:
            self.default_mat_edit.setText(dir_path)

    def _pick_bg_color(self):
        color = QColorDialog.getColor(self.bg_color, self, "Select Background Color")
        if color.isValid():
            self.bg_color = color
            self._update_color_btn()

    def _update_color_btn(self):
        self.bg_color_btn.setText(self.bg_color.name())
        self.bg_color_btn.setStyleSheet(
            f"background-color: {self.bg_color.name()}; color: {'#000000' if self.bg_color.lightness() > 128 else '#ffffff'};"
        )

    def _save_and_accept(self):
        self.settings.setValue("materials/auto_mount", self.auto_mount_check.isChecked())
        self.settings.setValue("materials/default_dir", self.default_mat_edit.text())
        self.settings.setValue("materials/filter_mode", self.filter_combo.currentIndex())

        self.settings.setValue("viewport/show_grid", self.show_grid_check.isChecked())
        self.settings.setValue("viewport/bg_color", self.bg_color.name())
        self.settings.setValue("viewport/fov", self.fov_spin.value())

        self.settings.setValue("defaults/fps", self.default_fps_spin.value())
        self.settings.setValue("defaults/culling", self.culling_default_check.isChecked())
        self.settings.setValue("defaults/proximity_skin", self.proximity_default_check.isChecked())

        self.settings.sync()
        self.accept()