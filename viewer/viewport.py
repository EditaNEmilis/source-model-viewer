from PySide6.QtCore import QEvent, Qt
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from viewer.renderer import Renderer
from viewer.smd_parser import SmdModel


class Viewport(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.renderer = Renderer()
        self.last_mouse_position = None
        self.model_move_mode = False

        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.setCursor(Qt.ArrowCursor)

    def initializeGL(self):
        self.renderer.initialize()

    def resizeGL(self, width, height):
        device_pixel_ratio = self.devicePixelRatioF()

        physical_width = int(width * device_pixel_ratio)
        physical_height = int(height * device_pixel_ratio)

        self.renderer.resize(physical_width, physical_height)

    def paintGL(self):
        self.renderer.paint()

    def set_model(self, model: SmdModel):
        self.renderer.set_model(model)
        self.update()

    def set_animation_targets(self, targets):
        self.renderer.set_animation_targets(targets)
        self.update()

    def set_animation_mode(self, mode):
        self.renderer.set_animation_mode(mode)
        self.update()

    def set_match_ids_by_position(self, enabled):
        self.renderer.set_match_ids_by_position(enabled)
        self.update()

    def set_progress(self, progress):
        self.renderer.set_progress(progress)
        self.update()

    def set_intensity(self, intensity):
        self.renderer.set_intensity(intensity)
        self.update()

    def set_skeletal_animation_model(self, model):
        self.renderer.set_skeletal_animation_model(model)
        self.update()

    def clear_skeletal_animation(self):
        self.renderer.clear_skeletal_animation()
        self.update()

    def set_skeletal_enabled(self, enabled):
        self.renderer.set_skeletal_enabled(enabled)
        self.update()

    def set_driver_enabled(self, enabled):
        self.renderer.set_driver_enabled(enabled)
        self.update()

    def set_driver_bone_name(self, name):
        self.renderer.set_driver_bone_name(name)
        self.update()

    def set_driver_progress_axis(self, axis):
        self.renderer.set_driver_progress_axis(axis)
        self.update()

    def set_skeletal_frame(self, frame):
        self.renderer.set_skeletal_frame(frame)
        self.update()
        
    def skeletal_rig_bone_count(self):
        return self.renderer.skeletal_rig_bone_count()

    def skeletal_anim_bone_count(self):
        return self.renderer.skeletal_anim_bone_count()

    def skeletal_mapped_count(self):
        return self.renderer.skeletal_mapped_count()

    def skeletal_driven_count(self):
        return self.renderer.skeletal_driven_count()

    def skeletal_unmapped_names(self):
        return self.renderer.skeletal_unmapped_names()
        
    def set_proximity_skin(self, enabled):
        self.renderer.set_proximity_skin(enabled)
        self.update()

    def skeletal_weight_stats(self):
        return self.renderer.skeletal_weight_stats()
        
    def set_backface_culling(self, enabled):
        self.renderer.set_backface_culling(enabled)
        self.update()

    def animation_mode(self):
        return self.renderer.animation_mode()

    def match_ids_by_position(self):
        return self.renderer.match_ids_by_position()

    def frame_count(self):
        return self.renderer.frame_count()

    def current_frame(self):
        return self.renderer.current_frame()

    def progress(self):
        return self.renderer.progress()

    def intensity(self):
        return self.renderer.intensity()

    def skeletal_frame_count(self):
        return self.renderer.skeletal_frame_count()

    def skeletal_current_frame(self):
        return self.renderer.skeletal_current_frame()

    def has_skeletal_animation(self):
        return self.renderer.has_skeletal_animation()

    def has_skeletal_source(self):
        return self.renderer.has_skeletal_source()

    def has_rig(self):
        return self.renderer.has_rig()

    def skeletal_enabled(self):
        return self.renderer.skeletal_enabled()

    def driver_enabled(self):
        return self.renderer.driver_enabled()

    def has_driver_bone(self):
        return self.renderer.has_driver_bone()

    def driver_bone_name(self):
        return self.renderer.driver_bone_name()

    def driver_progress_axis(self):
        return self.renderer.driver_progress_axis()

    def animation_name(self, index=None):
        return self.renderer.animation_name(index)

    def animation_ignored_count(self):
        return self.renderer.animation_ignored_count()

    def animation_max_vertex_id(self):
        return self.renderer.animation_max_vertex_id()

    def animation_reference_vertex_count(self):
        return self.renderer.animation_reference_vertex_count()

    def animation_first_target_count(self):
        return self.renderer.animation_first_target_count()

    def animation_matched_count(self):
        return self.renderer.animation_matched_count()

    def animation_matched_vertex_count(self):
        return self.renderer.animation_matched_vertex_count()

    def set_frame(self, frame):
        self.renderer.set_frame(frame)
        self.update()

    def reset_view(self):
        self.renderer.reset_view()
        self.update()

    def reset_camera(self):
        self.renderer.reset_camera()
        self.update()

    def reset_model_position(self):
        self.renderer.reset_model_transform()
        self.update()

    def reset_all(self):
        self.renderer.reset_all()
        self.update()

    def event(self, event):
        if event.type() == QEvent.Type.ShortcutOverride:
            if event.key() == Qt.Key_Alt or (event.modifiers() & Qt.AltModifier):
                event.accept()
                return True

        return super().event(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Alt:
            event.accept()
            return

        if event.key() == Qt.Key_M and not event.isAutoRepeat():
            self.model_move_mode = not self.model_move_mode

            if self.model_move_mode:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

            self.update()
            event.accept()
            return

        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        self.setFocus(Qt.MouseFocusReason)

        if event.button() in (
            Qt.LeftButton,
            Qt.RightButton,
            Qt.MiddleButton,
        ):
            self.last_mouse_position = event.position().toPoint()

        event.accept()

    def mouseMoveEvent(self, event):
        if self.last_mouse_position is None:
            return

        position = event.position().toPoint()

        dx = position.x() - self.last_mouse_position.x()
        dy = position.y() - self.last_mouse_position.y()

        buttons = event.buttons()
        modifiers = event.modifiers()

        model_move_modifier = modifiers & (Qt.ControlModifier | Qt.AltModifier)

        if self.model_move_mode and (buttons & Qt.LeftButton):
            self.renderer.move_model(dx, dy)

        elif model_move_modifier and (buttons & Qt.LeftButton):
            self.renderer.move_model(dx, dy)

        elif (modifiers & Qt.ShiftModifier) and (buttons & (Qt.LeftButton | Qt.MiddleButton)):
            self.renderer.camera.pan(dx, dy)

        elif buttons & Qt.LeftButton:
            self.renderer.camera.rotate(dx, dy)

        elif buttons & Qt.MiddleButton:
            self.renderer.camera.pan(dx, dy)

        elif buttons & Qt.RightButton:
            self.renderer.camera.zoom_drag(dy)

        self.last_mouse_position = position

        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        self.last_mouse_position = None
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()

        if delta == 0:
            delta = event.pixelDelta().y()

        if delta != 0:
            self.renderer.camera.zoom(delta)
            self.update()

        event.accept()

    def contextMenuEvent(self, event):
        event.ignore()