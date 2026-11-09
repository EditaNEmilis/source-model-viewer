"""Application icon loading from viewer/assets/icos/.

Two variants ship with the app:

* ``favicon.ico`` - the normal (colored) icon, used with the dark theme.
* ``faviconaltblack.ico`` - a black silhouette, used with the light
  theme so the icon stays visible on a light title bar.

Each variant may also provide ``_32`` and ``_64`` sized files, which
are added to the same QIcon so Qt can pick the closest size.

Paths resolve from this file, so the app works from any working
directory. Missing files degrade to an empty icon instead of failing.
"""

import os

from PySide6.QtCore import QSettings
from PySide6.QtGui import QIcon

from viewer.ui_style import LIGHT_THEME

ICONS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "icos"
)
DEFAULT_ICON_STEM = "favicon"
LIGHT_THEME_ICON_STEM = "faviconaltblack"
_ICON_SUFFIXES = ("", "_32", "_64")


def _current_theme():
    try:
        settings = QSettings("SourceModelViewer", "Settings")
        return settings.value("window/theme", 0, type=int)
    except (TypeError, ValueError):
        return 0


def _icon_files(stem):
    return [
        os.path.join(ICONS_DIR, stem + suffix + ".ico")
        for suffix in _ICON_SUFFIXES
    ]


def icon_paths(theme=None):
    """Existing icon file paths for ``theme`` (defaults to saved setting)."""
    if theme is None:
        theme = _current_theme()
    stem = LIGHT_THEME_ICON_STEM if theme == LIGHT_THEME else DEFAULT_ICON_STEM
    paths = [p for p in _icon_files(stem) if os.path.isfile(p)]
    if paths:
        return paths
    other = DEFAULT_ICON_STEM if stem == LIGHT_THEME_ICON_STEM else LIGHT_THEME_ICON_STEM
    return [p for p in _icon_files(other) if os.path.isfile(p)]


def app_icon_path(theme=None):
    """First icon path for ``theme``, or "" when no icon file exists."""
    paths = icon_paths(theme)
    return paths[0] if paths else ""


def get_app_icon(theme=None):
    """QIcon for ``theme``, or an empty QIcon when no icon file exists."""
    icon = QIcon()
    for path in icon_paths(theme):
        icon.addFile(path)
    return icon


def apply_app_icon(target, theme=None):
    """Set the app icon on a QApplication or a window widget."""
    if target is None:
        return
    target.setWindowIcon(get_app_icon(theme))
