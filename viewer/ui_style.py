"""Shared dialog styling for the settings and help dialogs.

Both dialogs use the same widget set, so the stylesheet lives here instead of
being duplicated. ``dialog_stylesheet`` returns the full sheet for a theme
index stored in ``window/theme`` (0 dark, 1 light).
"""

DARK_THEME = 0
LIGHT_THEME = 1

_DARK = """
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
    QTextBrowser {
        background-color: #17181d;
        color: #d6d8de;
        border: none;
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
    QPushButton:pressed {
        background-color: #22242b;
    }
    QCheckBox {
        color: #d6d8de;
    }
"""

_LIGHT = """
    QDialog {
        background-color: #f2f3f6;
        color: #20222a;
    }
    QTabWidget::pane {
        border: 1px solid #c8ccd6;
        background-color: #f2f3f6;
    }
    QTabBar::tab {
        background-color: #e2e5ec;
        color: #4a5060;
        padding: 8px 16px;
        margin-right: 2px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #f2f3f6;
        color: #20222a;
    }
    QGroupBox {
        font-weight: bold;
        border: 1px solid #c8ccd6;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 10px;
        color: #3f4ea8;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
    }
    QLabel {
        color: #20222a;
    }
    QTextBrowser {
        background-color: #f2f3f6;
        color: #20222a;
        border: none;
    }
    QLineEdit, QSpinBox, QComboBox {
        background-color: #ffffff;
        color: #20222a;
        border: 1px solid #b9bdc9;
        border-radius: 4px;
        padding: 4px 8px;
    }
    QPushButton {
        background-color: #e2e5ec;
        color: #20222a;
        border: 1px solid #b9bdc9;
        border-radius: 4px;
        padding: 6px 16px;
    }
    QPushButton:hover {
        background-color: #d5d9e3;
    }
    QPushButton:pressed {
        background-color: #c8ccd6;
    }
    QCheckBox {
        color: #20222a;
    }
"""


def dialog_stylesheet(theme=DARK_THEME):
    """Return the stylesheet for ``theme`` (0 dark, 1 light).

    Unknown values fall back to dark so a stale setting never breaks a dialog.
    """
    try:
        theme = int(theme)
    except (TypeError, ValueError):
        theme = DARK_THEME
    return _LIGHT if theme == LIGHT_THEME else _DARK
