from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QPushButton, QTextBrowser, QVBoxLayout

from viewer import __version__


DIALOG_STYLE = """
    QDialog {
        background-color: #17181d;
    }
    QTextBrowser {
        background-color: #17181d;
        color: #d6d8de;
        border: none;
    }
    QPushButton {
        background-color: #2a2c34;
        color: #d6d8de;
        border: 1px solid #3a3d47;
        border-radius: 4px;
        padding: 6px 20px;
    }
    QPushButton:hover {
        background-color: #343742;
    }
    QPushButton:pressed {
        background-color: #22242b;
    }
"""

_PAGE_STYLE = (
    "body { font-family: 'Segoe UI', 'Noto Sans', sans-serif; "
    "font-size: 13px; color: #d6d8de; line-height: 150%; }"
    "h1 { font-size: 20px; color: #f2f3f7; margin: 0 0 4px 0; }"
    "h2 { font-size: 12px; color: #a9b1e8; margin: 18px 0 6px 0; "
    "text-transform: uppercase; letter-spacing: 1.5px; }"
    "p { margin: 4px 0 10px 0; }"
    "code { font-family: 'Cascadia Mono', 'Consolas', monospace; "
    "font-size: 12px; color: #8fd0ff; background: #23252d; padding: 1px 5px; }"
    "table { border-collapse: collapse; margin: 6px 0 12px 0; }"
    "td { padding: 3px 16px 3px 0; vertical-align: top; }"
    "td.k { color: #f0c674; font-family: 'Cascadia Mono', 'Consolas', monospace; "
    "font-size: 12px; white-space: nowrap; }"
)


def _page(title, body):
    return (
        "<html><head><style>" + _PAGE_STYLE + "</style></head>"
        "<body><h1>" + title + "</h1>" + body + "</body></html>"
    )


def show_help_dialog(parent, title, html, width=560, height=540):
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.resize(width, height)
    dialog.setStyleSheet(DIALOG_STYLE)

    layout = QVBoxLayout(dialog)
    layout.setContentsMargins(16, 16, 16, 14)
    layout.setSpacing(10)

    browser = QTextBrowser()
    browser.setReadOnly(True)
    browser.setOpenExternalLinks(False)
    browser.setHtml(html)

    close_button = QPushButton("Close")
    close_button.clicked.connect(dialog.accept)

    layout.addWidget(browser)
    layout.addWidget(close_button, 0, Qt.AlignRight)

    dialog.exec()


def controls_html():
    rows = [
        ("Left drag", "Rotate camera"),
        ("Middle drag", "Pan camera"),
        ("Right drag", "Zoom in and out"),
        ("Wheel", "Zoom"),
        ("Shift + drag", "Pan camera"),
        ("Ctrl + Left drag", "Move model"),
        ("M", "Toggle model move mode, then Left drag"),
        ("Ctrl + O", "Open model (SMD, VTA, DMX, MDL)"),
        ("Ctrl + ,", "Open Settings dialog"),
        ("Ctrl + Shift + M", "Set materials folder manually"),
        ("Ctrl + R", "Reset camera"),
        ("Ctrl + M", "Reset model position"),
        ("Ctrl + Shift + R", "Reset camera and model"),
        ("F1", "Open this dialog"),
        ("Clip dropdown", "Switch between DMX animation clips"),
        ("Sequence slider", "Scrub through skeletal animation (SMD or DMX)"),
        ("Driver checkbox", "Let a driver bone control Intensity and Progress"),
    ]

    table = "<table>"
    for key, action in rows:
        table += (
            "<tr><td class=\"k\">" + key + "</td><td>" + action + "</td></tr>"
        )
    table += "</table>"

    body = (
        "<h2>Mouse and keyboard</h2>" + table +
        "<h2>Playback bar</h2>"
        "<p><code>Play</code> starts or pauses the active animation. "
        "<code>Progress</code> scrubs frames or shape keys. "
        "<code>Intensity</code> blends from the basis shape toward the selected shape. "
        "<code>FPS</code> sets playback speed.</p>"
        "<p><code>Shape Key Mode</code> shows each target as an independent shape. "
        "Turn it off to blend targets as a sequence. "
        "<code>Skel</code> applies skeletal deformation. "
        "<code>Driver</code> lets a driver bone control intensity and progress.</p>"
        "<p><code>Clip</code> selects a DMX animation clip. "
        "<code>Sequence</code> scrubs through the current skeletal animation.</p>"
        "<h2>Materials and Textures</h2>"
        "<p>Compiled MDL models auto-mount their workshop or game materials directory. "
        "You can also manually set a directory via <code>File &gt; Set Materials Folder...</code> "
        "or configure default search directories in <code>File &gt; Settings...</code> (Ctrl+,).</p>"
    )

    return _page("Viewer Controls", body)


def animation_html():
    body = (
        "<h2>Shape keys</h2>"
        "<p>Each target is one static shape. In <code>Shape Key Mode</code> the viewer "
        "shows targets one at a time, and <code>Intensity</code> blends from the basis "
        "mesh to the selected shape. This matches how VTA and DMX store flex shapes.</p>"
        "<h2>Sequences</h2>"
        "<p>With <code>Shape Key Mode</code> off, targets are treated as keyframes and "
        "blended in order. Use this for true vertex animation frames.</p>"
        "<h2>VTA notes</h2>"
        "<p>The first target is treated as the basis and should contain every vertex in "
        "its reference position. Later targets are sparse and only list changed vertices. "
        "Vertex IDs refer to the order of vertices in the reference SMD triangles block.</p>"
        "<h2>DMX notes</h2>"
        "<p>DMX delta states store offsets from the mesh, so the parser adds them onto "
        "the base positions. A synthetic basis target is inserted so the shapes line up "
        "with the VTA model.</p>"
        "<p>DMX files can contain multiple animation clips. Use the <code>Clip</code> "
        "dropdown to select one. The <code>Sequence</code> slider lets you scrub through "
        "the clip.</p>"
        "<h2>Driver bone</h2>"
        "<p>A bone such as <code>vertexAnimDriver</code> can drive a vertex animation. "
        "Its X position sets intensity from 0 to 1, and its Y position sets progress "
        "from 0 to 1.</p>"
    )

    return _page("Animation Guide", body)


def formats_html():
    body = (
        "<h2>MDL / VVD / VTX</h2>"
        "<p>Compiled Source engine models. The viewer parses the main <code>.mdl</code> header, "
        "reconstructs LOD 0 vertex pools from <code>.vvd</code> files, and reads optimized "
        "hardware triangle strips and lists from <code>.vtx</code> files (including dx90, dx80, "
        "and generic vtx formats). Bones and skin-family tables are loaded directly.</p>"
        "<h2>SMD</h2>"
        "<p>ASCII StudioModel Data. Reference files contain nodes, skeleton, and "
        "triangles. Animation files contain nodes and skeleton only.</p>"
        "<h2>VTA</h2>"
        "<p>Vertex animation library. Never has a triangles block and needs a matching "
        "reference SMD.</p>"
        "<h2>DMX</h2>"
        "<p>Data Model eXchange. Supports <code>keyvalues2</code> ASCII and <code>binary</code> "
        "versions 1 through 5, including shape keys and multiple animation clips.</p>"
        "<h2>VTF & VMT</h2>"
        "<p>Valve Texture Format (versions 7.0 to 7.5) and Valve Material templates. "
        "Supported VMT parameters include <code>$basetexture</code>, <code>$iris</code>, "
        "<code>$nocull</code>, <code>$no_draw</code>, <code>$selfillum</code>, <code>$color</code>, "
        "<code>$translucent</code>, and <code>$alphatest</code>.</p>"
    )

    return _page("Supported Formats", body)


def about_html():
    body = (
        "<h2>Version " + __version__ + "</h2>"
        "<h2>What it reads</h2>"
        "<p>Compiled Source models (.mdl, .vvd, .vtx), SMD reference meshes, SMD skeletal sequences, "
        "VTA vertex animation, DMX models in KeyValues2 and binary, DMX shape keys, and "
        "VTF/VMT materials.</p>"
        "<h2>Built with</h2>"
        "<p>PySide6, PyOpenGL, and numpy.</p>"
    )

    return _page("Source Model Viewer", body)