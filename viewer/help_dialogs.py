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
        ("Ctrl + O", "Open model"),
        ("Ctrl + Shift + M", "Set materials folder for VTF textures"),
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
        "<h2>Textures</h2>"
        "<p>Use <code>File &gt; Set Materials Folder...</code> to point the viewer at an "
        "extracted <code>materials/</code> directory. Each material on the model is then "
        "matched to a <code>.vtf</code> file and textured. Materials without a matching "
        "texture keep their flat placeholder color.</p>"
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
        "Vertex IDs refer to the order of vertices in the reference SMD triangles block, "
        "so the VTA must match the reference SMD.</p>"
        "<h2>DMX notes</h2>"
        "<p>DMX delta states store offsets from the mesh, so the parser adds them onto "
        "the base positions. A synthetic basis target is inserted so the shapes line up "
        "with the VTA model.</p>"
        "<p>DMX files can contain multiple animation clips. Use the <code>Clip</code> "
        "dropdown to select one. The <code>Sequence</code> slider lets you scrub through "
        "the clip, and the playback speed is controlled by the FPS spin box.</p>"
        "<h2>Driver bone</h2>"
        "<p>A bone such as <code>vertexAnimDriver</code> can drive a vertex animation. "
        "Its X position sets intensity from 0 to 1, and its Y position sets progress "
        "from 0 to 1. If your authoring tool is Z-up, progress may live on Z instead. "
        "The parser picks the axis with the most movement, and the flex info TXT can "
        "override it.</p>"
        "<p>The <code>Driver</code> checkbox enables this; when active, the "
        "Intensity and Progress sliders are disabled and the driver bone takes over.</p>"
    )

    return _page("Animation Guide", body)


def formats_html():
    body = (
        "<h2>SMD</h2>"
        "<p>ASCII StudioModel Data. Reference files contain nodes, skeleton, and "
        "triangles. Animation files contain nodes and skeleton only. X is north. "
        "Comments use <code>//</code>, <code>#</code>, or <code>;</code> in Source "
        "studiomdl.</p>"
        "<h2>VTA</h2>"
        "<p>Vertex animation library. Never has a triangles block and needs a matching "
        "reference SMD. The skeleton block only needs a <code>time</code> header per "
        "shape.</p>"
        "<h2>DMX</h2>"
        "<p>Data Model eXchange. Stores the reference mesh, shape keys, and animation "
        "in one file. Two encodings are supported.</p>"
        "<p><code>keyvalues2</code> is ASCII and hand-editable. <code>binary</code> "
        "versions 1 to 5 are supported, including string tables, element headers, and "
        "attribute arrays.</p>"
        "<p>DMX animation lists with multiple clips are fully supported, including "
        "metadata (duration, frame count, FPS) shown in the clip selector. Corrective "
        "shape flags are also parsed for future use.</p>"
        "<h2>VTF</h2>"
        "<p>Valve Texture Format, versions 7.0 to 7.5. Set a materials folder with "
        "<code>File &gt; Set Materials Folder...</code> and the viewer resolves each "
        "model material to a <code>.vtf</code> file, searching subfolders when the "
        "material name carries no path.</p>"
        "<p>Decoded formats: DXT1, DXT3, DXT5, BGRA8888, BGRX8888, BGR888, RGB888, "
        "BGR565, RGB565, BGRA4444, BGRA5551, RGBA8888, ABGR8888, ARGB8888, I8, IA88, "
        "A8, and UV88. The largest mipmap is used, with trilinear filtering.</p>"
        "<p>Materials with no matching or decodable texture fall back to a flat "
        "per-material color, so a model always renders even with a partial materials "
        "folder.</p>"
        "<h2>Binary quirks handled</h2>"
        "<p>Attribute type IDs differ per engine branch. The parser tries several index "
        "sizes, allows a null byte after the header, reads element headers before "
        "attribute blocks, and accepts array type ranges starting at 15, 32, 128, or "
        "224.</p>"
    )

    return _page("Supported Formats", body)


def about_html():
    body = (
        "<h2>Version " + __version__ + "</h2>"
        "<h2>What it reads</h2>"
        "<p>SMD reference meshes, SMD skeletal sequences, VTA vertex animation, DMX "
        "models in KeyValues2 and binary, DMX shape keys, DMX animation lists, and "
        "VTF textures from a user-supplied materials folder.</p>"
        "<h2>Built with</h2>"
        "<p>PySide6, PyOpenGL, and numpy.</p>"
        "<h2>Notes</h2>"
        "<p>Binary DMX layouts differ between engine branches. The parser auto-detects "
        "string tables, element headers, and array type ranges, and reports the best "
        "attempt if a file cannot be read.</p>"
        "<p>VTF texture support decodes DXT1, DXT3, DXT5, and the common uncompressed "
        "formats. Textures are matched to materials by name against the materials "
        "folder, with a recursive search for bare material names.</p>"
    )

    return _page("Source Model Viewer", body)