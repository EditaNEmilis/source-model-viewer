import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from viewer.main_window import MainWindow


def main():
    surface_format = QSurfaceFormat()
    surface_format.setDepthBufferSize(24)
    surface_format.setStencilBufferSize(8)
    surface_format.setSamples(4)
    surface_format.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(surface_format)

    app = QApplication(sys.argv)
    app.setApplicationName("Source Model Viewer")
    app.setOrganizationName("SourceModelViewer")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()