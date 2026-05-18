import sys

from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QWidget,
)


from pyNexafs.gui.widgets.fileloader import nexafsFileLoader


class browserWidget(QWidget):
    """
    A widget for browsing and loading NEXAFS files.

    Controls file parsing and reduction, as well as displays attributes.
    """

    def __init__(self, parent=None, init_dir: str | None = None):
        super().__init__(parent=parent)

        self._main_layout = QHBoxLayout()
        self.setLayout(self._main_layout)

        self.parser_loader = nexafsFileLoader(parent=self)
        self.parser_loader.directory_selector.folder_path = init_dir
        self._main_layout.addWidget(self.parser_loader)

        if parent is not None:
            # Enable borderless behaviour, so have tight fitting elements.
            self.setContentsMargins(0, 0, 0, 0)
            self._main_layout.setContentsMargins(0, 0, 0, 0)


def gui(directory: str | None = None):
    app = QApplication(sys.argv)
    window = browserWidget(init_dir=directory)
    window.show()
    window.setWindowTitle("pyNexafs File Browser")
    app.exec()


if __name__ == "__main__":
    import pyNexafs
    import os

    path_pkg = pyNexafs.__path__
    path_mex2 = os.path.normpath(
        os.path.join(path_pkg[0], "..", "tests", "test_data", "au", "MEX2", "2024-03")
    )
    print(f"Attempting to load `{path_mex2}`")
    if os.path.exists(path_mex2):
        gui(path_mex2)
    else:
        gui()
