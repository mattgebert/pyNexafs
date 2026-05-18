import sys

from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QWidget,
)


from pyNexafs.gui.widgets.fileloader import nexafsFileLoader


class browserWidget(QWidget):
    """A widget for browsing and loading NEXAFS files."""

    def __init__(self, parent=None, init_dir: str | None = None):
        super().__init__(parent=parent)
        # Initialise elements
        # self._draggable = QSplitter(Qt.Orientation.Horizontal)
        self._main_layout = QHBoxLayout()
        self.parser_loader = nexafsFileLoader(parent=self)
        # self.viewer = nexafsViewer(parent=self)
        # self.converter = converterWidget()

        # Add to layout
        # self._draggable.addWidget(self.parser_loader)
        # self._draggable.addWidget(self.viewer)
        # self._main_layout.addWidget(self._draggable)
        self._main_layout.addWidget(self.parser_loader)
        self.setLayout(self._main_layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._main_layout.setContentsMargins(0, 0, 0, 0)

        ### Connections
        # Load scans
        self.parser_loader.selectionLoaded.connect(self._on_load_selection)
        # Remove scans with parser / directory change.
        self.parser_loader.directory_selector.new_path.connect(
            self._on_dir_parser_change
        )
        self.parser_loader.nexafs_parser_selector.currentIndexChanged.connect(
            self._on_dir_parser_change
        )
        # self.parser_loader.relabelling.connect(self.viewer.on_relabel)

        ### Init directory
        if init_dir is not None:
            self.parser_loader.directory_selector.folder_path_edit.setText(init_dir)
            self.parser_loader.directory_selector.folder_path_edit.editingFinished.emit()

    def _on_dir_parser_change(self):
        # Delete the current scan objects

        # del self.viewer.scans
        pass

    def _on_load_selection(self):
        # Load in the scan objects
        # selection_parse_objs = self.parser_loader.loaded_parser_files_selection

        # self.viewer.add_parsers_to_scans(selection_parse_objs)
        # Change the file selection in the viewer.
        # self.viewer.selected_filenames = self.parser_loader.selected_filenames
        pass


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
    path_mex2 = os.path.join(
        path_pkg[0], "..", "tests", "test_data", "au", "MEX2", "2024-03"
    )
    if os.path.exists(path_mex2):
        gui(path_mex2)
    else:
        gui()
