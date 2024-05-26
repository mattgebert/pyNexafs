import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QWidget,
    QSplitter,
)
from PyQt6.QtGui import QLinearGradient
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas
from pyNexafs.gui.widgets.fileloader import nexafsFileLoader
from pyNexafs.gui.widgets.viewer import nexafsViewer
from pyNexafs.gui.widgets.converter import nexafsConverterQANT
from pyNexafs.nexafs.scan import scan_base
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTB
import numpy as np


class mainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._previousSelection = None
        self.draggable = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout = QHBoxLayout()
        self.loader = nexafsFileLoader(parent=self)
        self.viewer = nexafsViewer(parent=self)
        self.converter = nexafsConverterQANT(parent=self)

        # Add to layout
        self.draggable.addWidget(self.loader)
        self.draggable.addWidget(self.viewer)
        self.draggable.addWidget(self.converter)
        self.main_layout.addWidget(self.draggable)
        self.setLayout(self.main_layout)

        ### Init UI
        grad = QLinearGradient(0, 0, 1, 0)
        grad.setCoordinateMode(grad.CoordinateMode.ObjectMode)

        ### Connections
        # Load scans
        self.loader.selectionLoaded.connect(self._on_load_selection)
        # Remove scans with parser / directory change.
        self.loader.directory_selector.newPath.connect(self._on_dir_parser_change)
        self.loader.nexafs_parser_selector.currentIndexChanged.connect(
            self._on_dir_parser_change
        )

    def _on_dir_parser_change(self):
        # Delete the current scan objects
        del self.viewer.scans

    def _on_load_selection(self):
        # Store existing viewer selection
        self._previousSelection = self.viewer.dataseries_selected
        # Load in the scan objects
        selection_parse_objs = self.loader.loaded_parser_files_selection
        scans_copy = self.viewer.scans
        if scans_copy is not None:
            for name, parser in selection_parse_objs.items():
                if name not in scans_copy:
                    self.viewer.scans = {name: parser.to_scan(True)}
                else:
                    # If parser object is not the same, reload the scan object.
                    if scans_copy[name].parser is not parser:
                        self.viewer.scans = {name: parser.to_scan(True)}
        else:
            self.viewer.scans = {
                name: parser.to_scan(True)
                for name, parser in selection_parse_objs.items()
            }

        # Plot the scan objects
        self.viewer.set_file_selection(self.loader.selected_filenames)

        if self._previousSelection is not None:
            # Restore the previous selection if it exists.
            if np.all(
                [label in self.viewer.dataseries for label in self._previousSelection]
            ):
                self.viewer.dataseries_selected = self._previousSelection
            # Reset the previous selection to None
            self._previousSelection = None


def gui():

    app = QApplication(sys.argv)
    window = mainWidget()
    window.show()
    window.setWindowTitle("pyNexafs QANT Converter")
    app.exec()


if __name__ == "__main__":
    gui()
