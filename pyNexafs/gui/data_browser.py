import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTextEdit,
    QHBoxLayout,
    QWidget,
    QListView,
    QVBoxLayout,
    QListWidget,
    QSplitter,
    QSplitterHandle,
)
from PyQt6.QtGui import QColor, QPalette, QLinearGradient

import matplotlib.figure

from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas
from pyNexafs.gui.widgets.fileloader import nexafsFileLoader
from pyNexafs.gui.widgets.viewer import nexafsViewer
from pyNexafs.nexafs.scan import scan_base
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTB

import numpy as np


class mainWidget(QWidget):
    def __init__(self):
        super().__init__()
        # Initialise elements
        self.draggable = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout = QHBoxLayout()
        self.loader = nexafsFileLoader(parent=self)
        self.viewer = nexafsViewer(parent=self)
        # self.converter = converterWidget()

        # Add to layout
        self.draggable.addWidget(self.loader)
        self.draggable.addWidget(self.viewer)
        self.main_layout.addWidget(self.draggable)
        # self.main_layout.addWidget(self.loader)
        # self.main_layout.addWidget(self.viewer)
        # self.sub_layout.addWidget(self.converter)
        self.setLayout(self.main_layout)

        ### Init UI
        grad = QLinearGradient(0, 0, 1, 0)
        grad.setCoordinateMode(grad.CoordinateMode.ObjectMode)

        # # TODO: Find another way to define a black line on handles with some margin two widgets.
        # for handle in self.draggable.findChildren(QSplitterHandle):
        #     if handle.parent().orientation() == Qt.Orientation.Horizontal:
        #         # handle.setStyleSheet("background-color: black;")
        #         # handle.setStyleSheet("qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(255, 255, 255, 0), stop:0.407273 rgba(200, 200, 200, 255), stop:0.4825 rgba(101, 104, 113, 235), stop:0.6 rgba(255, 255, 255, 0));")
        #         # handle.setStyleSheet("qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #eee, stop:1 #ccc);")

        #         handle.set

        #         handle.setMinimumWidth(1)
        #         handle.setMinimumHeight(1)
        #         handle.setFixedWidth(1)

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
        # Store existing viewer dataseries selection
        previousSelection = self.viewer.dataseries_selected
        # Load in the scan objects
        selection_parse_objs = self.loader.loaded_parser_files_selection
        self.viewer.add_parsers_to_scans(selection_parse_objs)

        # Change the file selection in the viewer.
        self.viewer.selected_filenames = self.loader.selected_filenames

        # Update the selected dataseries in the viewer to graph.
        if previousSelection is not None:
            # Restore the previous selection if it exists.
            if np.all([label in self.viewer.dataseries for label in previousSelection]):
                self.viewer.dataseries_selected = previousSelection


def gui():

    app = QApplication(sys.argv)
    window = mainWidget()
    window.show()
    window.setWindowTitle("pyNexafs File Browser")
    app.exec()


if __name__ == "__main__":
    gui()
