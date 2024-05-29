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
from pyNexafs.gui.data_browser import mainWidget

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTB
import numpy as np
import overrides


class converterWidget(mainWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.converter = nexafsConverterQANT(parent=self)

        # Add to layout
        self._draggable.addWidget(self.converter)
        self._main_layout.addWidget(self._draggable)
        self.setLayout(self._main_layout)

        ### Init UI
        grad = QLinearGradient(0, 0, 1, 0)
        grad.setCoordinateMode(grad.CoordinateMode.ObjectMode)

        ### Connections
        ## Load scans
        # self.loader.selectionLoaded.connect(self._on_load_selection)
        ## Remove scans with parser / directory change.
        # self.loader.directory_selector.newPath.connect(self._on_dir_parser_change)

    def _on_dir_parser_change(self):
        # Delete the current scan objects
        del self.viewer.scans

    @overrides.overrides
    def _on_load_selection(self):
        super()._on_load_selection()


def gui():
    app = QApplication(sys.argv)
    window = converterWidget()
    window.show()
    window.setWindowTitle("pyNexafs QANT Converter")
    app.exec()


if __name__ == "__main__":
    gui()
