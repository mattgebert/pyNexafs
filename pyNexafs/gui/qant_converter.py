import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QWidget,
    QSplitter,
)
from PyQt6.QtGui import QLinearGradient
from PyQt6 import QtGui, QtWidgets, QtCore
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas
from pyNexafs.gui.widgets.fileloader import nexafsFileLoader
from pyNexafs.gui.widgets.viewer import nexafsViewer
from pyNexafs.gui.widgets.converter import nexafsConverterQANT
from pyNexafs.gui.data_browser import browserWidget

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTB
import numpy as np
import overrides


class converterWidget(browserWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.converter = nexafsConverterQANT(parent=self)
        self.converter.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum
        )

        # Add to layout
        self._draggable.addWidget(self.converter)

    @overrides.overrides
    def _on_load_selection(self):
        super()._on_load_selection()
        self.converter.parsers = self.loader.loaded_parser_files_selection.values()


def gui():
    app = QApplication(sys.argv)
    window = converterWidget()
    window.show()
    window.setWindowTitle("pyNexafs QANT Converter")
    app.exec()


if __name__ == "__main__":
    gui()
