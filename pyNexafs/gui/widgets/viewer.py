from PyQt6.QtWidgets import (
    QWidget,
    QFileDialog,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QComboBox,
    QFrame,
    QGridLayout,
    QSizeGrip,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QTextEdit
from PyQt6.QtGui import QColor, QPalette
import os, sys

from pyNexafs.parsers import parser_loaders, parser_base
from pyNexafs.gui.widgets.graphing.plotly_graphs import PlotlyGraph
from pyNexafs.gui.widgets.fileloader import nexafs_fileloader

import random


class nexafsViewer(QWidget):
    """
    General viewer for NEXAFS data.

    Includes various submodules, including directory browser, data viewer, and plot viewer.
    """

    def __init__(self):
        super().__init__()

        ## Instance attributes

        ## Initialise elements
        self.header_browser = nexafs_fileloader()
        self.graph = PlotlyGraph()
        self.data_grip = QSizeGrip(self.header_browser)
        self.graph_grip = QSizeGrip(self.graph)

        ## Viewer Layout
        viewer_layout = QHBoxLayout()
        viewer_layout.addWidget(self.header_browser)
        viewer_layout.addWidget(self.graph)

        # Assign viewer layout to widget.
        self.setLayout(viewer_layout)

        ## Element attributes
        viewer_layout.setStretch(0, 2)
        viewer_layout.setStretch(1, 3)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = nexafsViewer()
    window.show()
    window.setWindowTitle("pyNexafs Viewer")
    sys.exit(app.exec())
