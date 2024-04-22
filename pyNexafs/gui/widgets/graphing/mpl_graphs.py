import sys
import matplotlib

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication
)
import matplotlib.figure

from pyNexafs.gui.widgets.fileloader import nexafs_fileloader
from pyNexafs.nexafs.scan import scan_base

import numpy as np


matplotlib.use("QtAgg")

from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavTB
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt


class FigureCanvas(FigureCanvasQTAgg):
    def __init__(self, mpl_fig: matplotlib.figure.Figure):
        super().__init__(mpl_fig)


# Demo of QT app
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    myFig = matplotlib.figure.Figure()
    
    window = FigureCanvas(myFig)
    window.show()
    window.setWindowTitle("pyNexafs File Loader")
    
    x = np.linspace(0,1, 500)
    noise = np.random.rand(500)
    y = np.exp(x)
    
    ax = myFig.add_subplot(111)
    ax.scatter(x,y)
    
    app.exec()