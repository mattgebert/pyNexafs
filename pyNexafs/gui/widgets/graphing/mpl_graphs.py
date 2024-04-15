import sys
import matplotlib

matplotlib.use("QtAgg")

from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt


class FigureCanvas(FigureCanvasQTAgg):
    def __init__(self, mpl_fig: matplotlib.figure.Figure):
        super(FigureCanvas, self).__init__(mpl_fig)
