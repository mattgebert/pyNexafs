from PyQt6 import QtGui, QtWidgets, QtCore
import numpy as np
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import SpanSelector


class EnergyBinReducer(QtWidgets.QWidget):
    """
    Reduces detector energy bins to some sum of counts.

    Parameters
    ----------
    QtWidgets : _type_
        _description_
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)

        self.fig = fig = plt.figure()
        self.canvas = FigureCanvas(fig)
        self._layout.addWidget(self.canvas)
        self.ax = fig.add_subplot(111)
        assert isinstance(self.ax, plt.Axes)
        self.plot()

    def plot(self):
        self.x = x = np.linspace(0, 10, 100)
        self.y = y = np.sin(self.x)

        self.ax.clear()
        ax = self.ax
        self.ax.plot(x, y)
        (self.line2,) = ax.plot([], [])
        (self.line3,) = ax.plot([], [])
        self.canvas.draw()
        self.span = SpanSelector(
            self.ax,
            self.onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.2, facecolor="red"),
            # props=dict(alpha=0.5, facecolor="tab:blue"),
            interactive=True,
            drag_from_anywhere=True,
            button=MouseButton.LEFT,
        )
        # self.canvas.draw()

    def onselect(self, xmin, xmax):
        print("Select!")
        indmin, indmax = np.searchsorted(self.x, (xmin, xmax))
        indmax = min(len(self.x) - 1, indmax)

        region_x = self.x[indmin:indmax]
        region_y = self.y[indmin:indmax]

        if len(region_x) >= 2:
            self.line2.set_data(region_x, region_y)
            # self.ax.set_xlim(region_x[0], region_x[-1])
            self.ax.set_ylim(region_y.min(), region_y.max())
            self.fig.canvas.draw_idle()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = EnergyBinReducer()
    win.show()
    app.exec()
