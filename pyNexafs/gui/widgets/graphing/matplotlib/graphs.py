import os, sys
import matplotlib.figure
import overrides
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backend_bases import NavigationToolbar2 as NavTB
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTBQT
from matplotlib.backends.qt_compat import _to_int, __version__
from PyQt6 import QtGui, QtWidgets, QtCore
from PyQt6.QtWidgets import QStyle


class FigureCanvas(FigureCanvasQTAgg):
    def __init__(self, mpl_fig: matplotlib.figure.Figure):
        super().__init__(mpl_fig)


class NEXAFS_NavQT(NavTBQT):
    normalisationUpdate = QtCore.pyqtSignal(bool)

    toolitems = [*NavTBQT.toolitems]

    ## Example extension to add custom functions.
    # toolitems.append((None, None, None, None))
    # toolitems.append(
    #     ('Normalisation',
    #     'Allows use of normalisation and background subtraction options',
    #     ICONS["normalisation"],
    #     'norm_toolkit'),
    # )

    # Redefine init method to include custom icons in toolbar without attempting to join string to icon name.
    @overrides.overrides
    def __init__(self, canvas, parent=None, coordinates=True):
        """coordinates: should we show the coordinates on the right?"""
        QtWidgets.QToolBar.__init__(self, parent)
        self.setAllowedAreas(
            QtCore.Qt.ToolBarArea(
                _to_int(QtCore.Qt.ToolBarArea.TopToolBarArea)
                | _to_int(QtCore.Qt.ToolBarArea.BottomToolBarArea)
            )
        )
        self.coordinates = coordinates
        self._actions = {}  # mapping of toolitem method names to QActions.
        self._subplot_dialog = None

        for text, tooltip_text, image_file, callback in self.toolitems:
            if text is None:
                self.addSeparator()
            else:
                a = self.addAction(
                    self._icon(
                        image_file
                    ),  # moved + ".png" to redefined '_icon' method.
                    text,
                    getattr(self, callback),
                )
                self._actions[callback] = a
                if callback in ["zoom", "pan"]:
                    a.setCheckable(True)
                if tooltip_text is not None:
                    a.setToolTip(tooltip_text)

        # Add the (x, y) location widget at the right side of the toolbar
        # The stretch factor is 1 which means any resizing of the toolbar
        # will resize this label instead of the buttons.
        if self.coordinates:
            self.locLabel = QtWidgets.QLabel("", self)
            self.locLabel.setAlignment(
                QtCore.Qt.AlignmentFlag(
                    _to_int(QtCore.Qt.AlignmentFlag.AlignRight)
                    | _to_int(QtCore.Qt.AlignmentFlag.AlignVCenter)
                )
            )

            self.locLabel.setSizePolicy(
                QtWidgets.QSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Ignored,
                )
            )
            labelAction = self.addWidget(self.locLabel)
            labelAction.setVisible(True)

        NavTB.__init__(self, canvas)

        # Initialise normalisation options.
        self._normalisation_options = None

    @overrides.overrides
    def _icon(self, name: str | QtGui.QIcon | QStyle.StandardPixmap) -> QtGui.QIcon:
        if isinstance(name, QtGui.QIcon):
            # Provide your own icon
            return name
        elif name in QStyle.StandardPixmap:
            return QtWidgets.QApplication.style().standardIcon(name)
        elif isinstance(name, str):
            # Check if provided string is a icon path.
            if os.path.isfile(name):
                return QtGui.QIcon(name)
            else:
                # use matplotlib backend to get inbuilt images.
                return super()._icon(
                    name=name + ".png"
                )  # as per the original __init__ code.
        else:
            raise ValueError("Invalid icon type")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout()

    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.random.rand(10), np.random.rand(10), "Random scatters")
    graph = FigureCanvas(fig)
    nav = NEXAFS_NavQT(graph)

    main.setLayout(layout)
    layout.addWidget(graph)
    main.show()

    sys.exit(app.exec())
