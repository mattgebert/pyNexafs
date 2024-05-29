from PyQt6 import QtWidgets, QtCore, QtGui
from pyNexafs.parsers._base import parser_base
from typing import Type
import sys
from pyNexafs.gui.widgets.fileloader import directory_selector


class save_directory_selector(directory_selector):

    edit_description = "Path to save QANT-compatible data to."
    dialog_caption = "Select Save Directory"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.insertWidget(0, QtWidgets.QLabel("Save Directory:"))


class nexafsParserConverter(QtWidgets.QWidget):
    def __init__(self, parsers: list[Type[parser_base]] = [], parent=None):
        super().__init__(parent)
        self._parsers = parsers.copy()
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        self._draggable = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)

        ## Setup UI elements
        # Title
        title_label = QtWidgets.QLabel("Converstion:")
        self._title_label = title_label
        # Folder selector
        dir_sel = save_directory_selector()

        # Line for vertical separation.
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setLineWidth(1)
        line.setMinimumWidth(1)
        line.setMinimumHeight(1)
        line.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum
        )
        self._line = line
        # NaN inspector
        nan_widget = QtWidgets.QWidget()
        nan_layout = QtWidgets.QVBoxLayout()
        nan_label = QtWidgets.QLabel("NaN Inspector:")
        nan_layout.addWidget(nan_label)
        nan_widget.setLayout(nan_layout)
        nan_widget.setContentsMargins(0, 0, 0, 0)
        nan_layout.setContentsMargins(0, 0, 0, 0)
        self._draggable.addWidget(nan_widget)
        self._layout.addWidget(title_label)
        self._layout.addLayout(dir_sel)
        self._layout.addWidget(line)
        self._layout.addWidget(self._draggable)

        # Initialise UI
        self.on_recolour()

    @property
    def parsers(self) -> Type[parser_base]:
        return self._parsers

    @parsers.setter
    def parsers(self, parsers: Type[parser_base]):
        self._parsers = parsers.copy()

    def on_recolour(self):
        """
        Recolour the division lines based on the theme.
        """
        # Get theme
        toolbar_palette = self.palette()
        light_theme_bool = toolbar_palette.window().color().lightnessF() > 0.5
        self._line.setStyleSheet(
            "background-color: " + ("black;" if light_theme_bool else "white;")
        )

    def event(self, event: QtCore.QEvent) -> bool:
        """
        Event handler for the widget.

        Adds palette change control for light/dark mode to QWidget event handler.

        Parameters
        ----------
        event : QtCore.QEvent
            The event to handle.

        Returns
        -------
        bool
            Whether the event was handled.
        """
        if (
            event.type() == QtCore.QEvent.Type.PaletteChange
            or event.type() == QtCore.QEvent.Type.ApplicationPaletteChange
        ):
            self.on_recolour()
        return super().event(event)


class nexafsConverterQANT(nexafsParserConverter):
    def __init__(self, parsers: list[Type[parser_base]] = [], parent=None):
        super().__init__(parsers, parent)
        self._title_label.setText("QANT Conversion:")

    pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = nexafsConverterQANT()
    main.setWindowTitle("QANT Converter")
    main.show()
    app.exec()
