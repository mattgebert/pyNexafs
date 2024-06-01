from PyQt6 import QtWidgets, QtCore, QtGui
from pyNexafs.parsers._base import parser_base
from typing import Type
import sys
from pyNexafs.gui.widgets.fileloader import directory_selector
from enum import Enum
import numpy as np


class save_directory_selector(directory_selector):

    edit_description = "Path to save QANT-compatible data to."
    dialog_caption = "Select Save Directory"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.insertWidget(0, QtWidgets.QLabel("Save Directory:"))


class NaN_OPTION(Enum):
    """
    Enumerate to define methods of processing NaN (Not a Number) values.

    Attributes
    ----------
    NONE : int
        No special treatment. Data stored with NaN labels. Incompatible with QANT?
    REMOVE : int
        Removes NaN rows. Affects multiple data columns that might not have NaNs.
    INTERPOLATE_2 : int
        Interpolates the nearest 2 (non-NaN) point neighbours to
    """

    NONE = 0
    REMOVE = 1
    INTERPOLATE_2 = 2
    INTERPOLATE_4 = 3
    INTERPOLATE_N = 4


class nexafsParserConverter(QtWidgets.QWidget):
    def __init__(self, parsers: list[Type[parser_base]] = [], parent=None):
        super().__init__(parent)
        self._parsers = {parser.filename: parser for parser in parsers}
        self._conversions = []
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
        # File statistics
        file_widget = QtWidgets.QWidget()
        file_layout = QtWidgets.QGridLayout()
        file_label = QtWidgets.QLabel("File Statistics")
        file_layout.addWidget(file_label, 0, 0, 1, 4)
        file_widget.setLayout(file_layout)
        file_widget.setContentsMargins(0, 0, 0, 0)
        file_layout.setContentsMargins(0, 0, 0, 0)
        # Number of files
        files_num_label = QtWidgets.QLabel("Number of Files:")
        self.files_num_indicator = QtWidgets.QLabel("")
        file_layout.addWidget(files_num_label, 1, 0)
        file_layout.addWidget(self.files_num_indicator, 1, 1)
        # Total filesize
        files_size_label = QtWidgets.QLabel("Total Filesize:")
        self.files_size_indicator = QtWidgets.QLabel("")
        file_layout.addWidget(files_size_label, 1, 2)
        file_layout.addWidget(self.files_size_indicator, 1, 3)

        # NaN inspector
        nan_widget = QtWidgets.QWidget()
        nan_layout = QtWidgets.QGridLayout()
        nan_label = QtWidgets.QLabel("NaN Inspector:")
        nan_layout.addWidget(nan_label, 0, 0, 1, 2)
        nan_widget.setLayout(nan_layout)
        nan_widget.setContentsMargins(0, 0, 0, 0)
        nan_layout.setContentsMargins(0, 0, 0, 0)
        # Number of files with NaNs
        total_nan_files_label = QtWidgets.QLabel("Files with NaNs:")
        self.total_nan_files_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(total_nan_files_label, 0, 2)
        nan_layout.addWidget(self.total_nan_files_indicator, 0, 3)
        # Total nan indicator:
        total_nan_instances_label = QtWidgets.QLabel("Total NaN Instances:")
        self.total_nan_instances_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(total_nan_instances_label, 1, 0)
        nan_layout.addWidget(self.total_nan_instances_indicator, 1, 1)
        # Per file nan indicator:
        per_file_nan_label = QtWidgets.QLabel("NaN Instances per Nan File:")
        self.per_file_nan_indicator = QtWidgets.QLabel("")
        nan_layout.addWidget(per_file_nan_label, 1, 2)
        nan_layout.addWidget(self.per_file_nan_indicator, 1, 3)
        # Method of dealing with Nan Values.

        # Add to layout
        self._draggable.addWidget(file_widget)
        self._draggable.addWidget(nan_widget)
        self._layout.addWidget(title_label)
        self._layout.addLayout(dir_sel)
        self._layout.addWidget(line)
        self._layout.addWidget(self._draggable)

        # Initialise UI
        self.on_recolour()

    def analyse_nan(self):
        """
        Calculate the number of NaN values in the data.
        """
        parsers = self.parsers
        if len(parsers) > 0:
            total_nan_values = 0
            nanfiles = 0
            for parser in parsers:
                nans = np.isnan(parser.data).sum()
                nanfiles += nans > 0
                total_nan_values += nans
            self.total_nan_files_indicator.setText(str(nanfiles))
            self.total_nan_instances_indicator.setText(str(total_nan_values))
            self.per_file_nan_indicator.setText(f"{total_nan_values / nanfiles:.2f}")
        else:
            self.total_nan_files_indicator.setText("")
            self.total_nan_instances_indicator.setText("")
            self.per_file_nan_indicator.setText("")

    def analyse_filesize(self) -> None:
        """
        Analyses the total filesize of the data.
        """
        self.files_num_indicator.setText(str(self.num_files))
        total_filesize = sum([parser.filesize for parser in self.parsers])
        i = 0
        while total_filesize > 1024:
            total_filesize /= 1024
            i += 1
        total_filesize = (
            f"{total_filesize:.2f} {['B', 'KB', 'MB', 'GB', 'TB', 'PB'][i]}"
        )
        self.files_size_indicator.setText(total_filesize)

    def analyse(self):
        """
        Runs analysis on the provided parsers, updating GUI elements.
        """
        self.analyse_nan()
        self.analyse_filesize()

    @property
    def parsers(self) -> list[Type[parser_base]]:
        """
        The current set of parsers to use for conversion.

        Parameters
        ----------
        parsers : list[Type[parser_base]]
            A list of parser classes to use for conversion.
            None values are disregarded when setting.

        Returns
        -------
        list[Type[parser_base]]
            A list of parser classes to use for conversion.
        """
        return list(self._parsers.values())

    @parsers.setter
    def parsers(self, parsers: list[Type[parser_base]]):
        self._parsers = {
            parser.filename: parser for parser in parsers if parser is not None
        }
        self.analyse()

    @property
    def conversions(self) -> list[Type[parser_base]]:
        return

    @property
    def num_files(self):
        """
        Calculate the number of files in the data.
        """
        return len(self.parsers)

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
