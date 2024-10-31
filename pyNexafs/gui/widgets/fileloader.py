import re
import traceback
from PyQt6 import QtWidgets, QtCore, QtGui
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
    QTableView,
    QApplication,
    QLabel,
    QTextEdit,
    QCheckBox,
    QAbstractItemView,
    QHeaderView,
    QSplitter,
    QStyle,
    QProgressBar,
)

# from PyQt6.QtWidgets import QScrollBar, QHeaderView, QMainWindow, QTableWidget, QFrame, QGridLayout, QSizeGrip
from PyQt6.QtCore import (
    Qt,
    pyqtSignal,
    QAbstractTableModel,
    QSortFilterProxyModel,
    QRegularExpression,
    QSize,
)
from PyQt6.QtGui import (
    QColor,
    QPalette,
    QIcon,
)
import os, sys
from pyNexafs.parsers import parser_loaders, parser_base
from pyNexafs.nexafs import scan_base
from typing import Type
import overrides
import warnings
import numpy as np
from pyNexafs.gui.widgets.io.dir_selection import directory_selector


class nexafsFileLoader(QWidget):
    """
    Header-viewer and file loader for NEXAFS data.

    Includes a variety of elements for selecting a directory, parser,
    filetype filters and header-contents filters. Filtering and sorting
    is implemented through QSortFilterProxyModel.


    Attributes
    ----------
    selectionLoaded : pyqtSignal
        Signal for selection change and completion of consequent file loading.
        Does not trigger if selection changes while no parser is selected.
    """

    selectionLoaded = pyqtSignal(bool)
    relabelling = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        ## Instance attributes
        self._log_text = ""

        ## Initialise elements
        self.directory_selector = directory_selector()
        self._progress_bar = QProgressBar()
        self.directory_viewer = directory_viewer_table(progress_bar=self._progress_bar)
        self.nexafs_parser_selector = nexafs_parser_selector()
        self.filter_relabelling = QCheckBox()
        self.filter = directory_filters()
        self.log = QTextEdit()

        ## Subplayouts
        dir_parser_layout = QHBoxLayout()
        dir_parser_layout.addLayout(self.directory_selector)
        dir_parser_layout.addWidget(QLabel("Parser:"))
        dir_parser_layout.addWidget(self.nexafs_parser_selector)
        dir_parser_layout.addWidget(QLabel("Relabel:"))
        dir_parser_layout.addWidget(self.filter_relabelling)

        ## Qsplitter for expandable log
        draggable = QSplitter(Qt.Orientation.Vertical)
        draggable.addWidget(self.directory_viewer)
        draggable_log_widget = QWidget()
        draggable_log = QVBoxLayout()
        draggable_log.addWidget(QLabel("Log:"))
        draggable_log.addWidget(self.log)
        draggable_log_widget.setLayout(draggable_log)
        draggable_log_widget.setContentsMargins(0, 0, 0, 0)
        draggable.addWidget(draggable_log_widget)

        ## Dir loading progress bar
        self._progress_bar.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self._progress_bar.setFixedHeight(15)
        self._progress_bar.setTextVisible(False)

        ## Directory Layout
        dir_layout = QVBoxLayout()
        dir_layout.addLayout(dir_parser_layout)
        dir_layout.addLayout(self.filter)
        dir_layout.addWidget(self._progress_bar)
        dir_layout.addWidget(draggable)

        # Assign viewer layout to widget.
        self.setLayout(dir_layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            dir_layout.setContentsMargins(0, 0, 0, 0)

        ## Element attributes
        self.log.setReadOnly(True)
        self.directory_viewer.directory = (
            self.directory_selector.folder_path
        )  # Match initial value.
        draggable.setStretchFactor(0, 5)
        draggable.setStretchFactor(1, 1)

        ## Element Connections
        # Reload directory upon path change.
        self.directory_selector.newPath.connect(self.update_dir)
        # Reload directory upon parser change.
        self.nexafs_parser_selector.currentIndexChanged.connect(self.update_parser)
        # Reload directory upon filter change.
        self.filter.filterChanged.connect(self.update_filters)
        # Update relabelling
        self.filter_relabelling.stateChanged.connect(self.update_relabelling)
        # Signal for selection change.
        self.directory_viewer.selectionLoaded.connect(self.on_selection_loaded)

        # Initialize directory.
        self._log_entry()

    def on_selection_loaded(self):
        """
        Signal for selection change.
        """
        self.selectionLoaded.emit(True)
        return

    def update_relabelling(self) -> None:
        """
        Updates the fieldnames when relabelling is checked.
        """
        parser = self.directory_viewer.parser
        if parser is not None:
            old_relabel = parser.relabel
            new_relabel = self.filter_relabelling.isChecked()
            if old_relabel != new_relabel:
                parser.relabel = new_relabel
                self.directory_viewer.update_header()
                self.relabelling.emit(parser.relabel)

    def update_dir(self) -> None:
        """
        Update the dir selection
        """
        new_dir = self.directory_selector.folder_path
        # Update directory viewer
        self.directory_viewer.directory = new_dir
        # Log file updates.
        self._log_entry()
        return

    def update_parser(self) -> None:
        """
        Update the parser selection.
        """
        new_parser = self.nexafs_parser_selector.current_parser
        # Disable filter signalling while parser is changed.
        self.filter._changing_parser = True
        # Update directory viewer and filter parser selection
        self.filter.parser = new_parser
        self.directory_viewer.parser = new_parser
        self.filter._changing_parser = False
        self._log_entry()
        return

    def update_filters(self):
        """
        Update the directory viewer.
        """
        # Push an update to the directory viewer.
        self.directory_viewer.filters = (
            self.filter.filter_text,
            self.filter.filetypes_selection,
        )
        self._log_entry()
        return

    def _log_entry(self):
        # Log the current selection
        self._log_selection()
        # Log the number of files.
        self._log_files()

    def _log_selection(self):
        """
        Logs the path, parser and filter selection.
        """
        # Log path selection
        path = self.directory_selector.folder_path
        self.log_text += (
            f"Path '{path}' loaded"
            if len(path) < 30
            else f"Path '{path[:13]}...{path[-13:]}' loaded"
        )

        # Log parser selection
        parser = self.nexafs_parser_selector.current_parser  # current parser selection
        if parser is not None:
            self.log_text += f" using parser '{parser.__name__}'"

        # Log Filetype Selection?
        filetypes = self.filter.filetypes_selection
        if len(filetypes) == 1:
            self.log_text += f" with filetype '{filetypes[0]}'"

        # Log Filter Selection
        filter_text = self.filter.filter_text
        if filter_text != "" and filter_text is not None:
            self.log_text += f" with filter '{filter_text}'"

        # End line
        self.log_text += ". "  # no whitespace, assuming log files will be called next.
        return

    def _log_files(self):
        """
        Logs the number of files located after updating parameters.
        """
        file_count = self.directory_viewer.verticalHeader().count()  # filtered viewport
        dv_f = self.directory_viewer.files
        dv_ph = self.directory_viewer._parser_headers
        dv_p = self.directory_viewer.parser
        # Loading method is phrased differently depending on if a parser is selected.
        method = "parsed" if dv_p is not None and len(dv_ph) >= 0 else "located"
        # Determine endpoint loading.
        if file_count != 0 and len(dv_f) > 0:  # Filtered files loaded
            self.log_text += f"{file_count} {method} files.\n"
        elif file_count != 0 and file_count > 0:  # Filtered files
            self.log_text += f"Located {file_count} files.\n"
        else:
            self.log_text += "No files located.\n"

    @property
    def log_text(self):
        return self._log_text

    @log_text.setter
    def log_text(self, text: str):
        # Check if at max height
        sb = self.log.verticalScrollBar()
        atMaxHeight = True if sb.maximum() - sb.value() < 30 else False
        prevHeight = sb.value()
        # Update log
        self._log_text = text
        self.log.setText(text)
        # If was at max height, return to new max height.
        sb.setValue(sb.maximum() if atMaxHeight else prevHeight)

    @log_text.deleter
    def log_text(self):
        self._log_text = ""
        self.log.setText("")

    @property
    def loaded_parser_headers(self) -> dict[str, Type[parser_base] | None]:
        """
        Returns the parser headers loaded in the directory viewer.

        Returns
        -------
        dict[str, Type[parser_base]]
            A dictionary of filenames and their corresponding parser headers.
        """
        return self.directory_viewer._parser_headers

    @property
    def loaded_parsers_files(self) -> dict[str, Type[parser_base] | None]:
        """
        Returns the full-file parsers loaded in the directory viewer.

        Returns
        -------
        dict[str, Type[parser_base]]
            A dictionary of filenames and their corresponding parser objects.
        """
        return self.directory_viewer._parser_files

    @property
    def loaded_parser_files_selection(self) -> dict[str, Type[parser_base] | None]:
        """
        Returns the selection subset of full-file parsers from the directory viewer.

        Returns
        -------
        dict[str, Type[parser_base]]
            A dictionary of filenames and their corresponding scan objects.
        """
        selected = self.directory_viewer.selected_filenames
        loaded = self.loaded_parsers_files
        return {
            name: loaded[name]
            for name in selected
            if name in loaded and loaded[name] is not None
        }

    @property
    def selected_filenames(self) -> list[str]:
        """
        Returns the selected filenames in the directory viewer.

        Returns
        -------
        list[str]
            A list of filenames selected in the directory viewer.
        """
        return self.directory_viewer.selected_filenames


class nexafs_parser_selector(QComboBox):
    """
    Selector for the type of NEXAFS data to load.
    """

    def __init__(self):
        super().__init__()

        if "" in parser_loaders:
            raise ValueError("Empty string is a reserved parser name.")

        # Define the relationships
        self.parsers = {"": None}
        self.parsers.update(parser_loaders)
        # Add items to the combobox.
        self.addItems([key for key in self.parsers.keys()])

    @property
    def current_parser(self) -> type[parser_base] | None:
        """
        Current selected parser.

        Returns
        -------
        type[parser_base] | None
            Returns the current parser which inherits from parser_base, or None if no parser is selected.
        """
        return self.parsers[self.currentText()]

    def update_combo_list(self):
        """
        Update the combo box parser list from the parser attribute.
        """
        self.clear()
        self.addItems([key for key in self.parsers.keys()])


class directory_filters(QHBoxLayout):
    """
    Generates a QTextEdit & QComboBox for filtering the directory list.
    """

    filterChanged = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()

        # Instance attributes
        self.filter_text = ""
        self.filter_filetype = None
        self._parser_selection = None
        self._changing_parser = False  # Flag to prevent filterChanged signal emitting when parser is changed.

        # Instance widgets
        self.filter_text_edit = QLineEdit()
        self.filter_filetype_select = QComboBox()

        # Setup layout
        self.addWidget(self.filter_text_edit)
        self.addWidget(QLabel("Filetype: "))
        self.addWidget(self.filter_filetype_select)

        # Widget Attributes
        self.filter_text_edit.setPlaceholderText("Filter filename|header by text")

        # Connections
        self.filter_text_edit.editingFinished.connect(self.on_filter_text_edit)
        self.filter_filetype_select.currentIndexChanged.connect(self.on_filter_change)

    def on_filter_text_edit(self):
        """
        Update the filter text.
        """
        self.filter_text = self.filter_text_edit.text()
        self.on_filter_change()

    def on_filter_change(self):
        """
        Signal generator for filter changes if parser not changing.
        """
        if not self._changing_parser:
            self.filterChanged.emit(True)

    @property
    def filetypes_selection(self) -> list[str]:
        """
        Returns the currently selected filetype.

        Returns
        -------
        str
            The currently selected filetype, or a list of all possible filetypes, belonging to the parser.
            If parser is None, returns an empty list.
        """
        if self.parser is None:
            return []
        selection = self.filter_filetype_select.currentText()
        return [selection] if selection != "" else self.parser.ALLOWED_EXTENSIONS

    @property
    def parser(self) -> type[parser_base] | None:
        """
        Returns the currently selected parser.

        Returns
        -------
        type[parser_base] | None
            The currently used parser, or None if no parser is selected.
        """
        return self._parser_selection

    @parser.setter
    def parser(self, parser: type[parser_base] | None):
        """
        Sets the (external) parser selection.

        Additionally triggers the filetype selection to be updated and repopulates the filetype selection entries.

        Parameters
        ----------
        parser : type[parser_base] | None
            Currently used parser, or None if no parser is selected.
        """
        self._changing_parser = True
        self._parser_selection = parser
        self.filter_filetype_select.clear()  # clear existing file extensions.
        if parser is not None:
            self.filter_filetype_select.addItem("")
            self.filter_filetype_select.addItems(parser.ALLOWED_EXTENSIONS)
        self._changing_parser = False

    @parser.deleter
    def parser(self):
        """
        Removes the (external) parser selection.

        Additionally clears the filetype selections.
        """
        self._parser_selection = None
        self.filter_filetype_select.clear()


# Construct the Table Model
class table_model(QAbstractTableModel):

    _status_index = 0  # index for loaded status column.
    _status_header = ""  # string header for the loaded status column.

    def __init__(self, data, header=None):
        super(table_model, self).__init__()
        self._data = data
        self._header = header

        # Initalise graphics for loaded / unloaded files.
        self._icon_error = QApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogCancelButton
        )
        self._icon_success = QApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_DialogApplyButton
        )

    @overrides.overrides
    def headerData(self, section, orientation, role):
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return self._header[section] if len(self._header) > section else None
        else:
            return super().headerData(section, orientation, role)

    @overrides.overrides
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        # Check that second column is being indexed for "loaded" status, to display icons.
        if (
            len(self._header) > self._status_index
            and self._header[self._status_index] == self._status_header
            and index.column() == self._status_index
        ):
            if role == Qt.ItemDataRole.DecorationRole:
                # Icons
                val = (
                    self._icon_success
                    if self._data[index.row()][index.column()]
                    else self._icon_error
                )
                return val
            else:
                # val = "T" if self._data[index.row()][index.column()] else "F")
                return None

        # General data accessing
        if role == Qt.ItemDataRole.DisplayRole:
            return self._data[index.row()][index.column()]
        return None

    @overrides.overrides
    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    @overrides.overrides
    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        if (
            hasattr(self._data, "__len__")
            and len(self._data) > 0
            and hasattr(self._data[0], "__len__")
        ):
            return len(self._data[0])
        else:
            return 0


class directory_viewer_table(QTableView):

    selectionLoaded = pyqtSignal(bool)  # For emitting after selection change is loaded.
    __default_headers_list = [
        "#",
        "Filename",
    ]  # index and filename #TODO: Add "Created", "Modified" when implemented in parser_base.

    def __init__(
        self,
        init_dir: str = None,
        progress_bar: QProgressBar = None,
        parent: QWidget = None,
    ):
        super().__init__(parent)

        # Instance properties
        self.files = []  # list[str]
        self._header_names = None  # list[str]
        ## WE KEEP A DICT OF PARSERS, AND THEN USE THE PARSERS TO GENERATE THE TABLE.
        self._parser_headers = {}  # dict[str, Type[base_parser]]
        self._parser_files = {}  # dict[str, Type[base_parser]]

        self._viewing_directory = None  # string
        self._parser = None  # Type[parser_base]
        self._filetype_filters = None  # list[str]
        self._str_filter = None  # str

        # Instance Widgets
        self.files_model = None
        self.proxy_model = QSortFilterProxyModel()

        # Instance Attributes
        self.verticalHeader().hide()  # remove default row numbers
        self.setModel(self.proxy_model)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSortingEnabled(True)  # enabled by proxymodel.
        self.proxy_model.setSourceModel(None)
        self.proxy_model.setFilterRegularExpression(None)
        self.proxy_model.setFilterKeyColumn(
            -1
        )  # use all columns for filtering instead of just first.
        self.proxy_model.sort(0, Qt.SortOrder.AscendingOrder)
        # self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.resizeColumnToContents(0)
        self.setMinimumWidth(1)
        self.horizontalHeader().setMinimumSectionSize(10)

        # Event connections
        self.selectionModel().selectionChanged.connect(self.load_selection)

        # Progress bar
        self._progress_bar = progress_bar

        # Initialise viewing directory
        if init_dir is not None:
            self.directory = init_dir

        # Setup default header columns if parser is defined. Use header/index definition in table_model.
        self.__default_parsed_headers_list = self.__default_headers_list.copy()
        self.__default_parsed_headers_list.insert(
            self._status_index(), self._status_header()
        )

    @classmethod
    def _status_index(cls) -> int:
        """
        Returns the index of the loaded status column, an attribute of the abstract table model class.
        """
        return table_model._status_index

    @classmethod
    def _status_header(cls) -> str:
        """
        Returns the string header of the loaded status column, an attribute of the abstract table model class.
        """
        return table_model._status_header

    @property
    def _default_headers(self) -> list[str]:
        """
        Returns a copy of the default headers depending on if a parser is defined (adds a loading column).

        Returns
        -------
        list[str]
            List of the default headers
        """
        obj = (
            self.__default_headers_list
            if self.parser is None
            else self.__default_parsed_headers_list
        )
        return obj.copy()

    @property
    def header(self) -> list[str] | None:
        """
        Returns the current header list for parsed filed.

        Returns
        -------
        list[str]
            A list of column names currently observed by the parser.
        """
        return self._header_names

    @property
    def parser(self) -> type[parser_base] | None:
        """
        Returns the currently selected parser.

        Returns
        -------
        type[parser_base] | None
            The currently selected parser, or None if no parser is selected.
        """
        return self._parser

    @parser.setter
    def parser(self, parser: type[parser_base] | None):
        """
        Set the parser, used for tabling header information.

        Resets the directory_viewer_table._file_params and directory_viewer_table.filedata dictionaries.

        Parameters
        ----------
        parser : type[parser_base] | None
            The currently selected parser, or None if no parser is selected.
        """
        self._parser = parser
        # Reset existing file data to use new parser.
        self._parser_files = {}
        self._parser_headers = {}
        self._filetype_filters = (
            parser.ALLOWED_EXTENSIONS if parser is not None else None
        )
        self.update_table()

    @parser.deleter
    def parser(self):
        """
        Remove the parser from the directory viewer.
        """
        self._parser = None
        self._parser_files = {}
        self._parser_headers = {}
        self.update_table()

    @property
    def directory(self):
        """
        Returns the currently viewed directory.

        Returns
        -------
        str
            The currently viewed directory.
        """
        return self._viewing_directory

    @directory.setter
    def directory(self, target: str):
        """
        Set the directory to view.

        Only updates the files list if directory is changed.
        """

        # Check if same directory.
        if self._viewing_directory == target:
            return

        # Get new file list
        self.files = [
            file
            for file in os.listdir(target)
            if os.path.isfile(os.path.join(target, file))
        ]
        # Clear existing data
        self._parser_headers = {}
        self._parser_files = {}

        # Update directory
        self._viewing_directory = target

        # Push update to table.
        self.update_table()

    @property
    def filters(self) -> tuple[str | None, list[str] | None]:
        """
        Returns the currently set filters.

        Returns
        -------
        tuple[str | None, list[str] | None]
            A tuple containing the filename filter and the filetype filter.
        """
        return (self._str_filter, self._filetype_filters)

    @filters.setter
    def filters(self, filters: tuple[str | None, list[str] | None]):
        """
        Set the filters for the directory viewer.

        Parameters
        ----------
        filters : tuple[str | None, list[str] | None]
            A tuple containing the filename filter and the filetype list filter.
        """
        str_filter, filetype_filters = filters

        # Correct type for empty strings.
        if str_filter == "":
            str_filter = None

        # Check if str filter is changed:
        if str_filter != self._str_filter:
            self._str_filter = str_filter
            # Update proxy model
            QRegExp = QRegularExpression(
                str_filter, QRegularExpression.PatternOption.CaseInsensitiveOption
            )
            self.proxy_model.setFilterRegularExpression(
                None if str_filter is None else QRegExp
            )
            # self.proxy_model.setFilterFixedString(str_filter)

        # Check if filetype filter is changed:
        if filetype_filters is not None and isinstance(filetype_filters, list):
            # Check if filetype filter is changed:
            if (
                self._filetype_filters is not None
                and len(self._filetype_filters) == len(filetype_filters)
                and np.all(
                    [
                        (
                            filetype_filters[i] == self._filetype_filters[i]
                            or filetype_filters[i] in self._filetype_filters
                        )
                        for i in range(len(filetype_filters))
                    ]
                )
            ):
                # Everything the same...
                pass
            else:
                self._filetype_filters = filetype_filters
                self.update_table()
        elif filetype_filters is None:
            self._filetype_filters = (
                self._parser.ALLOWED_EXTENSIONS if self._parser is not None else None
            )
        else:
            raise AttributeError(
                f"filters[1] '{filetype_filters}' is not a list of filetype strings or None."
            )

    @filters.deleter
    def filters(self):
        """
        Remove the filters for the directory viewer.
        """
        self._str_filter, self._filetype_filters = None, None
        self.update_table()

    @property
    def progress_bar(self) -> QProgressBar:
        """
        Returns an associated progress bar widget if provided, else None.

        Parameters
        ----------
        progress_bar: QProgressBar
            The progress bar widget.

        Returns
        -------
        QProgressBar
            The progress bar widget.
        """
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, progress_bar: QProgressBar):
        self._progress_bar = progress_bar

    def update_header(self) -> None:
        """
        Generates the column headers using default parser important parameters.

        First attempts to call `summary_param_names_with_units` on the first non-None
        header-loaded parser object, to load the column labels and their appropriate units.
        Alternatively calls `summary_param_names` on the object, which doesn't require unit labels.
        These two methods take into account the parser `relabel` property.
        If no headers are loaded then calls the parser `summary_param_names`.
        """
        # Setup default header columns
        header = self._default_headers  # Get defaults

        # Add list values if parser is defined, and header objects allow unit calls.
        if self.parser is not None and len(self._parser_headers) > 0:
            val = None
            # Collect the first non-None parser object.
            for val in self._parser_headers.values():
                # Collect val (parser objects), where val is not None.
                if val is not None:
                    assert isinstance(val, self.parser)
                    break  # end for loop prematurely.
            # Establish parameter header names:
            if val is not None:
                try:
                    # Attempt to use 'summary_param_names_with_units' if available.
                    obj_head = val.summary_param_names_with_units
                except NotImplementedError as e:
                    warnings.warn(
                        f"{self.parser.__name__} has not implemented 'summary_param_names_with_units'. Defaulting to 'summary_param_names'."
                    )
                    obj_head = val.summary_param_names  # without units.
            else:
                # If all parser headers are None, then use parser default values.
                obj_head = self.parser.summary_param_names
            # Add obj_head to header
            header += obj_head
        # If _parser_headers are not defined but the parser is.
        elif self.parser is not None:
            header += self.parser.summary_param_names

        # Update internal header names
        self._header_names = header

        # Ideally just updates header names, but occasionally crashes when attempting to layoutChange.emit()
        if self.files_model is not None:
            old_selection = self.selectionModel().selectedRows()
            new_model = table_model(
                data=self.files_model._data, header=self._header_names
            )
            self.proxy_model.setSourceModel(new_model)
            self.files_model = new_model
            # Setup and restore selection: requires multiselection to select each row before using regular extended selection.
            self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            [self.selectRow(i.row()) for i in old_selection]
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

        # Update column widths with header update.
        for i in range(0, len(self._header_names)):
            # If the current column width is larger than the size hint, resize to fit for all columns.
            if self.columnWidth(i) > self.sizeHintForColumn(i):
                self.resizeColumnToContents(i)
        return

    def update_table(self):
        """
        Update the directory view.

        Applies text filter, parser and permissionable extensions to filter the files list.
        Loads the filtered header data for the table, appling updates to the class progress bar if defined.

        """
        # Do nothing if an empty directory.
        if self.directory is None:
            return

        ### COLLECT FILES AND APPLY FILETYPE FILTER
        # Collect available files
        files = self.files
        # 1: Filetypes
        if self._filetype_filters is not None and len(self._filetype_filters) > 0:
            files = [
                file
                for file in self.files
                if file.endswith(tuple(self._filetype_filters))
            ]
        # Set the progress bar to zero.
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)
            # Store number of files to be processed for progress bar
            self.progress_bar.setRange(0, len(files) if len(files) > 0 else 1)
        # 2: Load new fileheader data.
        if self.parser is not None:
            # Get header data
            with warnings.catch_warnings(record=True) as recorded_warnings:
                for i, file in enumerate(files):
                    if file not in self._parser_headers:
                        try:
                            self._parser_headers[file] = self.parser(
                                os.path.join(self.directory, file), header_only=True
                            )
                        # Catch unimplemented and import errors.
                        except (NotImplementedError, ImportError, PermissionError) as e:
                            self._parser_headers[file] = None
                    if self.progress_bar is not None:
                        # self.progress_bar.setValue(int((i + 1) / files_len * 100))
                        self.progress_bar.setValue(i + 1)
            # Process the caught warnings
            warning_categories = {}
            warn_re = re.compile(
                r"Attempted method '(.*)' failed to load '(.*)' from '(.*)' with (.*)."
            )
            for w in recorded_warnings:
                matches = warn_re.search(str(w.message))
                if matches and len(matches.groups()) == 4:
                    method, file, parser, error = matches.groups()
                    # Collect warning
                    parser_method = f"{parser}.{method}"
                    filetype = file.split(".")[-1]
                    filetype_error = f"{filetype}:{error}"

                    if parser_method not in warning_categories:
                        # Add parser_method to categories
                        warning_categories[parser_method] = {filetype_error: 1}
                    elif filetype_error not in warning_categories[parser_method]:
                        # Add filetype_error to parser_method
                        warning_categories[parser_method][filetype_error] = 1
                    else:
                        # Increment filetype_error count
                        warning_categories[parser_method][filetype_error] += 1
                else:
                    # re-issue warning
                    warnings.warn_explicit(
                        message=w.message,
                        category=w.category,
                        filename=w.filename,
                        lineno=w.lineno,
                        source=w.source,
                    )
            # Print warning categories
            for parser_method, file_errors in warning_categories.items():
                for filetype_error, count in file_errors.items():
                    filetype, error = filetype_error.split(":")
                    print(
                        f"Parser method '{parser_method}' failed {count} times on filetype '{filetype}' with error '{error}'."
                    )
        else:
            # If parser is None, empty the existing data.
            self._parser_headers = {}

        # Update headers
        self.update_header()

        # Get header data for table:
        data = []  # list of file header lists.
        excess = [
            "" for _ in range(len(self._header_names) - len(self._default_headers))
        ]  # empty columns for invalid file headers.
        for i, file in enumerate(files):
            filedata = [i + 1, file]  # add index (1, ...) and filename to data.
            # If parser specified (for loading) add field for successful loading.
            if self.parser:
                status = (
                    True
                    if file in self._parser_headers
                    and self._parser_headers[file] is not None
                    else False
                )
                filedata.insert(self._status_index(), status)

            # Raise an error if mismatch between info and header variables.
            if len(filedata) != len(self._default_headers):
                raise AttributeError("Default headers do not match filedata.")

            # Add parameter values
            if file in self._parser_headers and isinstance(
                self._parser_headers[file], parser_base
            ):
                # Match indexing of parser.SUMMARY_PARAM_RAW_NAMES
                filedata.extend(self._parser_headers[file].summary_param_values)
            else:
                filedata.extend(excess)
            # Add header data to data list.
            data.append(filedata)

        # New table - Ideally would be a model update, but would crash when attempting to layoutChange.emit()
        self.files_model = table_model(data=data, header=self._header_names)
        # Update proxy (filtering / sorting) model
        self.proxy_model.setSourceModel(self.files_model)

        self.resizeColumnToContents(0)
        for i in range(0, len(self._header_names)):
            # If the current column width is larger than the size hint, resize to fit for all columns.
            if self.columnWidth(i) > self.sizeHintForColumn(i):
                self.resizeColumnToContents(i)
        if self.progress_bar is not None:
            self.progress_bar.setValue(0)
        return

    def load_selection(self) -> None:
        """
        Loads full data from files of the current selection.

        Instead of just header information, loads all data lines
        from the selected files. Loaded datafiles are cached in
        self._parser_files, until the parser object is changed.
        """
        sm = self.selectionModel()
        # Require selection and parser to trigger load data.
        if sm.hasSelection() and self.parser is not None:
            rows = sm.selectedRows(
                column=1 if self._status_index() > 1 else 2
            )  # get filename, not index.

            # Set the progress bar to zero.
            if self.progress_bar is not None:
                self.progress_bar.setValue(0)
                # Store number of files to be processed for progress bar
                self.progress_bar.setRange(0, len(rows))

            # Setup warnings to show once.
            warnings.simplefilter("once", UserWarning)
            for i, row in enumerate(rows):
                filename = self.proxy_model.data(row, Qt.ItemDataRole.DisplayRole)
                # Check if the file already loaded:
                if (
                    filename not in self._parser_files
                    or self._parser_files[filename] is None
                ):
                    # Check for a header object (should always exist).
                    if (
                        filename in self._parser_headers
                        and self._parser_headers[filename] is not None
                    ):
                        # Use existing parser instance, add to files.
                        parser = self._parser_headers[filename]
                        assert isinstance(parser, parser_base)
                        try:
                            parser.load()  # use internal filepath to load
                            if parser.is_loaded:
                                self._parser_files[filename] = parser
                        except Exception as e:
                            traceback.print_exception(e)
                    # else: ignore unloaded headers (None) in the selection.
                if self.progress_bar is not None:
                    self.progress_bar.setValue(i + 1)
            # Restore warnings functionality.
            warnings.resetwarnings()
            # Trigger a loading completed signal.
            self.progress_bar.setValue(0)
            self.selectionLoaded.emit(True)
        return

    @property
    def selected_filenames(self) -> list[str]:
        """
        Returns the selected filenames in the directory viewer.

        Returns
        -------
        list[str]
            A list of filenames selected in the directory viewer.
        """
        sm = self.selectionModel()
        if sm.hasSelection():
            rows = sm.selectedRows(
                column=self._index_filename_column
            )  # Filename in second column.
            # return self.files_model.data(rows, Qt.ItemDataRole.DisplayRole)
            return [
                self.proxy_model.data(row, Qt.ItemDataRole.DisplayRole) for row in rows
            ]
        return []

    @property
    def _index_filename_column(self) -> int:
        """
        Returns the index of the filename column in the table model.
        """
        if self.parser is None:
            return 1
        else:
            return 2 if self._status_index() < 2 else 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = nexafsFileLoader()
    window.show()
    window.setWindowTitle("pyNexafs File Loader")
    sys.exit(app.exec())
