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
)
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QTextEdit,
    QCheckBox,
    QAbstractItemView,
    QHeaderView,
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
from PyQt6.QtGui import QColor, QPalette
import os, sys
from pyNexafs.parsers import parser_loaders, parser_base
from pyNexafs.nexafs import scan_base
from typing import Type
import warnings
import numpy as np


class nexafs_fileloader(QWidget):
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

    def __init__(self):
        super().__init__()

        ## Instance attributes
        self._log_text = ""

        ## Initialise elements
        self.directory_selector = directory_selector()
        # self.directory_viewer = directory_viewer_list()
        self.directory_viewer = directory_viewer_table()
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

        ## Directory Layout
        dir_layout = QVBoxLayout()
        dir_layout.addLayout(dir_parser_layout)
        dir_layout.addLayout(self.filter)
        dir_layout.addWidget(self.directory_viewer)
        dir_layout.addWidget(QLabel("Log:"))
        dir_layout.addWidget(self.log)

        # Assign viewer layout to widget.
        self.setLayout(dir_layout)

        ## Element attributes
        self.log.setReadOnly(True)
        self.directory_viewer.directory = (
            self.directory_selector.folder_path
        )  # Match initial value.
        dir_layout.setStretch(2, 3)  # viewer stretch 3x
        dir_layout.setStretch(4, 1)  # log stretch 1x

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

    def update_relabelling(self):
        """
        Updates the fieldnames when relabelling is checked.
        """
        parser = self.directory_viewer.parser
        if parser is not None:
            parser.relabel = self.filter_relabelling.isChecked()
            self.directory_viewer.relabel_header()

    def update_dir(self):
        """
        Update the dir selection
        """
        new_dir = self.directory_selector.folder_path
        # Update directory viewer
        self.directory_viewer.directory = new_dir
        # Log file updates.
        self._log_entry()
        return

    def update_parser(self):
        """
        Update the parser selection.
        """
        new_parser = self.nexafs_parser_selector.current_parser
        # Disable filter signalling while parser is changed.
        self.filter._changing_parser = True
        # Update directory viewer and filter parser selection
        self.directory_viewer.parser = new_parser
        self.filter.parser = new_parser
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
        if parser is None:
            self._log_text += (
                ". "  # no whitespace, assuming log files will be called next.
            )
            return

        # If parser selected, continue!
        self.log_text += f" using parser '{parser.__name__}'"

        # Log Filetype Selection?
        filetypes = self.filter.filetypes_selection
        if len(filetypes) == 1:
            self.log_text += f" with filetype '{filetypes[0]}'"

        # Log Filter Selection
        filter_text = self.filter.filter_text
        if filter_text != "":
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
    def loaded_selection(self) -> dict[str, Type[parser_base] | None]:
        """
        Returns the selection subset of full-file parsers from the directory viewer.

        Returns
        -------
        dict[str, Type[scan_base]]
            A dictionary of filenames and their corresponding scan objects.
        """
        selected = self.directory_viewer.selected_filenames()
        loaded = self.loaded_parsers_files
        return {name: loaded[name] for name in selected if name in loaded}

    @property
    def selected_filenames(self) -> list[str]:
        """
        Returns the selected filenames in the directory viewer.

        Returns
        -------
        list[str]
            A list of filenames selected in the directory viewer.
        """
        return self.directory_viewer.selected_filenames()


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


class directory_selector(QHBoxLayout):
    """
    Module for selecting a directory path.

    Parameters
    ----------
    QWidget : _type_
        _description_
    """

    newPath = pyqtSignal(bool)

    def __init__(self):
        super().__init__()

        # Instance attributes
        self.folder_path = os.path.expanduser("~")  # default use home path.
        self.folder_path = directory_selector.format_path(
            self.folder_path
        )  # adds final slash if not present.

        ### Instance widgets
        # Folder path
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setText(self.folder_path)
        self.folder_path_edit.accessibleName = "Directory"
        self.folder_path_edit.accessibleDescription = "Path to load NEXAFS data from."
        self.folder_path_edit.editingFinished.connect(self.validate_path)
        # Folder select button
        self.folder_select_button = QPushButton("Browse")
        self.folder_select_button.clicked.connect(self.select_path)

        # Setup layout
        self.addWidget(self.folder_path_edit)
        self.addWidget(self.folder_select_button)

    def select_path(self):
        """
        Generates a file dialog to select a directory, then updates the path.
        """
        path = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select NEXAFS Directory",
            directory=self.folder_path,
            options=QFileDialog.Option.ShowDirsOnly,
        )
        path = None if path == "" else path
        if path is None:
            # invalid path
            self.folder_path_edit.setText(
                self.folder_path
            )  # reset the text to the original path.
        else:
            # set editable path and validate.
            self.folder_path_edit.setText(path)
            self.validate_path()

    def validate_path(self):
        """
        Validate a manual entry path.

        Only updates internal path if the path is valid, otherwise sets background color to red.
        """
        # Get path text
        editable_path = directory_selector.format_path(self.folder_path_edit.text())

        # Check validity
        if editable_path is None:
            # invalid path
            self.folder_path_edit.setText(
                self.folder_path
            )  # reset the text to the original path.
            return

        if not os.path.isdir(editable_path):
            # invalid path
            self.folder_path_edit.setStyleSheet("background-color: red;")
        else:
            # Valid path

            # Check if path has changed
            new_path = False
            if editable_path != self.folder_path:
                # if path has changed, perform extra functions...
                new_path = True
            self.folder_path_edit.setStyleSheet("background-color: white;")
            self.folder_path = editable_path
            self.folder_path_edit.setText(editable_path)

            if new_path:
                self.newPath.emit(True)

    @staticmethod
    def format_path(path: str) -> str:
        """
        Formats a path string to be consistent.

        Parameters
        ----------
        path : str
            The incoming path string. Must be a valid path, tested by os.path.isdir().

        Returns
        -------
        str
            A formatted path string that always contains a trailing slash, with slashes matching OS type.
        """

        # Strip whitespace, ensure tailing slash, and convert mixed slashes to forward slashes.
        formatted_path = os.path.join("", path.strip(), "").replace("\\", "/")
        # Remove redundant slash duplicates
        while "//" in formatted_path:
            formatted_path = formatted_path.replace("//", "/")
        # Convert slashes to match OS
        slashes = os.sep
        formatted_path = formatted_path.replace("/", slashes)

        return formatted_path


# Construct the Table Model
class table_model(QAbstractTableModel):
    def __init__(self, data, header=None):
        super(table_model, self).__init__()
        self._data = data
        self._header = header
        # if header is not None:
        #     for i in range(len(header)):
        #         self.setHeaderData(i, Qt.Orientation.Horizontal, header[i])

    def headerData(self, section, orientation, role):
        if (
            orientation == Qt.Orientation.Horizontal
            and role == Qt.ItemDataRole.DisplayRole
        ):
            return self._header[section] if len(self._header) > section else None
        else:
            return super().headerData(section, orientation, role)

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]
        return None

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

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
    default_headers = ["#", "Filename"]  # index and filename

    def __init__(self, init_dir=None):
        super().__init__()

        # Instance properties
        self.files = []  # list[str]
        self._filtered_files = []  # list[str]
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

        # Event connections
        self.selectionModel().selectionChanged.connect(self.load_selection)

        # Initialise viewing directory
        if init_dir is not None:
            self.directory = init_dir

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
        self._parser_files = None
        self._parser_headers = None
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
        self._parser_files = None
        self._parser_headers = None
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
        if filetype_filters is not None:
            # Check if filetype filter is changed:
            if self._filetype_filters is not None and np.all(
                [
                    self._filetype_filters[i] == filetype_filters[i]
                    for i in range(len(filetype_filters))
                ]
            ):
                # Everything the same...
                pass
            else:
                self._filetype_filters = filetype_filters
                self.update_table()

    @filters.deleter
    def filters(self):
        """
        Remove the filters for the directory viewer.
        """
        self._str_filter, self._filetype_filters = None, None
        self.update_table()

    def relabel_header(self):
        if self.parser is not None and self._parser_headers is not None:
            header = None  # intially set to None, to assign default value if no headers are found.
            if self._parser_headers is not None and len(self._parser_headers) > 0:
                for key, val in self._parser_headers.items():
                    # Collect first key, val where val is not None.
                    if val is not None:
                        # Attempt to use 'summary_param_names_with_units' if available.
                        try:
                            header = (
                                self.default_headers
                                + val.summary_param_names_with_units
                            )
                        except NotImplementedError as e:
                            warnings.warn(
                                f"{self.parser.__name__} has not implemented 'summary_param_names_with_units'. Defaulting to 'summary_param_names'."
                            )
                            header = (
                                self.default_headers + val.summary_param_names
                            )  # without units.
                        break  # end for loop prematurely.
            # Assign default value if None.
            if header is None:
                header = self.default_headers + self.parser.SUMMARY_PARAM_RAW_NAMES
            self._header_names = header  # internal header object
        else:
            if self.parser is None:
                # Default for no parser
                self._header_names = self.default_headers
            else:
                # Parser exists, but names not initialised.
                self._header_names = (
                    self.default_headers + self.parser.SUMMARY_PARAM_RAW_NAMES
                )

        # Model header update
        if self.files_model is not None:
            self.files_model._header = self._header_names
            self.files_model.layoutChanged.emit()
        return

    def update_table(self):
        """
        Update the directory view.
        """
        # Do nothing if an empty directory.
        if self.directory is None:
            return

        # Collect available files
        files = self.files

        ## Apply filetype filter(s) to files
        # Filetypes
        if self._filetype_filters is not None and len(self._filetype_filters) > 0:
            files = [
                file
                for file in self.files
                if file.endswith(tuple(self._filetype_filters))
            ]

        # Collect the new fileheader data.
        if self.parser is not None:
            if self._parser_headers is None:
                self._parser_headers = {}
            # Get header data
            for file in files:
                if file not in self._parser_headers:
                    try:
                        self._parser_headers[file] = self.parser(
                            os.path.join(self.directory, file), load_head_only=True
                        )
                    except NotImplementedError as e:
                        self._parser_headers[file] = None
        else:
            # If parser is None, empty the existing data.
            self._parser_headers = None

        # Update headers
        self.relabel_header()

        # Get header data for table:
        data = []
        excess = [
            "" for _ in range(len(self._header_names) - len(self.default_headers))
        ]  # empty columns for invalid file headers.
        for i in range(len(files)):
            file = files[i]

            filedata = [i + 1, file]  # add index (1, ...) and filename to data.
            if len(filedata) != len(self.default_headers):
                raise AttributeError("Default headers do not match filedata.")

            if self._parser_headers is not None:
                if self._parser_headers[file] is not None:
                    # Match indexing of parser.SUMMARY_PARAM_RAW_NAMES
                    spv = self._parser_headers[file].summary_param_values
                    for j in range(len(self._header_names) - len(self.default_headers)):
                        filedata.append(spv[j])
                else:
                    filedata.extend(excess)
            else:
                filedata.extend(excess)
            data.append(filedata)

        # Add files to table if exists
        if self.files_model is None:
            # New table
            self.files_model = table_model(data=data, header=self._header_names)
            # Update proxy (filtering / sorting) model
            self.proxy_model.setSourceModel(self.files_model)
        else:
            # Update table
            self.files_model._data = data
            self.files_model._header = self._header_names
            self.files_model.layoutChanged.emit()
        self.resizeColumnToContents(0)
        for i in range(1, len(self._header_names)):
            # If the current column width is larger than the size hint, resize to fit.
            if self.columnWidth(i) > self.sizeHintForColumn(i):
                self.resizeColumnToContents(i)
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
            rows = sm.selectedRows(column=1)  # get filename, not index.
            for row in rows:
                filename = self.proxy_model.data(row, Qt.ItemDataRole.DisplayRole)
                # Initialize parser_files if not already defined.
                if self._parser_files is None:
                    self._parser_files = {}
                # Check if the file already loaded:
                if (
                    filename not in self._parser_files
                    or self._parser_files[filename] is None
                ):
                    # Is the header is loaded
                    if (
                        filename not in self._parser_headers
                        or self._parser_headers[filename] is None
                    ):
                        # Create new parser instance, add to headers and files.
                        parser = self.parser(os.path.join(self.directory, filename))
                        self._parser_headers[filename] = parser
                        self._parser_files[filename] = parser
                    else:
                        # Use existing parser instance, add to files.
                        parser = self._parser_headers[filename]
                        assert isinstance(parser, parser_base)
                        parser.load()  # use internal filepath to load
                        self._parser_files[filename] = parser
            # Trigger a loading completed signal.
            self.selectionLoaded.emit(True)
        return

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
            rows = sm.selectedRows(column=1)  # Filename in second column.
            # return self.files_model.data(rows, Qt.ItemDataRole.DisplayRole)
            return [
                self.files_model.data(row, Qt.ItemDataRole.DisplayRole) for row in rows
            ]
        return []


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
        self.filter_text_edit.setPlaceholderText("Filter filename by text")

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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = nexafs_fileloader()
    window.show()
    window.setWindowTitle("pyNexafs File Loader")
    sys.exit(app.exec())
