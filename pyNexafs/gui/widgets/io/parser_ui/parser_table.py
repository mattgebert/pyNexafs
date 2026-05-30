"""
The table model and view for the Parser Loading module.
"""

from enum import StrEnum
from typing import overload, override, Any
from pyNexafs.gui.config import QtCore, QtWidgets, QtGui
from typing import Sequence, Literal
from pyNexafs.parsers import parserBase
from abc import abstractmethod
import typing
import os
import datetime
import inspect


def _safe_style_icon(
    pixmap: QtWidgets.QStyle.StandardPixmap | None = None,
) -> QtGui.QIcon:
    """Return a standard icon without assuming a QApplication already exists."""
    if pixmap is None:
        return QtGui.QIcon()
    style = QtWidgets.QApplication.style()
    if style is not None:
        return style.standardIcon(pixmap)

    app = QtWidgets.QApplication.instance()
    if app is None:
        raise RuntimeError(
            "No QApplication instance found. Cannot retrieve style for standard icon."
        )

    if isinstance(app, QtWidgets.QApplication):
        style = app.style()
    if style is None:
        fallback_style = QtWidgets.QStyleFactory.create("Fusion")
        if fallback_style is not None:
            return fallback_style.standardIcon(pixmap)
        return QtGui.QIcon()
    return style.standardIcon(pixmap)


class loadStatus(StrEnum):
    """
    The enumerate class for the load status of the file.
    """

    NONE = "None"
    ERROR = "Error Loading"
    HEADER_ONLY = "Header Only Loaded"
    FULL_DATA = "Full Data Loaded"


# Define data types for rows
data_row_parser = tuple[loadStatus, int, str, *tuple[Any, ...]]
data_row_no_parser = tuple[int, str, *tuple[Any, ...]]
data_row_type = data_row_parser | data_row_no_parser


class ParserDataModel:
    """
    The base model for the parser loading module, containing the data and headers to be displayed in the table.

    Parameters
    ----------
    data : Sequence[tuple[Any]]
        A sequence of tuples containing the data to be displayed in the table.
    header : Sequence[str] | None, optional
        A sequence containing the headers to be displayed in the table, by default None.
    progress_bar : QtWidgets.QProgressBar | None, optional
        A progress bar widget to be updated during file loading / parsing, by default None.
    """

    loadStatus = loadStatus

    def __init__(
        self,
        path: str,
        parser_cls: type[parserBase] | None = None,
        progress_bar: QtWidgets.QProgressBar | None = None,
    ):
        self._parser = parser_cls
        """The parser class to be used for loading the file, if any."""

        self._param_selection: None | list[str] = None
        """
        The parameters to be displayed in the table, if any.

        This is used for the column headers after the load status and filename, and by default is determined by the parser class used for loading the file.

        Can also be modified via the `param_selection` property, which will update the table data accordingly.
        """

        self._param_selection_default: None | list[str] = None
        """The default parameters determined by the parser class."""

        self._param_selection_possible: list[str] = []
        """The list of possible parameters that are available for display in the table, determined by the parsed files."""

        self._path = path
        """The path to the file being represented by this model."""

        self.parsed: dict[str, tuple[ParserDataModel.loadStatus, parserBase]] = {}
        """A dictionary mapping filepaths to their corresponding parser instances, for caching loaded files."""

        self._data: list[data_row_no_parser] | list[data_row_parser] = []
        """
        A list of tuples containing the data to be displayed in the table.

        If a parser is defined, the first column is the load status. The subsequent columns are constant, the file index number (alphabetical), the filename, and the parameters specified in `param_selection` (or the default parameters determined by the parser class if `param_selection` is None).

        The raw data is always sorted by the second column, which is the filename.
        """

        self._header: list[str]
        """A sequence containing the headers to be displayed in the table."""

        self._filelist: tuple[str, ...]
        """A tuple containing the list of files scanned in the current path."""

        self._filetype_filters: list[str] | None = None
        """A list of file extensions to be used as filters for determining which files to include in the file list and table, or None."""

        self._show_cached = False
        """Whether to show cached files in the file list or not."""

        self.progress_bar: QtWidgets.QProgressBar | None = progress_bar
        """A progress bar widget to be updated during file loading / parsing, or None if no progress bar is used."""

        # Initialize if conditions are met:
        if self._parser is not None:
            self.make_default_selection()  # Set the default parameter selection based on the parser.

        if self._path is not None:
            self.load_path()  # Load the file list for the initial path.

        self.column_idxs_datetime: set[int]
        """Set of column indices corresponding to datetime parameters, for appropriate formatting in the table view."""
        self.column_idxs_bytes: set[int]
        """Set of column indices corresponding to byte parameters, for appropriate formatting in the table view."""

    def load_path(self, path: str | None = None) -> None:
        """
        Loads the file at the specified path using the set parser, and returns the corresponding parser instance.

        Parameters
        ----------
        path : str | None, optional
            The path to the file to be loaded, or None to use the current path of the model, by default None.
        """
        # Check for new changes
        curr_path = self._path
        if path is not None:
            if curr_path is not None and os.path.abspath(path) == os.path.abspath(
                curr_path
            ):
                return  # No change in path, do nothing.
            # Update the path.
            self.path = path  # This will validate the path and set it to the model.
            self.layoutAboutToChange()  # Signal the view that the layout is about to change, to prepare for the new data.

        # Get the current path and parser from the model.
        load_path = self.path  # Get the current path from the model.
        parser = self.parser  # Get the current parser from the model.

        files = os.listdir(load_path)
        if parser is None:
            self._filelist = tuple(files)
        else:
            # Collect the list of files
            self._filelist = tuple(
                f
                for f in files
                if any(f.endswith(ext) for ext in parser.ALLOWED_EXTENSIONS)
            )

            # Reset progress
            if self.progress_bar is not None:
                self.progress_bar.setMaximum(len(self._filelist))
                self.progress_bar.setValue(0)

            # Parse each file.
            for i, filename in enumerate(self._filelist):
                filepath = os.path.join(load_path, filename)
                if (
                    filepath in self.parsed
                    and self.parsed[filepath] is not None
                    and self.parsed[filepath][1].__class__ == parser
                ):
                    continue  # Skip files that have already been parsed (using the current parser) and cached.
                try:
                    # Load headers only
                    parser_instance = parser(filepath, header_only=True)
                    self.parsed[filepath] = (
                        ParserDataModel.loadStatus.HEADER_ONLY,
                        parser_instance,
                    )
                except Exception as e:
                    print(f"Error parsing file '{filepath}': {e}")

                # Update progress
                if self.progress_bar is not None:
                    self.progress_bar.setValue(i + 1)

        # Collect the set of parameters that are available for display in the table, based on the parsed files.
        self.make_param_selection_possible()

        # Create the table data and headers based on the current filelist and parser settings.
        self.construct_table()

        self.layoutChange()

    @property
    def filelist(self) -> tuple[str, ...]:
        """
        The list of files scanned in the current path.

        `load_path` already filters files based on the current parser selection, but `filetype_filters` can be used to further filter to a specific subset of file types, if desired.

        Returns
        -------
        tuple[str, ...]
            A tuple containing the list of files scanned in the current path.
        """
        return tuple(
            f
            for f in self._filelist
            if (
                self.filetype_filters is None
                or any(f.endswith(ext) for ext in self.filetype_filters)
            )
        )

    @property
    def filetype_filters(self) -> list[str] | None:
        """
        An additional filter for determining file inclusion, based on file extensions.

        Parameters
        ----------
        filetype_filters : list[str] | None
            A list of file extensions to be used as filters for determining which files to include in the file list and table, or None to clear the filetype filter.

        Returns
        -------
        list[str] | None
            A list of file extensions to be used as filters for determining which files to include in the file list and table, or None if no filetype filter is set.
        """
        return self._filetype_filters

    @filetype_filters.setter
    def filetype_filters(self, filetype_filters: list[str] | None):
        self._filetype_filters = filetype_filters
        self.load_path()  # Reload the path to update the file list based on the new filetype filters.

    @property
    def parser(self) -> type[parserBase] | None:
        """
        The parser class to be used for loading the file, if any.

        Parameters
        ----------
        parser_cls : type[parserBase] | None
            The parser class to be set for loading the file, or None to clear the parser.

        Returns
        -------
        type[parserBase] | None
            The parser class currently set for loading the file, or None if no parser is set.
        """
        return self._parser

    @parser.setter
    def parser(self, parser_cls: type[parserBase] | None):
        self._parser = parser_cls
        self.make_default_selection()  # Update the default parameter selection based on the new parser.

    @parser.deleter
    def parser(self):
        self._parser = None
        self._param_selection_default = None

    @property
    def path(self) -> str:
        """
        Path property for generating data and header information for the files being represented by this model.

        Parameters
        ----------
        path : str
            The path to be set for the file being represented by this model.

        Returns
        -------
        str
            The path to the file being represented by this model.

        Raises
        ------
        ValueError
            If the provided path is not a valid directory.
        """

        return self._path

    @path.setter
    def path(self, path: str):
        if not os.path.isdir(path):
            raise ValueError(f"Provided path '{path}' is not a valid directory.")
        self._path = path
        self.load_path()  # Load the file list for the new path.

    def make_param_selection_possible(self):
        """
        Construct the list of possible parameters that are available for display in the table, based on the parsed files.

        This method should be called after loading the files and parsing their headers, to update the list of possible parameters for display in the table. If `show_cached` is True, this will include all parsed files in the cache, otherwise it will only include the files that were parsed during the current loading process. The list of possible parameters is determined by collecting the keys from the `params` attribute of the parser instances for each file, and taking the union of these keys across all files.
        """
        possible_params = []
        if self._show_cached:
            for load_status, parser_instance in self.parsed.values():
                if (
                    load_status != ParserDataModel.loadStatus.ERROR
                    or load_status != ParserDataModel.loadStatus.NONE
                ) and parser_instance is not None:
                    for param in parser_instance.params.keys():
                        if param not in possible_params:
                            possible_params.append(param)
        else:
            for filename in self._filelist:
                filepath = os.path.join(self._path, filename)
                if filepath in self.parsed:
                    load_status, parser_instance = self.parsed[filepath]
                    if (
                        load_status != ParserDataModel.loadStatus.ERROR
                        or load_status != ParserDataModel.loadStatus.NONE
                    ) and parser_instance is not None:
                        for param in parser_instance.params.keys():
                            if param not in possible_params:
                                possible_params.append(param)
        self._param_selection_possible = possible_params

    @overload
    def _construct_data_rows(
        self, filepaths: list[str] | None, with_headers: Literal[False]
    ) -> list[data_row_parser] | list[data_row_no_parser]: ...

    @overload
    def _construct_data_rows(
        self, filepaths: list[str] | None, with_headers: Literal[True]
    ) -> tuple[list[data_row_parser] | list[data_row_no_parser], list[str]]: ...

    def _construct_data_rows(
        self, filepaths: list[str] | None = None, with_headers: bool = False
    ) -> (
        list[data_row_parser]
        | list[data_row_no_parser]
        | tuple[list[data_row_no_parser] | list[data_row_parser], list[str]]
    ):
        """
        Constructs data rows for specific files.

        Parameters
        ----------
        filepaths : list[str] | None, optional
            A list of file paths for which to construct data rows, or None to construct data rows for all files in the current file list, by default None.
        with_headers : bool, optional
            Whether to also return the headers along with the data rows, by default False.
            This adds a return element to the output.

        Returns
        -------
        list[data_row_parser] | list[data_row_no_parser] | tuple[
            list[data_row_parser] | list[data_row_no_parser],
            list[str]
        ]
            If `with_headers`, returns a tuple containing a list of the data rows (either with or without parser loading information)
        """
        # Collect the parser
        parser = self.parser

        # Find the parameters to collect
        param_sel = self.param_selection
        specials = ["filesize", "created", "modified"]
        """The special parameters that are always available for display in the table, regardless of the parser parameters, and can be included in the parameter selection for display in the table."""

        if param_sel is None:
            # Default parameters if no parser and no default selection, to have some information in the table.
            param_sel = specials.copy()
            defaults = self._param_selection_default
            if defaults is not None:
                for key in defaults:
                    if key not in param_sel:
                        param_sel.append(key)

        if filepaths is None:
            filepaths = [
                os.path.join(self._path, filename) for filename in list(self._filelist)
            ]

        # Collect and collate the data
        parser_instance: parserBase | None = None
        new_data = list[data_row_no_parser] | list[data_row_parser]
        if parser is not None:
            new_data_parser: list[data_row_parser] = []
            for i, filepath in enumerate(filepaths):
                filename = os.path.basename(filepath)
                if filepath in self.parsed:
                    # The file has already been parsed, use cached result.
                    parser_instance = self.parsed[filepath][1]
                elif parser is not None:
                    # Try to parse the file
                    try:
                        parser_instance = parser(filepath)
                    except Exception:
                        parser_instance = None
                else:
                    # No parser, don't load the file.
                    parser_instance = None

                # Assign the load_status
                if parser_instance is None:
                    load_status = self.loadStatus.ERROR
                elif parser_instance.is_loaded:
                    load_status = self.loadStatus.FULL_DATA
                else:
                    load_status = self.loadStatus.HEADER_ONLY

                if parser_instance is not None:
                    filesize = parser_instance.params.get(
                        "filesize", os.path.getsize(filepath)
                    )
                    modified = parser_instance.params.get(
                        "modified", os.path.getmtime(filepath)
                    )
                    created = parser_instance.params.get(
                        "created", os.path.getctime(filepath)
                    )
                else:
                    filesize = os.path.getsize(filepath)
                    modified = os.path.getmtime(filepath)
                    created = os.path.getctime(filepath)

                # format as datetime
                modified = (
                    datetime.datetime.fromtimestamp(modified)
                    if isinstance(modified, (int, float))
                    else modified
                )
                created = (
                    datetime.datetime.fromtimestamp(created)
                    if isinstance(created, (int, float))
                    else created
                )
                specials_data = {
                    "filesize": filesize,
                    "modified": modified,
                    "created": created,
                }

                row_data_parsed = [load_status, i, filename]
                for key in param_sel:
                    if key in specials:
                        value = specials_data[key]
                    elif (
                        parser_instance is not None
                        and parser_instance.params is not None
                    ):
                        try:
                            # Use the custom method for the parser value if available.
                            value = parser_instance.param_value(key)
                        except (ValueError, NotImplementedError, KeyError):
                            # if the value isn't found (keyerror)
                            # or the method isn't implemented for this parser (notimplementederror)
                            # or specific value not valid.
                            value = parser_instance.params.get(key)
                    else:
                        value = None
                    row_data_parsed.append(value)
                row_data_typed: data_row_parser = tuple(row_data_parsed)
                new_data_parser.append(row_data_typed)
            # Sort by the filename (third column).
            new_data_parser = sorted(new_data_parser, key=lambda x: x[2])
            new_data = new_data_parser  # Set the new data to the parser data format.
        else:
            new_data_no_parser: list[data_row_no_parser] = []
            for i, filepath in enumerate(filepaths):
                filename = os.path.basename(filepath)
                filesize = os.path.getsize(filepath)
                modified = os.path.getmtime(filepath)
                created = os.path.getctime(filepath)

                # format as datetime
                modified = (
                    datetime.datetime.fromtimestamp(modified)
                    if isinstance(modified, (int, float))
                    else modified
                )
                created = (
                    datetime.datetime.fromtimestamp(created)
                    if isinstance(created, (int, float))
                    else created
                )
                specials_data = {
                    "filesize": filesize,
                    "modified": modified,
                    "created": created,
                }
                row_data_no_parser: data_row_no_parser = (
                    i,
                    filename,
                    filesize,
                    modified,
                    created,
                )
                new_data_no_parser.append(row_data_no_parser)
            # Sort by the filename (second column when no parser).
            new_data_no_parser = sorted(new_data_no_parser, key=lambda x: x[1])
            new_data = (
                new_data_no_parser  # Set the new data to the no parser data format.
            )

        # Process the headers to show units as well if available.
        if not with_headers:
            return new_data
        else:
            if parser_instance is not None and parser_instance.params is not None:
                for i, key in enumerate(param_sel):
                    try:
                        param_sel[i] = parser_instance.param_name_with_unit(key)
                    except (ValueError, NotImplementedError, KeyError):
                        # Do not redefine the header name with a unit.
                        pass

            # Construct the header and the sets of column indices for special formatting.
            if parser is None:
                header = ["#", "Filename", *param_sel]
            else:
                header = ["", "#", "Filename", *param_sel]
            return new_data, header

    def construct_table(self) -> None:
        """
        Construct data and headers based on the current filelist and parser settings.

        Constructs from scratch, resetting existing data. Utilizes `self._filelist` to generate `self._data` and `self._header` for the table view. The `self._show_cached` attribute determines whether to include files that have already been parsed and cached in `self._parsed` when constructing the table data.
        """

        new_data, headers = self._construct_data_rows(filepaths=None, with_headers=True)
        # Assign the data and headers
        self._data = new_data
        self._header = headers
        # Calculate the sets of column indices for special formatting based on the headers.
        self.column_idxs_datetime = {
            idx for idx, param in enumerate(headers) if param in {"modified", "created"}
        }
        self.column_idxs_bytes = {
            idx for idx, param in enumerate(headers) if param == "filesize"
        }

    def construct_row(
        self,
        filepath: str,
    ) -> data_row_type:
        """
        Construct a data row for the specified file, based on the current parser settings and parameter selection.

        This method can be used to construct a new data row for a file that has been reloaded or updated, without needing to reconstruct the entire table. The constructed row will be based on the current parser settings and parameter selection, and will include the load status, file index number (alphabetical), filename, and the parameters specified in `param_selection` (or the default parameters determined by the parser class if `param_selection` is None).

        Parameters
        ----------
        filepath : str
            The path to the file for which to construct the data row.

        Returns
        -------
        data_row_type
            A tuple containing the load status, file index number (alphabetical), filename, and the parameters specified in `param_selection` for the specified file. The load status is excluded if no parser is set for the model.
        """
        return self._construct_data_rows(filepaths=[filepath], with_headers=False)[0]

    @property
    def param_selection(self) -> list[str] | None:
        """
        The parameters to be displayed in the table, if any.

        This property is used to establish the column headers after the load status and filename, as well as to collect parameters for table data.
        If none (as default), `ParserDataModel` instead uses the  parser class to
        determine the `_param_selection_default` attribute, and the consequential headers.

        Modifying this property will update the table data accordingly.

        Parameters
        ----------
        param_selection : list[str] | None
            The parameters to be set for display in the table, or None to use the default parameters determined by the parser class.

        Returns
        -------
        list[str] | None
            The parameters to be displayed in the table, or None if no specific parameters are set.
        """
        return self._param_selection

    @param_selection.setter
    def param_selection(self, param_selection: list[str] | None):
        if param_selection is not None:
            ps_set = set(param_selection)
            if len(ps_set) != len(param_selection):
                duplicates = set(
                    x for x in param_selection if param_selection.count(x) > 1
                )
                raise ValueError(
                    f"Parameter selection must not contain duplicates ({', '.join(duplicates)})."
                )

        # Update to the new selection
        self._param_selection = param_selection
        self.construct_table()
        return

    def make_default_selection(self):
        """
        Infer the default parameter selection for the table based on the current parser.

        This requires the implementation of `parser.SUMMARY_PARAMS`
        Preserves custom selections if possible, but adds default parameters of the new parser if they are not already included in the custom selection.

        Updates the `param_selection_default` attribute with the inferred default parameters.

        Returns
        -------
        set[str] | None
            The inferred parameters to be displayed in the table, or None if no specific parameters can be inferred.
        """
        parser_cls = self.parser
        if parser_cls is not None:
            self._param_selection_default = parser_cls.SUMMARY_PARAMS.copy()

            # If there is a custom selection, adjust the selection to include the default parameters.
            old_selection = self._param_selection
            if old_selection is not None:
                new_selection = old_selection
                for param in self._param_selection_default:
                    if param not in old_selection and param not in new_selection:
                        new_selection.append(param)
                self.param_selection = new_selection
            else:
                # Do not update the table data if there is no existing custom selection, to preserve the default selection until the user changes it.
                pass
        else:
            self._param_selection_default = None
        return

    @abstractmethod
    def layoutAboutToChange(
        self,
        # parents: list[QtCore.QPersistentModelIndex] = [],
        # hint: QtCore.QAbstractTableModel.LayoutChangeHint = QtCore.QAbstractTableModel.LayoutChangeHint.NoLayoutChangeHint
    ) -> None:
        """
        Abstract method to signal a `layoutAboutToBeChanged` event to the implementing view.
        """
        raise NotImplementedError(
            "The layoutAboutToChange method must be implemented by the subclass to signal a layoutAboutToChange event to the view."
        )

    @abstractmethod
    def layoutChange(
        self,
        # parents: list[QtCore.QPersistentModelIndex] = [],
        # hint: QtCore.QAbstractTableModel.LayoutChangeHint = QtCore.QAbstractTableModel.LayoutChangeHint.NoLayoutChangeHint
    ) -> None:
        """
        Abstract method to signal a `layoutChanged` event to the implementing view.
        """
        raise NotImplementedError(
            "The layoutChanged method must be implemented by the subclass to signal a layoutChanged event to the view."
        )


class ParserTableModel(ParserDataModel, QtCore.QAbstractTableModel):
    """
    Implements the QAbstractTableModel for a fileviewer.

    Implements headersData to display provided headers,
    and the adds three icons for load status (error, header, success).

    Parameters
    ----------
    path : str
        The path to the file to be displayed in the table.
    parser_cls : type[parserBase] | None
        The parser class to be used for loading the file, or None to clear the parser.
    progress_bar : QtWidgets.QProgressBar | None, optional
        A progress bar widget to be updated during file loading / parsing, by default None.
    parent : QtCore.QObject | None, optional
        The parent object for the model, by default None.
    """

    # Initialise graphics for loaded / unloaded files.
    _ICON_NONE: QtGui.QIcon
    """Icon for file load none status."""

    _ICON_ERROR: QtGui.QIcon
    """Icon for file load `error` status."""

    _ICON_HEADER_ONLY: QtGui.QIcon
    """Icon for file load `header_only` status."""

    _ICON_SUCCESS: QtGui.QIcon
    """Icon for file load `success` status."""

    def __init__(
        self,
        path: str,
        parser_cls: type[parserBase] | None,
        progress_bar: QtWidgets.QProgressBar | None = None,
        parent: QtCore.QObject | None = None,
    ):
        # Super initialisation.
        # super().__init__(parent=parent, path=path, parser_cls=parser_cls, progress_bar=progress_bar)
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        ParserDataModel.__init__(
            self, path=path, parser_cls=parser_cls, progress_bar=progress_bar
        )

        self.__class__._ICON_NONE = _safe_style_icon()
        self.__class__._ICON_ERROR = _safe_style_icon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogCancelButton
        )
        self.__class__._ICON_HEADER_ONLY = _safe_style_icon(
            QtWidgets.QStyle.StandardPixmap.SP_FileIcon
        )
        self.__class__._ICON_SUCCESS = _safe_style_icon(
            QtWidgets.QStyle.StandardPixmap.SP_DialogApplyButton
        )

    def construct_table(self) -> None:
        """
        Construct data and headers based on the current filelist and parser settings.

        Overrides the base method to signal layout changes to the view after constructing the table data.
        """
        self.beginResetModel()
        super().construct_table()
        self.endResetModel()

    def reload_file(self, filepath: str, header_only: bool = True) -> None:
        """
        Reload a specific file. Updates the table data, but does not trigger a view update.

        Parameters
        ----------
        filepath : str
            The path to the file to be reloaded.
        header_only : bool, optional
            Whether to load only the header of the file (True) or the full data (False), by default True.

        Raises
        ------
        ValueError
            If the file is not currently parsed and cached, or if the file is not a valid file or no parser is set, and therefore cannot be reloaded.
        """
        if filepath not in self.parsed:
            raise ValueError(
                f"File '{filepath}' is not currently parsed and cached, cannot reload."
            )

        # Get the index of the file in the current filelist.
        if os.path.isfile(filepath) and self.parser is not None:
            try:
                new_instance = self.parser(filepath, header_only=header_only)
                self.parsed[filepath] = (
                    self.loadStatus.HEADER_ONLY
                    if header_only
                    else self.loadStatus.FULL_DATA,
                    new_instance,
                )
                return
            except Exception:
                pass
        else:
            raise ValueError(
                f"File '{filepath}' is not a valid file or no parser is set, cannot reload."
            )
        # Keep the old instance and update the load status to error.
        self.parsed[filepath] = (self.loadStatus.ERROR, self.parsed[filepath][1])
        return

    def reload(
        self, index: QtCore.QModelIndex = QtCore.QModelIndex(), header_only: bool = True
    ) -> None:
        """
        Reload specific data in the model and signal the view to update accordingly.

        Overrides the base method to emit a `dataChanged` event for the entire model after reloading, to ensure that all changes are reflected in the view.

        Parameters
        ----------
        index : QtCore.QModelIndex, optional
            The index of the data to be reloaded, by default an invalid index which will trigger a full reload.
        header_only : bool, optional
            Whether to load only the header of the file (True) or the full data (False), by default True.
        """

        if index.isValid():
            # Remove the old index
            # self.beginRemoveRows(QtCore.QModelIndex(), index.row(), index.row())
            row = index.row()
            old_data = self._data[row]
            if self.parser is not None:
                filename = old_data[2]  # Get the filename from the old data row.
                assert isinstance(filename, str)
            else:
                filename = old_data[1]  # Get the filename from the old data row.
                assert isinstance(filename, str)
            filepath = os.path.join(
                self._path, filename
            )  # Get the filepath from the old data row.

            assert (
                os.path.basename(filepath) == old_data[2]
                if self.parser is not None
                else old_data[1]
            ), (
                f"The filename in the data row ({old_data[2] if self.parser is not None else old_data[1]}) does not match the filename from the filepath ({os.path.basename(filepath)}), cannot reload."
            )

            # Reload the file and update the data row with the new information.
            if self.parser is not None:
                # Get an updated parser instance in the cache.
                self.reload_file(filepath, header_only=header_only)
                # Use cache to update table
                # new_instance = self._parsed[filepath][1]

            # Update the data row with the new information.
            assert isinstance(self._data[row][0], self.loadStatus)
            self._data[row] = self.construct_row(filepath)  # type: ignore[assignment]

            # Signal the view that the data has changed for this row, to trigger an update.
            self.dataChanged.emit(
                self.index(row, 0), self.index(row, len(self._header) - 1)
            )

    def delete(self, index: QtCore.QModelIndex) -> None:
        """
        Delete specific data in the model and signal the view to update accordingly.

        Overrides the base method to emit a `dataChanged` event for the entire model after deleting, to ensure that all changes are reflected in the view.

        Parameters
        ----------
        index : QtCore.QModelIndex
            The index of the data to be deleted.
        """
        if index.isValid():
            row = index.row()
            self.beginRemoveRows(QtCore.QModelIndex(), row, row)
            # Remove the row
            data = self._data.pop(row)
            # Remove the file from the files list
            filename = data[2] if self.parser is not None else data[1]
            if filename in self._filelist:
                # Remove it
                self._filelist = tuple(f for f in self._filelist if f != filename)
            # Remove the row from cache if it exists in the cache.
            self.endRemoveRows()

    @override
    def headerData(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,  # This doesn't override correctly, because of type hint.
        /,
        role: int,
    ) -> Any:
        """
        Overridden method to provide header data for the table view.

        Parameters
        ----------
        section : int
            The section of the header to be displayed.
        orientation : QtCore.Qt.Orientation
            The orientation of the header (horizontal or vertical).
        role : int
            The role of the data to be displayed (e.g. display, decoration, etc.).

        Returns
        -------
        Any
            The header data for the specified section and role.
        """

        if self._header is None:
            return None

        match role:
            case (
                QtCore.Qt.ItemDataRole.DisplayRole | QtCore.Qt.ItemDataRole.ToolTipRole
            ):
                if orientation == QtCore.Qt.Orientation.Horizontal:
                    if len(self._header) > section:
                        head = self._header[section]
                        if isinstance(head, str):
                            return head.capitalize()
                        return head
                    else:
                        return None
                else:
                    # Add index numbers for vertical headers.
                    return str(section + 1)
            case _:
                return super().headerData(section, orientation, role)

    @override
    def data(
        self,
        index: QtCore.QModelIndex | QtCore.QPersistentModelIndex,
        role: int = QtCore.Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        """
        Overridden method to provide data for the table view.

        Implements icons for load status.

        Parameters
        ----------
        index : QtCore.QModelIndex | QtCore.QPersistentModelIndex
            The index of the data to be displayed.
        role : int, optional
            The role of the data to be displayed (e.g. display, decoration, etc.), by default QtCore.Qt.ItemDataRole.DisplayRole

        Returns
        -------
        Any
            The data for the specified index and role.

        Raises
        ------
        ValueError
            If the data for the specified index and role is not valid.
        """
        parser = self.parser
        load_status_idx = 0 if parser is not None else None
        col_idx = index.column()
        row_idx = index.row()
        # Check that first column is being indexed for "loaded" status, to display icons.
        if (
            self._header is not None
            and load_status_idx is not None
            and len(self._header) > 0
            and col_idx == load_status_idx  # Is the status column.
            and parser is not None
        ):
            if (
                role == QtCore.Qt.ItemDataRole.DecorationRole
                or role == QtCore.Qt.ItemDataRole.ToolTipRole
            ):
                # Icons
                dat = self._data[row_idx][col_idx]
                if isinstance(dat, int):
                    return dat

                dat = self.loadStatus(dat)  # Convert to enum.
                match dat:
                    case self.loadStatus.ERROR:
                        val = self._ICON_ERROR
                    case self.loadStatus.HEADER_ONLY:
                        val = self._ICON_HEADER_ONLY
                    case self.loadStatus.FULL_DATA:
                        val = self._ICON_SUCCESS
                    case self.loadStatus.NONE:
                        val = self._ICON_NONE
                    case _:
                        raise ValueError(f"Incorrect data match for {dat}")
                if role == QtCore.Qt.ItemDataRole.ToolTipRole:
                    filename = self._data[row_idx][2]
                    assert isinstance(filename, str)
                    filepath = os.path.join(
                        self._path, filename
                    )  # Get the filepath from the data row.
                    parser_instance = self.parsed[filepath][1]
                    parser_method = parser_instance.parser_fn_name
                    return f"{dat.name.lower().replace('_', ' ').capitalize()} - {parser_method}"
                return val
            else:
                return None

        # General data accessing
        match role:
            case (
                QtCore.Qt.ItemDataRole.DisplayRole | QtCore.Qt.ItemDataRole.ToolTipRole
            ):
                val = self._data[row_idx][col_idx]
                # Special formatting for bytes and datetime parameters, if the column index matches the corresponding sets.
                if col_idx in self.column_idxs_datetime and isinstance(
                    val, datetime.datetime
                ):
                    return val.strftime("%Y-%m-%d %H:%M:%S")
                elif col_idx in self.column_idxs_bytes and isinstance(
                    val, (int, float)
                ):
                    if role == QtCore.Qt.ItemDataRole.DisplayRole:
                        # Format bytes in a human-readable format.
                        for unit in ["B", "KB", "MB", "GB", "TB"]:
                            if val < 1024.0:
                                return f"{val:.2f} {unit}"
                            val /= 1024.0
                        return f"{val:.2f} PB"
                    else:
                        return f"{val} bytes"

                # Default behavior for display and tooltip roles.
                if role == QtCore.Qt.ItemDataRole.ToolTipRole and not isinstance(
                    val, str
                ):
                    return None
                return val

            case _:
                return None

    @override
    def rowCount(
        self,
        /,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        """
        The number of rows in the table is determined by the length of the outer list in the data.

        Parameters
        ----------
        parent : QtCore.QModelIndex | QtCore.QPersistentModelIndex, optional
            Parent is used for hierarchical models to check `isValid()` and return 0,
            not used in this implementation as flat model.

        Returns
        -------
        int
            The number of rows in the table.
        """
        try:
            # The length of the outer list.
            if hasattr(self, "_data") and hasattr(self._data, "__len__"):
                return len(self._data)
            return 0
        except Exception as e:
            print(f"Error in rowCount: {e}")
            return 0

    @override
    def columnCount(
        self,
        /,
        parent: QtCore.QModelIndex
        | QtCore.QPersistentModelIndex = QtCore.QModelIndex(),
    ) -> int:
        """
        The number of columns in the table is determined by the length of the inner lists in the data.

        Parameters
        ----------
        parent : QtCore.QModelIndex | QtCore.QPersistentModelIndex, optional
            Parent is used for hierarchical models to check `isValid()` and return 0,
            not used in this implementation as flat model.

        Returns
        -------
        int
            The number of columns in the table.
        """
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

    @override
    def layoutAboutToChange(
        self,
        # parents: list[QtCore.QPersistentModelIndex] = [],
        # hint: QtCore.QAbstractTableModel.LayoutChangeHint = QtCore.QAbstractTableModel
        # .LayoutChangeHint.NoLayoutChangeHint
    ) -> None:
        """
        Implementation of the abstract method to signal a `layoutAboutToBeChanged` event.

        """
        self.layoutAboutToBeChanged.emit()
        return

    @override
    def layoutChange(
        self,
        # parents: list[QtCore.QPersistentModelIndex] = [],
        # hint: QtCore.QAbstractTableModel.LayoutChangeHint = QtCore.QAbstractTableModel.LayoutChangeHint.NoLayoutChangeHint
    ) -> None:
        """
        Implementation of the abstract method to signal a dataChanged event to the view.
        """
        self.layoutChanged.emit()
        return


parser_menu_function = (
    typing.Callable[[parserBase], typing.Any]
    | typing.Callable[[parserBase, bool], typing.Any]
)
"""
A type definition for functions that can be added to the context menu of the parser table view.

The first function parameter should be the the parser instance, and a second optional parameter is the boolean of a toggle action if applicable."""


class ParserTableView(QtWidgets.QTableView):
    """
    A table widget that displays pyNexafs related parser files.

    The table can display either parser header rows or parser full-data rows,
    depending on how entries are loaded.

    Parameters
    ----------
    parent : QtWidgets.QWidget | None, optional
        The parent widget for the view, by default None.
    """

    @override
    def __init__(
        self,
        path: str = "",
        parser_cls: type[parserBase] | None = None,
        progress_bar: QtWidgets.QProgressBar | None = None,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent)

        # Instance Widgets
        self.files_model = ParserTableModel(
            path=path, parser_cls=parser_cls, progress_bar=progress_bar
        )

        # TODO: Add a custom sort for the load status column.
        self.proxy_model = QtCore.QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.files_model)

        # Instance Attributes
        self.verticalHeader().hide()  # remove default row numbers
        self.setModel(self.proxy_model)
        # self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        # self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSortingEnabled(True)  # enabled by proxymodel.
        self.proxy_model.setFilterRegularExpression(QtCore.QRegularExpression(""))
        self.proxy_model.setFilterKeyColumn(
            -1
        )  # use all columns for filtering instead of just first.
        self.proxy_model.sort(0, QtCore.Qt.SortOrder.AscendingOrder)
        # self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for i in range(self.files_model.columnCount()):
            self.resizeColumnToContents(i)
        self.horizontalHeader().setMinimumSectionSize(10)

        # Progress bar
        self._progress_bar = progress_bar

        # Menu action list
        self._menu_functions: list[parser_menu_function] = []
        self._menu_actions: list[QtGui.QAction] = []

    def reload_file(self, index: QtCore.QModelIndex):
        """
        Reload a specific file based on the provided index.

        Parameters
        ----------
        index : QtCore.QModelIndex
            The index of the file to be reloaded.
        """
        if index.isValid():
            source_index = self.proxy_model.mapToSource(index)
            self.files_model.reload(source_index)

    @property
    def menu_functions(self) -> list[parser_menu_function]:
        """
        A list of custom functions added after the delete/reload actions.

        Parameters
        ----------
        actions : list[QtGui.QAction]
            A list of actions to be included in the context menu,
            which will be added after the default delete and reload actions.

        Returns
        -------
        list[QtGui.QAction]
            A list of actions to be included in the context menu.
        """
        return self._menu_functions

    @menu_functions.setter
    def menu_functions(self, functions: list[parser_menu_function]):
        if not isinstance(functions, list) or not all(
            callable(func) for func in functions
        ):
            raise ValueError(
                "Menu functions must be provided as a list of callable instances."
            )
        self._menu_functions = functions

        # Create corresponding actions for the functions, which will be added to the context menu.
        self._menu_actions = []
        for func in functions:
            action = QtGui.QAction(
                text=func.__name__.replace("_", " ").capitalize(), parent=self
            )
            # Get the number of parameters the function accepts, to determine how to connect it.
            sig = inspect.signature(func)
            params = sig.parameters

            if len(params) == 1:
                action.triggered.connect(
                    lambda checked, f=func: f(self.files_model.parser)
                )
            elif len(params) == 2:
                action.triggered.connect(
                    lambda checked, f=func: f(self.files_model.parser, checked)
                )
            else:
                raise ValueError(
                    f"Menu function '{func.__name__}' must accept at least one parameter "
                    f"(the parser instance), and at most two parameters (the parser instance "
                    f"and an optional boolean for toggle actions), but it accepts {len(params)} parameters."
                )
            self._menu_actions.append(action)

    @menu_functions.deleter
    def menu_functions(self):
        self._menu_functions = []

    @override
    def contextMenuEvent(self, arg__1: QtGui.QContextMenuEvent) -> None:
        context = QtWidgets.QMenu(self)
        # Get the filename:
        index = self.indexAt(arg__1.pos())
        if index.isValid():
            source_index = self.proxy_model.mapToSource(index)
            filename = (
                self.files_model._data[source_index.row()][
                    2
                ]  # Filename is the third column (after load status and index).
                if self.files_model.parser is not None
                else self.files_model._data[source_index.row()][
                    1
                ]  # Filename is the second column
            )
            assert isinstance(filename, str)

            # Setup the actions:
            label_filename = QtWidgets.QLabel(text=filename)
            label_filename.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            label_filename.setStyleSheet("font-weight: bold; padding: 5px;")
            action_filename = QtWidgets.QWidgetAction(context)
            action_filename.setDefaultWidget(label_filename)
            action_delete = QtGui.QAction(
                "Delete",
                parent=self,
            )
            action_delete.triggered.connect(
                lambda: self.files_model.delete(source_index)
            )
            action_reload = QtGui.QAction(
                parent=self,
                text="Reload (header only)",
            )
            action_reload.triggered.connect(
                lambda: self.files_model.reload(source_index, header_only=True)
            )
            action_reload_full = QtGui.QAction(
                parent=self,
                text="Reload (with data)",
            )
            action_reload_full.triggered.connect(
                lambda: self.files_model.reload(source_index, header_only=False)
            )

            # Create the context menu and add the actions:
            context.addAction(action_filename)
            context.addSeparator()
            context.addAction(action_reload)
            context.addAction(action_reload_full)
            context.addAction(action_delete)

            # Add custom actions
            custom_actions = self._menu_actions
            for action in custom_actions:
                context.addAction(action)

        # Show the context menu at the cursor position.
        context.exec(arg__1.globalPos())

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
    def filters(self, filters: tuple[str | None, Sequence[str] | None]):
        """
        Set the filters for the directory viewer.

        Parameters
        ----------
        filters : tuple[str | None, Sequence[str] | None]
            A tuple containing the filename filter and the filetype sequence filter.

        Raises
        ------
        AttributeError
            If the filters are not provided as a tuple of (str_filter, filetype_filters), or if the filetype filter is not a sequence of strings, or None.
        """
        if not isinstance(filters, tuple) or len(filters) != 2:
            raise AttributeError(
                f"Filters must be a tuple of (str_filter, filetype_filters), got {filters}."
            )
        str_filter, filetype_filters = filters

        # Correct type for empty strings.
        if str_filter == "":
            str_filter = None

        # Check if str filter is changed:
        old_str_filter = self._str_filter
        if str_filter != old_str_filter:
            self._str_filter = str_filter
            # Update proxy model
            QRegExp = (
                QtCore.QRegularExpression(
                    str_filter,
                    QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption,
                )
                if str_filter is not None
                else QtCore.QRegularExpression("")
            )
            self.proxy_model.setFilterRegularExpression(QRegExp)
            # self.proxy_model.setFilterFixedString(str_filter)

        # Pass filetype filter onto the ParserDataModel, which will use it to determine which files to parse and include in the table.

    @filters.deleter
    def filters(self):
        """
        Remove the filters for the directory viewer.
        """
        self._str_filter, self._filetype_filters = None, None
        self.update_table()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    from pyNexafs.parsers import MEX2_NEXAFS

    # Test we can load each file within the directory
    path = r"D:\Github\pynexafs\tests\test_data\au\MEX2\2024-03"
    filelist = os.listdir(path)
    for filename in filelist:
        filepath = os.path.join(path, filename)
        try:
            parser_instance = MEX2_NEXAFS(filepath, header_only=True)
            print(f"Successfully parsed {filepath}")
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    view = ParserTableView(
        path=path,
        # parser_cls=None,
        parser_cls=MEX2_NEXAFS,
    )

    view.show()
    app.exec()
