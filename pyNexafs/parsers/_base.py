import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abc
import sys, io, os
import types
import warnings
from io import TextIOWrapper
from typing import Any, TypeVar, Type, Self
from numpy.typing import NDArray
from pyNexafs.nexafs.scan import scan_base
import datetime


class parser_meta(abc.ABCMeta):
    """
    Metaclass to implement class properties for parser classes.

    Defines getter/setter/deleter methods for class properties such as
    `RELABELS`, `COLUMN_ASSIGNMENTS`, and `ALLOWED_EXTENSIONS`.
    """

    _COLUMN_ASSIGNMENTS = {}
    _ALLOWED_EXTENSIONS = []
    _SUMMARY_PARAM_RAW_NAMES = []
    _RELABELS = {}
    _relabel = False

    def __new__(
        __mcls: type[Self],
        __name: str,
        __bases: tuple[type, ...],
        __namespace: dict[str, Any],
        **kwargs: Any,
    ) -> Self:

        # Perform checks on parsers that implement parser_base.
        if __name != "parser_base":
            # If class does not define the important parameters, then set to empty list.
            if "SUMMARY_PARAM_RAW_NAMES" not in __namespace:
                __namespace["SUMMARY_PARAM_RAW_NAMES"] = []
            if "RELABELS" not in __namespace:
                __namespace["RELABELS"] = {}

            # Raise error if necessary variables are not defined.
            for name in ["ALLOWED_EXTENSIONS", "COLUMN_ASSIGNMENTS"]:
                if name not in __namespace:
                    raise ValueError(f"Class {__name} does not define {name}.")

            # Rename attributes, avoid overriding property.
            for name in [
                "ALLOWED_EXTENSIONS",
                "COLUMN_ASSIGNMENTS",
                "SUMMARY_PARAM_RAW_NAMES",
                "RELABELS",
            ]:
                __namespace[f"_{name}"] = __namespace[
                    name
                ]  # Adjust assignments to an internal variable i.e. _ALLOWED_EXTENSIONS
                del __namespace[name]  # Remove old assignment

            # Validate column assignments
            __namespace["_COLUMN_ASSIGNMENTS"] = parser_meta.__validate_assignments(
                __namespace["_COLUMN_ASSIGNMENTS"]
            )

        return super().__new__(__mcls, __name, __bases, __namespace, **kwargs)

    def __init__(name, bases, dict, kwds):
        super().__init__(name, bases, dict, **kwds)

        # Gather internal parser methods at class creation for use in file loading.
        name.parse_functions = []
        for fn_name in dir(name):
            if fn_name.startswith("parse_"):
                fn = getattr(name, fn_name)
                if callable(fn):
                    arg_names = fn.__code__.co_varnames[: fn.__code__.co_argcount]
                    # Check the parameters of each function match requirements.
                    if type(fn) == types.FunctionType:  # Static methods
                        if arg_names[0] != "file":
                            raise TypeError(
                                f"First argument of static parser method must be 'file'. Is {arg_names[0]}."
                            )
                        if len(arg_names) == 2 and arg_names[1] != "header_only":
                            raise TypeError(
                                f"Second (optional) argument of static parser method must be 'header_only'. Is {arg_names[2]}."
                            )
                        name.parse_functions.append(fn)
                    elif type(fn) == types.MethodType:  # Class methods
                        if len(arg_names) < 2 or len(arg_names) > 3:
                            raise TypeError(
                                f"Parser method must only have 2-3 arguments: 'cls', 'file' and (optional) 'header_only'. Has {arg_names}"
                            )
                        if arg_names[0] != "cls":
                            raise TypeError(
                                f"First argument of parser method must be 'cls', i.e. the class. It is instead {arg_names[0]}."
                            )
                        if arg_names[1] != "file":
                            raise TypeError(
                                f"Second argument of parser method must be 'file'. Is {arg_names[1]}."
                            )
                        if len(arg_names) == 3 and arg_names[2] != "header_only":
                            raise TypeError(
                                f"Third (optional) argument of parser method must be 'header_only'. Is {arg_names[2]}."
                            )
                        name.parse_functions.append(fn)
        return

    @property
    def ALLOWED_EXTENSIONS(cls) -> list[str]:
        """
        Returns the allowed extensions for the parser.

        'parser_base.file_parser' will check validity of file extensions.
        Inheriting parser instance will need to deal with each filetype
        in the file_parser method.

        Returns
        -------
        list[str]
            List of allowed extensions. Read only.


        Examples
        --------
        synchrotron_parser.ALLOWED_EXTENSIONS = [".dat", ".txt", ".csv"]

        """
        return cls._ALLOWED_EXTENSIONS

    @staticmethod
    def __validate_assignments(
        assignments: dict[str, str | list[str] | None]
    ) -> dict[str, str | list[str] | None]:
        """
        Validation method for column assignments for a parser.

        Validates by checking the custom string label entries for assigning data columns to a scan object.
        Requires 'x', 'y' keys, and can optionally contain 'y_errs', and 'x_errs' keys which are defaulted to None.

        Parameters
        ----------
        assignments : dict[str, str | list[str] | None]
            Dictionary of column assignments.

        Returns
        -------
        dict[str, str | list[str] | None]

        Raises
        ------
        ValueError
            'x' assignment not found in assignments.
        ValueError
            'x' assignment is not a string.
        ValueError
            'y' assignment is not a list or string.
        ValueError
            'y' assignment is a list, but subelement is not a string.
        ValueError
            'y_errs' assignment is not a list, string, or None.
        ValueError
            'y_errs' assignment is a list, but subelement is not a string.
        ValueError
            'y_errs' assignment does not match length of 'y' assignment.
        ValueError
            'x_errs' assignment is not a string or None.
        """
        # X
        if not "x" in assignments:
            raise ValueError("'x' assignment not found in assignments.")
        if not isinstance(assignments["x"], str):
            raise ValueError(f"'x' assignment {assignments['x']} is not a string.")
        # Y
        if not "y" in assignments:
            raise ValueError("'y' assignment not found in assignments.")
        if not isinstance(assignments["y"], (list, str)):
            raise ValueError(
                f"'y' assignment {assignments['y']} is not a list or string."
            )
        if isinstance(assignments["y"], list):
            for y in assignments["y"]:
                if not isinstance(y, str):
                    raise ValueError(f"'y' list element {y} is not a string.")
        # Yerrs
        if "y_errs" not in assignments:
            assignments["y_errs"] = None  # set a default value.
        if not isinstance(assignments["y_errs"], (list, str, type(None))):
            raise ValueError(
                f"'y_errs' assignment {assignments['y_errs']} is not a list, string, or None."
            )
        if isinstance(assignments["y_errs"], list):
            if len(assignments["y_errs"]) != len(assignments["y"]):
                raise ValueError(
                    f"'y_errs' list assignment {assignments['y_errs']}\n does not match length of 'y' list assignment {assignments['y']}."
                )
            for y in assignments["y_errs"]:
                if not isinstance(y, str):
                    raise ValueError(f"'y_errs' list assignment {y} is not a string.")
        # Xerrs
        if "x_errs" not in assignments:
            assignments["x_errs"] = None
        if not isinstance(assignments["x_errs"], (str, type(None))):
            raise ValueError(
                f"'x_errs' assignment {assignments['x_errs']} is not a string or None."
            )
        return assignments

    @property
    def COLUMN_ASSIGNMENTS(cls) -> dict[str, str | list[str] | None]:
        """
        Assignments of scan input variables to column names.

        'parser_base.to_scan' will use construct the scan parameters.
        Assignments can be a single column name, or a list of column names.
        y_errs and x_errs can be None if not present in the data.
        Use '|' to separate equivalent column names. Scan_base will use the first column name found.

        Returns
        -------
        dict[str, str | list[str] | None]
            A dictionary of column assignments.
            Requires 'x', 'y', 'y_errs', and 'x_errs' keys.

        Examples
        --------
        synchrotron_parser.COLUMN_ASSIGNMENTS ={
            "x": "Data_Column_1_Label",
            "y": [
                "Data_Column_2_Label",
                "Data_Column_3_Label|Alternative_Column_3_Label",
                "Data_Column_4_Label",
            ],  # or "Data_Column_2_Label"
            "y_errs": [
                "Data_Column_5_Label",
                "Data_Column_6_Label",
                "Data_Column_7_Label",
            ],  # or "Data_Column_5_Label" or None
            "x_errs": None,  # or "Data_Column_8_Label"
        }
        """
        return cls._COLUMN_ASSIGNMENTS

    @COLUMN_ASSIGNMENTS.setter
    def COLUMN_ASSIGNMENTS(cls, assignments: dict[str, str | list[str] | None]) -> None:
        """
        Setter for the COLUMN_ASSIGNMENTS property.

        Parameters
        ----------
        assignments : dict[str, str | list[str] | None]
            Dictionary of column assignments.
        """
        cls._COLUMN_ASSIGNMENTS = cls.__validate_assignments(assignments)

    @property
    def SUMMARY_PARAM_RAW_NAMES(cls) -> list[str]:
        """
        A list of important parameters, for displaying file summary information.

        Used by GUI methods for displaying summary file information.

        Returns
        -------
        list[str]
            List of important parameter strings, matching keys in 'parser_base.params'.
        """
        return cls._SUMMARY_PARAM_RAW_NAMES

    @SUMMARY_PARAM_RAW_NAMES.setter
    def SUMMARY_PARAM_RAW_NAMES(cls, summary_params: list[str]) -> None:
        """
        Setter for the SUMMARY_PARAM_RAW_NAMES property.

        Parameters
        ----------
        summary_params : list[str]
            List of important parameter strings, matching keys in 'parser_base.params'.
        """
        cls._SUMMARY_PARAM_RAW_NAMES = summary_params

    @property
    def summary_param_names(cls) -> list[str]:
        """
        Returns a list of important parameter names of the data file.

        Sources from cls.SUMMARY_PARAM_RAW_NAMES.
        If names are defined in cls.RELABELS, then the relabelled names are returned.

        Returns
        -------
        list[str]
            List of important parameter names.
        """
        return [
            cls.RELABELS[name] if (name in cls.RELABELS and cls.relabel) else name
            for name in cls.SUMMARY_PARAM_RAW_NAMES
        ]

    @property
    def RELABELS(cls) -> dict[str, str]:
        """
        Renames labels to more useful names (optional property).

        'parser_base.to_scan' will use this to relabel the scan params or column labels.
        Assignments should be dict entries in the form of 'old_label' : 'new_label'.

        Returns
        -------
        dict[str, str]
            Dictionary of old labels to new labels, matching names in 'parser_base.labels' or 'parser_base.params'.
        """
        return cls._RELABELS

    @RELABELS.setter
    def RELABELS(cls, relabels: dict[str, str]) -> None:
        """
        Setter for the RELABELS property.

        Parameters
        ----------
        relabels : dict[str,str]
            Dictionary of old labels to new labels, matching names in 'parser_base.labels' or 'parser_base.params'.
        """
        cls._RELABELS = relabels

    @property
    def relabel(cls) -> bool:
        """
        Whether the parser object returns relabelled defined in cls.RELABELS.

        If True, by the provided params-names and column-headers are relabelled.
        If False, then the original params-names and column-headers are used by the class.

        Returns
        -------
        bool
            True if the parser object returns relabelled params/column-headers.
        """
        return cls._relabel

    @relabel.setter
    def relabel(cls, value) -> None:
        """
        Setter for the relabel property.

        Parameters
        ----------
        value : bool
            True if the parser object returns relabelled params/column-headers.
        """
        cls._relabel = value


class parser_base(metaclass=parser_meta):
    """
    General class that parses raw files to acquire data/meta information.

    Requires implementation of `file_parser` method, as well as the property
    methods `ALLOWED_EXTENSIONS`, `COLUMN_ASSIGNMENTS` and (optional) `RELABELS`.
    These properties modify the behaviour of the parser object.

    Parameters
    ----------
    filepath : str | None, optional
        The filepath of the data file, by default None
    load_head_only : bool, optional
        If True, then only the header of the file loaded, by default False
    relabel : bool, optional
        If True, then column and parameter labels are returned as more useful
        names defined in 'parser_base.RELABELS', by default False

    Attributes
    ----------
    filepath : str | None
        The filepath of the data file.
    data : NDArray | None
        The data array of the file.
    units : list[str] | None
        The units of the data columns.

    Properties
    ----------
    labels : list[str] | None
        The labels of the data columns. Affected by relabel.
    params : dict[str, Any]
        Additional parameters of the data file. Affected by relabel.
    ALLOWED_EXTENSIONS : list[str]
        Allowable extensions for the parser.
    COLUMN_ASSIGNMENTS : dict[str, str | list[str] | None]
        Assignments of scan input variables to column names.
    SUMMARY_PARAM_RAW_NAMES : list[str]
        A list of important parameters, for displaying file summary information.
    RELABELS : dict[str, str]
        Renames labels to more useful names (optional property) when calling to_scan().
    summary_params : dict[str, Any]
        Returns a dictionary of important parameters of the data file.

    """

    # Set attributes to class properties. Necessary for calling class properties on the instance.
    ALLOWED_EXTENSIONS = parser_meta.ALLOWED_EXTENSIONS
    COLUMN_ASSIGNMENTS = parser_meta.COLUMN_ASSIGNMENTS
    SUMMARY_PARAM_RAW_NAMES = parser_meta.SUMMARY_PARAM_RAW_NAMES
    RELABELS = parser_meta.RELABELS
    # summary_param_names_with_units = parser_meta.summary_param_names_with_units
    summary_param_names = parser_meta.summary_param_names

    def __init__(
        self,
        filepath: str,
        load_head_only: bool = False,
        relabel: bool | None = None,
    ) -> None:

        # parser_meta super
        super().__init__()

        # Initialise variables
        self._filepath = filepath  # filepath of the data file
        self.data = None
        self._labels = []
        self.units = []
        self.params = {}

        if filepath is None:
            return
        elif type(filepath) is str:
            self.load(header_only=load_head_only)  # Load data
        else:
            raise TypeError(f"Filepath is {type(filepath)}, not str.")

        # Copy class value at initialisation if None
        self._relabel = relabel

    @property
    def relabel(self) -> bool:
        """
        Whether the parser object returns relabelled defined in cls.RELABELS.

        If True, by the provided params-names and column-headers are relabelled.
        If False, then the original params-names and column-headers are used by the class.

        Returns
        -------
        bool
            True if the parser object returns relabelled params/column-headers.
        """
        if self._relabel is None:
            return type(self)._relabel
        else:
            return self._relabel

    @relabel.setter
    def relabel(self, value: bool) -> None:
        """
        Setter for the relabel property.

        Parameters
        ----------
        value : bool
            True if the parser object returns relabelled params/column-headers.
        """
        self._relabel = value

    @relabel.deleter
    def relabel(self) -> None:
        """
        Deleter for the relabel property. Resets to class default behaviour.
        """
        self._relabel = None

    @property
    def filepath(self) -> str:
        """
        Returns the filepath of the data file.

        There is no setter for the filepath attribute, a new instance needs to be created
        which also allows the header_only boolean option.

        Returns
        -------
        str
            Filepath of the data file.
        """
        return self._filepath

    @property
    def filename(self) -> str:
        return os.path.basename(self._filepath)

    @property
    def labels(self) -> list[str]:
        """
        Returns the labels of the data columns.

        If relabel is True, then the labels are returned as more useful names defined in 'parser_base.RELABELS'.

        Returns
        -------
        list[str]
            Labels of the data columns.
        """
        if not self.relabel:
            return self._labels
        else:
            # Collect labels if they exist.
            relabels = []
            for label in self._labels:
                relabels.append(
                    self.RELABELS[label] if label in self.RELABELS else label
                )
            return relabels

    def to_scan(self, load_all_columns: bool = False) -> Type[scan_base] | scan_base:
        """
        Converts the parser object to a scan_base object.

        Parameters
        ----------
        load_all_columns : bool, optional
            If True, then columns unreferenced by COLUMN_ASSIGNMENTS are also
            loaded into the y attribute of the scan object. By default False.

        Returns
        -------
        type[scan_base]
            Returns a scan_base object.
        """
        if self.data is None:
            raise ValueError(
                "No data loaded into the parser object, only header information. Use parser.load() to load data."
            )
        return scan_base(parser=self, load_all_columns=load_all_columns)

    @classmethod
    def file_parser(
        cls, file: TextIOWrapper, header_only: bool = False
    ) -> tuple[NDArray | None, list[str], list[str], dict[str, Any]]:
        """
        Class method that tries to call cls.parse_functions methods.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWapper of the datafile (i.e. file=open('file.csv', 'r'))
        header_only : bool, optional
            If True, then only the header of the file is read and NDArray is returned as None, by default False

        Returns
        -------
        tuple[NDArray | None,
            list[str] | None,
            list[str] | None,
            dict]
            A tuple of the data (NDArray), labels (list), units (list) and parameters (dict) of the datafile.
        """

        # Check if allowed extension
        if not file.name.endswith(tuple(cls.ALLOWED_EXTENSIONS)):
            raise ValueError(
                f"File {file.name} is not a valid file type for {cls.__name__}."
            )

        if len(cls.parse_functions) > 0:
            # Check if any parse functions match the file type.
            for parse_fn in cls.parse_functions:
                # Attempt to use parse functions that contain a string that matches the extension
                if file.name.split(".")[-1] in parse_fn.__name__:
                    try:
                        arg_names = parse_fn.__code__.co_varnames[
                            : parse_fn.__code__.co_argcount
                        ]
                        if type(parse_fn) == types.FunctionType:  # staticmethod
                            return (
                                parse_fn(file, header_only)
                                if "header_only" in arg_names
                                else parse_fn(file)
                            )
                        else:  # classmethod
                            # type(parse_fn == types.MethodType)
                            # cls is the first argument of the method, already incorporated into the function call.
                            return (
                                parse_fn(file, header_only)
                                if "header_only" in arg_names
                                else parse_fn(file)
                            )
                    except Exception as e:
                        # Method failed, continue to next method.
                        warnings.warn(
                            f"Attempted method '{parse_fn.__name__}' failed to load '{file.name}' from '{cls.__name__}'. {type(e).__name__}: '"
                            + str(e)
                            + "'.",
                            UserWarning,
                        )
                        continue

            # If no parse functions successfully import the file type,
            raise ImportError(
                f"No parser method in {cls.__name__} succeeded on {file.name}."
            )
        # If no parse functions match the file type, raise an error.
        raise ImportError(f"No parser method in {cls.__name__} found for {file.name}.")

    def load(
        self, file: str | TextIOWrapper | None = None, header_only: bool = False
    ) -> None:
        """
        Loads data from the specified file, and attaches it to the object.

        Additionally rewrites filepath attribute if a new file is loaded.
        Will also save additional "created" and "modified" entries in param list
        using `os` library if not already generated by parser methods.

        Parameters
        ----------
        file : str | TextIOWrapper | None, optional
            File information can be passed as a string or a TextIOWrapper object.
            If None, then the object filepath attribute is used to load the data.
            If filepath is also None, then a ValueError is raised.
        header_only : bool, optional
            If True, then only the header of the file is read, by default False
            Consequently loads labels, units and params, but sets the data as None.

        Raises
        ------
        ValueError
            Raised if no file is provided and the object filepath attribute is None.
        """
        # Load object filepath
        load_filepath = self._filepath  # Might be None

        try:
            # Check if file parameter is provided:
            if type(file) is TextIOWrapper:
                load_filepath = file.name
                # File already loaded
                self.data, self._labels, self.params = self.file_parser(
                    file, header_only=header_only
                )
                # If a file is provided override filepath.
                self._filepath = file.name
                return
            elif type(file) is str:
                # Update filepath
                load_filepath = file

            # Try to load filepath
            if load_filepath is None:
                raise ValueError("No file/filepath provided to load data.")
            else:
                with open(load_filepath, "r") as load_file:
                    data, labels, units, params = self.file_parser(
                        load_file, header_only=header_only
                    )
            if data is not None or self.data is not None:
                # Use existing data to check consistency if only loading header.
                if data is None and self.data is not None:
                    data = self.data
                # Pull column length of data to compare to units and labels.
                col_len = data.shape[1]
                if labels is not None and len(labels) != col_len:
                    raise ValueError(
                        f"Labels length {len(labels)} does not match data columns {col_len}."
                    )
                if units is not None and len(units) != col_len:
                    raise ValueError(
                        f"Units length {len(units)} does not match data columns {col_len}."
                    )
        except AssertionError:
            # Assume file cannot be loaded using the parser methods. Raise loading error.
            raise ImportError(
                f"{load_filepath} encountered an assertion error while loading using the parse methods of {type(self)}."
            )

        # Add modified and created parameter entries from system OS, if not existing in params already.
        # If aleady provided, then convert to datetime object using `convert_to_datetime` method below.
        if "created" not in params or params["created"] is None:
            params["created"] = datetime.datetime.fromtimestamp(
                os.path.getctime(load_filepath)
            )
        if "modified" not in params:
            params["modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(load_filepath)
            )

        # Assign data, labels, units, and params to object.
        self.data, self._labels, self.units, self.params = data, labels, units, params

        # Update filepath after data load if successful!
        self._filepath = load_filepath
        return

    @staticmethod
    def convert_to_datetime(time_input: Any) -> datetime.datetime:
        if isinstance(time_input, datetime.datetime):
            return time_input
        elif isinstance(time_input, float):
            return datetime.datetime.fromtimestamp(time_input)
        elif isinstance(time_input, int):
            return datetime.datetime.fromordinal(time_input)
        elif isinstance(time_input, str):
            # First try ISO 8601 time format.
            try:
                return datetime.datetime.fromisoformat(time_input)
            except ValueError:
                pass
            # Try to convert to float or int
            try:
                return parser_base.convert_to_datetime(float(time_input))
            except ValueError:
                pass
            try:
                return parser_base.convert_to_datetime(int(time_input))
            except ValueError:
                pass
        # If none of these work, raise an error.
        raise ValueError(
            f"Time input {time_input} could not be converted to a datetime object."
        )

    @property
    def ctime(self) -> datetime.datetime:
        """
        Returns the creation time of the file as a datetime object.
        """
        return self.convert_to_datetime(self.params["created"])

    @property
    def mtime(self) -> datetime.datetime:
        """
        Returns the modification time of the file as a datetime object.
        """
        return self.convert_to_datetime(self.params["modified"])

    def copy(self, clone: type[Self] | None = None) -> type[Self]:
        """
        Generates a copy of the parser object.

        Parameters
        ----------
        clone : type[PARSER] | None, optional
            If a clone instance is provide, then properties are copied to the clone.
            If None, then a new copy is made from the current object.

        Returns
        -------
        type[PARSER]
            A copy of the parser object.
        """
        # Use the provided clone, otherwise create newobj.
        newobj = type(self)(None) if clone is not None else clone

        # Perform deep copy of data, labels, and params.
        newobj.filepath = self._filepath  # str copy
        newobj.data = self.data.copy()  # numpy copy
        newobj.labels = self._labels.copy()  # str list copy
        newobj.units = self.units.copy()  # str list copy
        for key in self.params:  # dict key str - value Any copy
            value = self.params[key]
            newobj.params[key] = (
                value if isinstance(value, (int, str, float, bool)) else value.copy()
            )
        return newobj

    @property
    def filesize(self) -> int:
        """Returns the size of the file in bytes."""

        if self._filepath is None:
            raise ValueError("No file loaded.")
        return os.path.getsize(self._filepath)

    @property
    def memorysize(self) -> int:
        """Returns the size of the object (and it's attributes) in memory bytes."""
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.data)
            + sys.getsizeof(self._labels)
            + sys.getsizeof(self.params)
        )

    def to_DataFrame(self) -> pd.DataFrame:
        """
        Converts the data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame of the data and labels.
        """
        return pd.DataFrame(
            self.data, columns=self._labels if len(self._labels) > 0 else None
        )

    def search_label_index(self, search: str, search_relabels=True) -> int:
        """
        Returns the index of the label in the labels list.

        Searches for raw labels first, then checks and searches for any RELABELS match.

        Parameters
        ----------
        search : str
            String label to search for in the labels list.
            Can also accommodate the '|' character to search for multiple labels, in priority of the left-right order.

        search_relabels : bool, optional
            To additionally search check the RELABELS dictionary for the label and subsequently search. By default True.

        Returns
        -------
        int
            Index of the label in the labels list.

        Raises
        ------
        ValueError
            Raised if the label is not found in the labels list.
        """
        queries = search.split("|")
        for query in queries:
            # Check default queries
            try:
                return self._labels.index(query)  # throws value error if not found.
            except ValueError as e:
                pass

            # Also check relabel if relabel is True.
            if self.relabel and search_relabels:
                # Search keys
                if query in self.RELABELS:
                    try:
                        return self._labels.index(
                            self.RELABELS[query]
                        )  # throws value error if not found
                    except ValueError as e:
                        pass

                # Search values
                if query in self.RELABELS.values():
                    # Check for multiple matching values (opposed to unique keys).
                    warn = (
                        True if list(self.RELABELS.values()).count(query) > 1 else False
                    )
                    #
                    i = list(self.RELABELS.values()).index(query)
                    if warn:
                        warnings.warn(
                            f"Multiple labels matched {query} in {type(self)}.RELABELS. Using key '{self.RELABELS.keys()[i]}' as the label."
                        )
                    return i

        raise ValueError(f"Label '{search}' not found in labels {self._labels}.")

    @property
    def summary_params(self) -> dict[str, Any]:
        """
        Returns a dictionary of important parameters of the data file.

        Returns
        -------
        dict
            Dictionary of important parameters.
        """
        return {key: self.params[key] for key in self.SUMMARY_PARAM_RAW_NAMES}

    @property
    def summary_param_names(self) -> list[str]:
        """
        Returns a list of important parameter names of the data file.

        Sources from cls.SUMMARY_PARAM_RAW_NAMES.
        If names are defined in cls.RELABELS, then the relabelled names are returned.

        Returns
        -------
        list[str]
            List of important parameter names.
        """
        return [
            self.RELABELS[name] if (name in self.RELABELS and self.relabel) else name
            for name in self.SUMMARY_PARAM_RAW_NAMES
        ]

    @property
    def summary_param_values(self) -> list[Any]:
        """
        Returns a list of important parameter values of the data file.

        Returns
        -------
        list
            List of important parameter values.
        """
        return [
            self.params[key] if key in self.params else None
            for key in self.SUMMARY_PARAM_RAW_NAMES
        ]

    @property
    def summary_param_names_with_units(self) -> list[str]:
        """
        Returns a list of important parameter names with units.

        Requires a loaded dataset to return the units of the parameters.
        Not a pre-defined class method.

        Returns
        -------
        list
            List of important parameter names with units.
        """
        raise NotImplementedError(
            f"Method 'summary_param_names_with_units' not implemented for {type(self)}."
        )
        return self.SUMMARY_PARAM_RAW_NAMES
