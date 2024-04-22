"""
Pynexafs.nexafs.scan: A set of classes for handling synchrotron scan data.

This module defines validated scan objects that can perform intelligent operations
on synchrotron data. These objects are constructed from a `parser` object that contains
the raw data and metadata.

# See Also
# --------
pynexafs.parsers.parser_base : Base class for synchrotron data parsers.
"""

from __future__ import annotations  # For type hinting within class definitions
import numpy.typing as npt
import numpy as np
import matplotlib as mpl
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import warnings
from typing import Type, TypeVar, TYPE_CHECKING, Self

if TYPE_CHECKING:
    from ..parsers import parser_base


class scan_base:
    """
    Base class for synchrotron measurements that scans across photon beam energies (eV).

    Links to a `parser` instance that contains `parser.data` and `parser.params`.
    The `parser.COLUMN_ASSIGNMENTS` property must reflect the correct column assignments to x/y data.
    This allows for multiple Y channels, reflecting various collectors that can be used in the beamline for absorption data.

    Parameters
    ----------
    parser : Type[parser_base] | parser_base | None
        A parser object that contains the raw data and metadata.
        Parser data is loaded, and can be re-loaded.
        If None, the scan object is empty and must be loaded from a parser object.
    load_all_columns : bool, optional
        By default False, only loads the columns defined in `parser.COLUMN_ASSIGNMENTS`.
        If True, load all columns from the parser object, including those not defined in `parser.COLUMN_ASSIGNMENTS`.

    See Also
    --------
    parser_base : Base class for synchrotron data parsers.
    """

    def __init__(
        self, parser: Type[parser_base] | parser_base | None, load_all_columns=False
    ) -> None:
        # Create link to parser object
        self.parser = parser

        # Internal variables
        self._x = None  # NDArray - single variable
        self._y = None  # NDArray - multiple variables
        self._x_errs = None  # NDArray - single variable
        self._y_errs = None  # NDArray - multiple variables
        self._x_label = None  # str
        self._x_unit = None  # str
        self._y_labels = None  # List[str]
        self._y_units = None  # List[str]

        # Load data from parser
        if self.parser is not None:
            self._load_from_parser(load_all_columns=load_all_columns)

        return

    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def y_errs(self):
        return self._y_errs

    @property
    def x_errs(self):
        return self._x_errs    

    def reload(self) -> None:
        """
        Reload data from the parser object.
        """
        self._load_from_parser()
        return

    def _load_from_parser(self, load_all_columns=False) -> None:
        """
        Load data from parser object.

        By default, only loads the y data specified by the class COLUMN_ASSIGNMENTS property.
        If `load_all_columns` is True, all columns are loaded.
        Data is indexed by the COLUMN_ASSIGNMENTS property.

        Parameters
        ----------
        load_all_columns : bool, optional

            Load all columns from the parser object, by default False.

        """
        print(self.parser._filepath)
        # Get assignments from parser object.
        assignments = self.parser.COLUMN_ASSIGNMENTS  # Validated column assignments.

        ### Load data from parser
        # X data - locate column
        x_index = self.parser.search_label_index(assignments["x"])
        # Y data - locate columns
        y_labels = assignments["y"]
        if isinstance(y_labels, list):
            y_indices = []
            for label in y_labels:
                try:
                    index = self.parser.search_label_index(label)
                    y_indices.append(index)
                except AttributeError as e:
                    warnings.warn(
                        f"Label {label} not found in parser object {self.parser}."
                    )
        else:  # Singular index.
            y_indices = [self.parser.search_label_index(assignments["y"])]
        # Y errors - locate columns
        y_errs_labels = assignments["y_errs"]
        if isinstance(y_errs_labels, list):
            y_errs_indices = [
                self.parser.search_label_index(label)
                if label in y_errs_labels and y_errs_labels[label] is not None
                else None
                for label in y_errs_labels
            ]
        elif isinstance(y_errs_labels, str):
            y_errs_indices = [self.parser.search_label_index(y_errs_labels)]
        else:  # y_errs_labels is None:
            y_errs_indices = None
        # X errors - locate column
        x_errs_label = assignments["x_errs"]
        x_errs_index = (
            self.parser.search_label_index(x_errs_label)
            if x_errs_label is not None
            else None
        )

        ### add conditional for load_all_columns.
        if load_all_columns:
            # iterate over columns, and add to y_indices if not existing indices.
            for i in range(self.parser.data.shape[1]):

                if (
                    i not in y_indices
                    and (y_errs_indices is None or i not in y_errs_indices)
                    and i != x_index
                    and (x_errs_index is None or i != x_errs_index)
                ):
                    y_indices.append(i)

        ### Generate data clones.
        # Data-points
        self._x = self.parser.data[:, x_index].copy()
        self._y = self.parser.data[:, y_indices].copy()
        self._y_errs = (
            self.parser.data[:, y_errs_indices].copy()
            if y_errs_indices is not None
            else None
        )
        self._x_errs = (
            self.parser.data[:, x_errs_index].copy()
            if x_errs_index is not None
            else None
        )
        # Labels and Units
        self._x_label = (
            self.parser.labels[x_index] if self.parser.labels is not None else None
        )
        self._x_unit = (
            self.parser.units[x_index] if self.parser.units is not None else None
        )
        self._y_labels = (
            [self.parser.labels[i] for i in y_indices]
            if self.parser.labels is not None
            else None
        )
        self._y_units = (
            [self.parser.units[i] for i in y_indices]
            if self.parser.units is not None
            else None
        )
        return

    def copy(self, *args, **kwargs) -> Type[Self]:
        """
        Creates a copy of the scan object.
        Does reload parser object data, but does link to the same parser object.

        Returns
        -------
        Type[scan_base]
            A copy of the scan object with unique data.
        """
        newobj = type(self)(parser=None, *args, **kwargs)
        newobj.parser = self.parser

        # Copy Data
        newobj._x = self._x.copy()
        newobj._y = self._y.copy()
        newobj._y_errs = self._y_errs.copy() if self._y_errs is not None else None
        newobj._x_errs = self._x_errs.copy() if self._x_errs is not None else None

        # Copy Labels and Units
        newobj._x_label = self._x_label
        newobj._x_unit = self._x_unit
        if self._y_labels is not None:
            if isinstance(self._y_labels, list):
                newobj._y_labels = [label for label in self._y_labels]
            else:  # string
                newobj._y_labels = self._y_labels
        else:
            newobj._y_labels = None
        if self._y_units is not None:
            if isinstance(self._y_units, list):
                newobj._y_units = [unit for unit in self._y_units]
            else:  # string
                newobj._y_units = self._y_units
        else:
            newobj._y_units = None
        return newobj

    def snapshot(self, columns: int = None) -> matplotlib.figure.Figure:
        """
        Generates a grid of plots, showing all scan data.

        Parameters
        ----------
        columns : int, optional
            Number of columns used to grid data, by default None
            will use the square root of the number of Y channels.
        """
        if columns is None:
            columns = np.ceil(np.sqrt(len(self._y)))
        rows = np.ceil(len(self._y) / columns)

        rcParams = {"dpi": 300, "figure.figsize": (10, 10)}
        plt.rcParams.update(rcParams)
        fig, ax = plt.subplots(rows, columns)

        for i, ydata in enumerate(self._y):
            ax[i].plot(self._x, ydata)
            ax[i].set_title(self._y_labels[i])
        return fig
