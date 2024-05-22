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
import abc
import warnings
import overrides
from scipy import optimize as sopt
from enum import Enum
from types import NoneType
from typing import Type, TypeVar, TYPE_CHECKING, Self

if TYPE_CHECKING:
    from ..parsers import parser_base


class scan_abstract(metaclass=abc.ABCMeta):
    """
    Abstract class for defining properties of a scan object.

    Base abstract class includes x,y attributes for data, errors, labels and units.
    """

    def __init__(self) -> None:
        # Internal variables
        self._x = None  # NDArray - single variable
        self._y = None  # NDArray - multiple variables
        self._x_errs = None  # NDArray - single variable
        self._y_errs = None  # NDArray - multiple variables
        self._x_label = None  # str
        self._x_unit = None  # str
        self._y_labels = []  # List[str]
        self._y_units = []  # List[str]
        return

    @property
    def filename(self) -> str:
        """
        Property for the filename of the scan object.

        Returns
        -------
        str
            The filename of the scan object.
        """
        return ""

    @property
    def x(self) -> npt.NDArray:
        """
        Property for the 1D X data (beam energy) of the scan.

        Parameters
        ----------
        val: npt.NDArray | list[int|float]
            A new 1D array to define beam energies. Removes x_errs if new dimensions don't match.

        Returns
        -------
        npt.NDArray
            The beam energies of the scan.
        """
        return self._x

    @x.setter
    def x(self, val: npt.NDArray | list[int | float]) -> None:
        if isinstance(val, np.ndarray):
            self._x = val.copy()
        elif isinstance(val, list):
            self._x = np.array(val)
        else:
            raise ValueError("Is not a list or a np.ndarray.")
        # Remove errors if dimensions have changed.
        if self._x_errs is not None and self._x_errs.shape != self._x.shape:
            self._x_errs = None

    @property
    def y(self) -> npt.NDArray:
        """
        Property for the Y data of the scan.

        Parameters
        ----------
        val: npt.NDArray | list[int|float]
            A numpy array of data or a list of datapoints.

        Returns
        -------
        npt.NDArray
            The beam energies of the scan.
        """
        return self._y

    @y.setter
    def y(
        self, vals: npt.NDArray | list[list[int | float]] | list[int | float]
    ) -> None:
        if isinstance(vals, np.ndarray):
            if len(vals.shape) == 2:
                self._y = vals.copy()
            else:
                # Add additional axis to make y 2D instead of 1D
                self._y = vals.copy()[:, np.newaxis]
        elif isinstance(vals, list):
            if isinstance(vals[0], list):
                self._y = np.array(vals)
            else:
                # Assuming int/floats, add additional axis as well.
                self._y = np.array(vals)[:, np.newaxis]
        else:
            raise ValueError("Is not a list or a np.ndarray.")

        # Remove attributes if dimensions have changed.
        if self.y_errs is not None and self.y_errs.shape != self._y.shape:
            self.y_errs = None
        if len(self.y_labels) != len(self._y.shape[1]):
            self.y_labels = None
        if len(self.y_units) != len(self._y.shape[1]):
            self.y_units = None

    @property
    def y_errs(self) -> npt.NDArray | None:
        """
        Property for error data corresponding to the Y values.

        Parameters
        ----------
        errs : np.NDArray | None
            Error values in a numpy array matching y shape, or None to remove errors.

        Returns
        -------
        np.NDArray | None
            Errors corresponding to y values or None if not defined.
        """
        return self._y_errs

    @y_errs.setter
    def y_errs(self, errs: npt.NDArray | None) -> None:
        if errs is None:
            self._y_errs = None
        elif isinstance(errs, np.ndarray) and self.y.shape == errs.shape:
            self._y_errs = errs.copy()
        else:
            raise ValueError(
                f"Shape of errors {errs.shape} doesn't match existing y shape {self.y.shape}."
            )
        return

    @property
    def x_errs(self) -> npt.NDArray:
        """
        Property for error data corresponding to the X values.

        Parameters
        ----------
        errs : np.NDArray | None
            Error values in a numpy array matching x shape, or None to remove errors.

        Returns
        -------
        np.NDArray | None
            Errors corresponding to x values or None if not defined.
        """
        return self._x_errs

    @x_errs.setter
    def x_errs(self, errs: npt.NDArray | None) -> None:
        if errs is None:
            self._x_errs = None
        elif isinstance(errs, npt.NDArray) and errs.shape == self.x.shape:
            self._x_errs = errs.copy()
        else:
            raise ValueError(
                f"Shape of errors {errs.shape} doesn't match existing x shape {self.x.shape}."
            )

    @property
    def x_label(self) -> str:
        """
        Property for the label of the x axis.

        Parameters
        ----------
        label : str
            String to replace existing x label.

        Returns
        -------
        str
            The x label.
        """
        return self._x_label

    @x_label.setter
    def x_label(self, label: str):
        if isinstance(label, str):
            self._x_label = label
        else:
            raise ValueError(f"New label `{label}` is not string.")

    @property
    def y_labels(self) -> list[str]:
        """
        Property for the labels of the y axis.

        In the event new y data is supplied and doesn't match number of existing y_labels,
        generic y_labels are created in the form 'Data Col. N' where N is the column index.

        Parameters
        ----------
        label : list[str]
            String to replace existing y labels. Must match number of columns in the y attribute.

        Returns
        -------
        list[str]
            The y labels. If y data has been redefined without redefining y_labels, y labels are
            generated in the form 'Data Col. N'.
        """
        if self._y_labels is not None:
            return self._y_labels
        else:
            chars_col_len = np.ceil(
                np.log10(self.y.shape[1] + 1)
            )  # Gets number of characters for a given number.
            ylabel_fmt_str = "Data Col. {:" + str(chars_col_len) + "d}"
            return [ylabel_fmt_str.format(i + 1) for i in range(self.y.shape[1])]

    @y_labels.setter
    def y_labels(self, labels: list[str]) -> None:
        if isinstance(labels, list) and np.all(
            [isinstance(label, str) for label in labels]
        ):
            if len(labels) == self.y.shape[1]:
                self._y_labels = labels.copy()
            else:
                raise ValueError(
                    f"Number of labels ({len(labels)}) does not match y data columns ({self.y.shape[1]})."
                )
        else:
            raise ValueError(
                f"Provided `labels` {labels} is not a list of only strings."
            )
        return

    @property
    def x_unit(self) -> str:
        """
        Property for the unit of the x data.

        Parameters
        ----------
        unit : str
            New string for the x data unit.

        Returns
        -------
        str
            The label of the x data.
        """
        return self._x_unit

    @x_unit.setter
    def x_unit(self, unit: str) -> None:
        if isinstance(unit, str):
            self._x_unit = unit
        else:
            raise ValueError(f"Provided `unit` {unit} is not a string.")
        return

    @property
    def y_units(self) -> list[str]:
        return self._y_units

    @y_units.setter
    def y_units(self, units: list[str] | None):
        if units is None:
            self._y_units = None
        elif isinstance(units, list) and np.all(
            [isinstance(unit, str) for unit in units]
        ):
            if len(units) == self.y.shape[1]:
                self._y_units = units.copy()
            else:
                raise ValueError(
                    f"Shape of 'units' ({len(units)}) does not match shape of y data ({self.y.shape[1]})."
                )
        else:
            raise ValueError(f"Provided 'units' {units} is not a list of strings.")
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


class scan_base(scan_abstract):
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

    @overrides.overrides
    def __init__(
        self,
        parser: Type[parser_base] | parser_base | None,
        load_all_columns: bool = False,
    ) -> None:
        # Initialise data arrays
        super().__init__()

        # Store parser reference.
        self._parser = parser

        # Load data from parser
        if self.parser is not None:
            self._load_from_parser(load_all_columns=load_all_columns)

    @scan_abstract.filename.getter
    def filename(self) -> str:
        """
        Property for the filename of the scan object.

        Returns
        -------
        str
            The filename of the scan object.
        """
        return self.parser.filename

    @property
    def parser(self) -> Type[parser_base] | parser_base:
        """
        Property for the parser object linked to the scan object.

        Parameters
        ----------
        parser : Type[parser_base] | parser_base
            A parser object that contains the raw data and metadata.
            Parser data is loaded, and can be re-loaded.
            If None, the scan object is empty and must be loaded from a parser object.

        Returns
        -------
        Type[parser_base] | parser_base
            The parser object linked to the scan object.
        """
        return self._parser

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


class scan_abstract_normalised(scan_abstract):
    """
    Abstract class to define the common methods used in all normalized scans.

    Parameters
    ----------
    scan_base : Type[scan_base]
        A scan object. Could be already normalised scan or a scan_base object.
    """

    def __init__(self, scan: Type[scan_abstract]):
        self._origin = scan
        return

    @scan_abstract.filename.getter
    def filename(self) -> str:
        """
        Property for the filename of the scan object.

        Returns
        -------
        str
            The filename of the linked origin scan object.
        """
        return self._origin.filename

    @abc.abstractmethod
    def _scale_from_normalisation_data(self) -> None:
        """
        Abstract method to scale the y data according to defined normalisation data.


        """
        pass

    def _load_from_origin(self) -> None:
        """
        Method to reload data from the origin scan object. Uses instantiated scan reference.
        """
        # Copy data
        self._x = self._origin.x.copy()
        self._x_errs = (
            self._origin.x_errs.copy() if self._origin.x_errs is not None else None
        )
        self._x_label = self._origin.x_label  #
        self._x_unit = self._origin.x_unit
        self._y = self._origin.y.copy()
        self._y_errs = (
            self._origin.y_errs.copy() if self._origin.y_errs is not None else None
        )
        self._y_labels = self._origin.y_labels.copy()
        self._y_units = self._origin.y_units.copy()

    def load_and_normalise(self) -> None:
        """
        (Re)Loads data from origin and applies normalisation.

        Calls self._load_from_origin method in conjunction with
        self._scale_from_normalisation_data method.
        """
        self._load_from_origin()
        self._scale_from_normalisation_data()
        return

    @property
    def origin(self) -> Type[scan_base]:
        """
        Property for the original scan object.

        Returns
        -------
        scan_base
            The original scan object.
        """
        return self._origin


class scan_background_subtracted(scan_abstract_normalised):
    def __init__(
        self,
        scan: Type[scan_abstract],
        scan_background: Type[scan_abstract],
    ) -> None:
        if scan.x != scan_background.x:
            raise ValueError("X data for scan and background scan do not match.")
        if scan.y.shape != scan_background.y.shape:
            raise ValueError("Y data for scan and background scan do not match.")
        self._origin = scan
        self._background = scan_background
        return

    @overrides.overrides
    def _scale_from_normalisation_data(self) -> None:
        """
        Subtracts the background data from the y data.
        """
        # Determine if normalisation can occur over all signals.
        if set(self._background.y_labels) == set(self.y_labels):
            if self._background.y_labels == self.y_labels:
                # Order matches.
                self.y -= self._background.y
                if self.y_errs is not None and self._background.y_errs is not None:
                    self.y_errs = np.sqrt(
                        np.square(self._y_errs) + np.square(self._background.y_errs)
                    )
                elif self.y_errs is not None:
                    warnings.warn("No errors in background data to subtract.")
                elif self._background.y_errs is not None:
                    warnings.warn("No errors in scan data to subtract from.")
            else:
                # Requires re-matching of background indexes.
                for i, label in enumerate(self.y_labels):
                    ind = self._background.y_labels.index(label)
                    self.y[:, i] -= self._background.y[:, ind]
                    if self.y_errs is not None and self._background.y_errs is not None:
                        self.y_errs[:, i] = np.sqrt(
                            np.square(self._y_errs[:, i])
                            + np.square(self._background.y_errs[:, ind])
                        )
                    elif self.y_errs is not None:
                        warnings.warn("No errors in background data to subtract.")
                    elif self._background.y_errs is not None:
                        warnings.warn("No errors in scan data to subtract from.")
        else:
            raise ValueError(
                f"Scan Y labels {self.y_labels} don't match background Y labels {self._background.y_labels}"
            )
        return


class scan_normalised(scan_abstract_normalised):
    """
    General class for a normalisation of a scan_base.

    Normalisation is usually performed in reference to some flux measurement,
    such as the current of a mesh or photodiode. This class requires
    `norm_channel` or `norm_data` to be defined, but not both.

    Parameters
    ----------
    scan : Type[scan_base]
        The initial scan object to collect and normalise y dataseries.
    norm_channel : int | str | None, optional
        The y-channel name corresponding to normalisation data.
    norm_data : npt.NDArray | None, optional
        Custom data to normalise the y data. Must match y-data length.

    Raises
    ------
    ValueError
        If both `norm_channel` and `norm_data` are simultaneously None or not-None. Only one can be defined.
    ValueError
        If `norm_channel` is not found in y_labels.

    """

    @overrides.overrides
    def __init__(
        self,
        scan: Type[scan_base] | scan_base,
        norm_channel: str | None = None,
        norm_data: npt.NDArray | None = None,
        norm_data_errs: npt.NDArray | None = None,
    ) -> None:
        if norm_channel is None and norm_data is None:
            raise ValueError(
                "A normalisation channel label or normalisation data is required for initialisation."
            )
        elif norm_channel is not None and norm_data is not None:
            raise ValueError(
                "Both `norm_channel` and `norm_data` parameters are defined. Only one can be defined."
            )
        else:
            # Set reference for original scan object
            super().__init__(scan)

            # Store provided normalisation data
            self._norm_channel = norm_channel
            self._norm_data = norm_data
            self._norm_data_errs = norm_data_errs

            # Process normalisation data if conditions are met.
            if norm_channel is not None or len(norm_data) == len(scan.y):
                self.load_and_normalise()
            else:
                raise ValueError(
                    f"Normalisation data with shape {norm_data.shape} doesn't match length of scan y-data with shape {scan.y.shape}"
                )
            return

    @property
    def norm_data(self) -> npt.NDArray:
        """
        Property for normalisation data for the normalised scan.

        Can also be used to define errors. Setting values creates a clone.
        Upon defining new normalisation data, self.load_and_normalise are called.

        Parameters
        ----------
        values : npt.NDArray | tuple[npt.NDArray, npt.NDArray]
            A set of data values setting normalisation data and errors.
            If NDArray is 1D, assumes error values are None.
            To define error values, use tuple (data, errors) or 2D array.

        Returns
        -------
        npt.NDArray
            A 1D array of normalisation values.

        See Also
        --------
        scan_normalised.norm_data_errors
            Property for getting/setting normalisation errors.
        """
        return self._norm_data

    @norm_data.setter
    def norm_data(
        self, values: tuple[npt.NDArray, npt.NDArray | None] | npt.NDArray
    ) -> None:
        if isinstance(values, tuple):
            if (
                len(values) == 2
                and isinstance(values[0], np.ndarray)
                and isinstance(values[1], (np.ndarray, NoneType))
            ):
                norm_data, norm_data_errs = values
                if norm_data_errs is not None:
                    if norm_data.shape == norm_data_errs.shape:
                        self._norm_data = norm_data.copy()
                        self._norm_data_errs = norm_data_errs.copy()
                    else:
                        raise ValueError(
                            f"Shape of normalisation data {norm_data.shape} and errors {norm_data_errs.shape} to not match."
                        )
                else:
                    self._norm_data = norm_data.copy()
                    self._norm_data_errs = None
            else:
                raise ValueError(
                    "Tuple does not match required properties; two objects of numpy arrays, the latter allowed to be None."
                )
        elif isinstance(values, np.ndarray):
            self._norm_data = values.copy()
        else:
            raise TypeError(
                "`values` parameter is not a single numpy array or a tuple of 2 numpy arrays."
            )
        self.load_and_normalise()

    @property
    def norm_data_errs(self) -> npt.NDArray:
        """
        Returns the normalisation error data for the normalised scan.

        Can be set together with normalisation data using the `norm_data` property.

        Returns
        -------
        npt.NDArray
            A 1D array of errors.

        See Also
        --------
        scan_normalised.norm_data
            Property for getting/setting normalisation data and errors.
        """
        return self._norm_data_errs

    @overrides.overrides
    def _scale_from_normalisation_data(self) -> None:
        """
        Normalises the y data (and y_err data if present).

        If errors provided for normalisation data, incorporates into calculating new y_err data,
        using a sum of squares as follows:
            x = data,
            z = normalisation data
            y = f(x, z) = x/z,
            unc_y = u(y) = sqrt( (df/dx * unc(x)) ^2 + (df/dz * unc(z) )^2)
            df/dx = 1/z
            df/dz = -x/z**2
        """

        # Scale normalisation data point #0 to 1, so it affects from unity.
        scaled_norm = self._norm_data / self._norm_data[0]
        scaled_norm_errs = (
            self._norm_data_errs / self._norm_data[0]
            if self._norm_data_errs is not None
            else None
        )
        # Scale y, yerr data by normalisation.
        self._y /= scaled_norm
        # Scale y_errs if defined
        if self._y_errs is not None:
            if self._norm_data_errs is not None:
                # Two error sources, use sum of square errors if errors is defined:
                square_y_errs = np.square(1 / scaled_norm * self._y_errs)
                square_norm_errs = np.square(
                    self._y / scaled_norm**2 * scaled_norm_errs
                )
                self._y_errs = np.sqrt(square_y_errs + square_norm_errs)
            else:
                # No norm data errors
                self._y_errs /= scaled_norm
        elif self._norm_data_errs is not None:
            # Create y_errs from norm_data_errors.
            self._y_errs = self.y / scaled_norm**2 * scaled_norm_errs
        # pass for no definition of errors.
        return

    @overrides.overrides
    def _load_from_origin(self) -> None:
        """
        Re-loads data, refreshing data, errors, labels and units for x,y variables.

        Overrides `scan_abstract_normalised._load_from_origin` depending on the `_norm_channel` attribute being NoneType.
        """
        # Y Reloading
        if self._norm_channel is not None:
            # Collect index of normalisation data
            ind = self._origin.y_labels.index(self._norm_channel)
            # Collect normalisaton data:
            self._norm_data = self._origin.y[:, ind]
            self._norm_data_errs = (
                self._origin.y_errs[:, ind] if self._origin.y_errs is not None else None
            )

            # Collect Y data, removing index of normalisation channel.
            self._y = np.delete(
                self._origin.y, ind, 1
            )  # copy data, removing the index from the existing set of data.
            self._y_errs = (
                np.delete(self._origin.y_errs, ind, 1)
                if self._origin.y_errs is not None
                else None
            )
            self._y_labels = (
                self._origin.y_labels[0:ind] + self._origin.y_labels[ind + 1 :]
            )
            self._y_units = (
                self._origin.y_units[0:ind] + self._origin.y_units[ind + 1 :]
            )
            # Collect X data normally.
            self._x = self._origin.x.copy()
            self._x_errs = (
                self._origin.x_errs.copy() if self._origin.x_errs is not None else None
            )
            self._x_label = self._origin.x_label
            self._x_unit = self._origin.x_unit
        else:
            # Load x and y data regularly.
            super()._load_from_origin()


class scan_normalised_edges(scan_abstract_normalised):
    """
    Normalising a scan_base across pre &/ post edges.

    Uses two band definitions to normalise the y data centred around a chemical resonant edge
    such as the K-edge of carbon.

    Parameters
    ----------
    scan : Type[scan_base] | scan_base
        The initial scan object to collect and normalise y dataseries.
    pre_edge_domain : _type_, optional
        Data to define the domain of pre-edge normalisation. Can take the following data forms:
        - A list of indices (integers) to subset the y data.
        - A tuple of floats to define the inclusive-domain of the x data.
        - None, to not perform pre-edge normalisation. Overrides pre_edge_normalisation enumerate.
    post_edge_domain : _type_, optional
        Data to define the domain of post-edge normalisation. Same format as pre-edge.
        If None overrides post_edge_normalisation enumerate.
    pre_edge_normalisation : EDGE_NORM_TYPE, optional
        Normalisation type for pre-edge, by default EDGE_NORM_TYPE.LINEAR
    pre_edge_level : float, optional
        Normalisation level for pre-edge, by default 0.1.
    post_edge_normalisation : EDGE_NORM_TYPE, optional
        Normalisation type for post-edge, by default EDGE_NORM_TYPE.LINEAR
    post_edge_level : float, optional
        Normalisation level for post-edge, by default 1.0.

    Attributes
    ----------
    EDGE_NORM_TYPE : enumerate
        Enumerate types for edge normalisation.

    Raises
    ------
    ValueError
        If both `norm_channel` and `norm_data` are simultaneously None or not-None. Only one can be defined.
    ValueError
        If `norm_channel` is not found in y_labels.

    """

    class PREEDGE_NORM_TYPE(Enum):
        """
        Enumerated definitions for pre-edge normalisation.

        Attributes
        ----------
        NONE : int
            No pre-edge normalisation.
        CONSTANT : int
            Constant normalisation over the edge domain.
        LINEAR : int
            Linear normalisation over the edge domain.
        EXPONENTIAL : int
            EXPONENTIAL normalisation over the edge domain.
        """

        NONE = 0
        CONSTANT = 1
        LINEAR = 2
        EXPONENTIAL = 3

    DEFAULT_PRE_EDGE_LEVEL_CONSTANT = 0.0
    DEFAULT_PRE_EDGE_LEVEL_LINEAR = 0.0
    DEFAULT_PRE_EDGE_LEVEL_EXP = 0.1

    class POSTEDGE_NORM_TYPE(Enum):
        """
        Enumerated definitions for post-edge normalisation.

        Attributes
        ----------
        NONE : int
            No post-edge normalisation.
        CONSTANT : int
            Constant normalisation over the edge domain.
        """

        NONE = 0
        CONSTANT = 1

    DEFAULT_POST_EDGE_LEVEL_CONSTANT = 1.0

    def __init__(
        self,
        scan: Type[scan_base] | scan_base,
        pre_edge_domain=list[int] | tuple[float, float] | None,
        post_edge_domain=list[int] | tuple[float, float] | None,
        pre_edge_normalisation: PREEDGE_NORM_TYPE = PREEDGE_NORM_TYPE.CONSTANT,
        post_edge_normalisation: POSTEDGE_NORM_TYPE = PREEDGE_NORM_TYPE.CONSTANT,
        pre_edge_level: float = DEFAULT_PRE_EDGE_LEVEL_CONSTANT,
        post_edge_level: float = DEFAULT_POST_EDGE_LEVEL_CONSTANT,
    ) -> None:

        super().__init__(scan)
        # Use properties to define the values.
        self.pre_edge_domain = pre_edge_domain
        self.post_edge_domain = post_edge_domain
        self.pre_edge_normalisation = pre_edge_normalisation
        self.post_edge_normalisation = post_edge_normalisation
        # Level defined after normalisation.
        self.pre_edge_level = pre_edge_level
        self.post_edge_level = post_edge_level
        # Define variables for edge fitting
        self.pre_edge_fit_params = None
        self.post_edge_fit_params = None
        # Perform normalisation
        self._scale_from_normalisation_data()
        return

    @overrides.overrides
    def _scale_from_normalisation_data(self) -> None:
        """
        Scales y data by the normalisation regions.
        """
        # Define fitting functions
        lin_fn = lambda x, m, c: m * x + c
        exp_fn = lambda x, a, b, c: a * np.exp(b * x) + c
        # Perform dual normalisation
        if (
            self.pre_edge_normalisation
            is not scan_normalised_edges.PREEDGE_NORM_TYPE.NONE
            and self.post_edge_normalisation
            is not scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE
            and self._pre_edge_domain is not None
            and self._post_edge_domain is not None
        ):

            # Collect pre-edge indexes
            if isinstance(self.pre_edge_domain, list):
                pre_inds = self.pre_edge_domain
            elif isinstance(self.pre_edge_domain, tuple):
                pre_inds = np.where(
                    (self.x >= self.pre_edge_domain[0])
                    & (self.x <= self.pre_edge_domain[1])
                )
            else:
                raise AttributeError(
                    "Pre-edge domain is not defined correctly. Should be a list of indexes or the range in a tuple."
                )

            # Collect post-edge indexes:
            if isinstance(self.post_edge_domain, list):
                post_inds = self.post_edge_domain
            elif isinstance(self.post_edge_domain, tuple):
                post_inds = np.where(
                    (self.x >= self.post_edge_domain[0])
                    & (self.x <= self.post_edge_domain[1])
                )
            else:
                raise AttributeError(
                    "Post-edge domain is not defined correctly. Should be a list of indexes or the range in a tuple."
                )

            # Calculate pre-edge and normalise
            match self.pre_edge_normalisation:
                case scan_normalised_edges.PREEDGE_NORM_TYPE.CONSTANT:
                    mean = np.mean(self.y[pre_inds])
                    stdev = np.std(self.y[pre_inds])
                    self.pre_edge_fit_params = mean.tolist()
                    self.y -= mean + self.DEFAULT_PRE_EDGE_LEVEL_CONSTANT
                    if self.y_errs is not None:
                        self.y_errs = np.sqrt(np.square(self.y_errs) + np.square())
                case scan_normalised_edges.PREEDGE_NORM_TYPE.LINEAR:
                    popt, pcov = sopt.curve_fit(
                        lin_fn, self.x[pre_inds], self.y[pre_inds]
                    )
                    self.pre_edge_fit_params = popt
                    self.y -= lin_fn(popt) + self.DEFAULT_PRE_EDGE_LEVEL_LINEAR
                case scan_normalised_edges.PREEDGE_NORM_TYPE.EXPONENTIAL:
                    popt, pcov = sopt.curve_fit(
                        exp_fn, self.x[pre_inds], self.y[pre_inds]
                    )
                    self.y = self.y - exp_fn(popt) + self.DEFAULT_PRE_EDGE_LEVEL_EXP
                case _:
                    # Should never reach here, and dual normalisation excludes None type.
                    raise ValueError("Pre-edge normalisation type not defined.")

            # Calculate post-edge and normalise
            postave = np.mean(self.y[post_inds])

            # Normalise data.

        # Perform single normalisation
        elif (
            self.pre_edge_normalisation
            is not scan_normalised_edges.PREEDGE_NORM_TYPE.NONE
        ):
            # Collect pre-edge indexes
            if isinstance(self.pre_edge_domain, list):
                pre_inds = self.pre_edge_domain
            elif isinstance(self.pre_edge_domain, tuple):
                pre_inds = np.where(
                    (self.x >= self.pre_edge_domain[0])
                    & (self.x <= self.pre_edge_domain[1])
                )
            else:
                raise AttributeError("Pre-edge domain is not defined correctly.")

            # Calculate pre-edge and normalise
            match self.pre_edge_normalisation:
                case scan_normalised_edges.EDGE_NORM_TYPE.CONSTANT:
                    mean = np.mean(self.y[pre_inds])
                    self.pre_edge_fit_params = mean.tolist()
                    self.y -= mean + self.DEFAULT_PRE_EDGE_LEVEL_CONSTANT
                case scan_normalised_edges.EDGE_NORM_TYPE.LINEAR:
                    popt, pcov = sopt.curve_fit(
                        lin_fn, self.x[pre_inds], self.y[pre_inds]
                    )
                    self.pre_edge_fit_params = popt
                    self.y -= lin_fn(popt) + self.DEFAULT_PRE_EDGE_LEVEL_LINEAR
                case scan_normalised_edges.EDGE_NORM_TYPE.EXPONENTIAL:
                    popt, pcov = sopt.curve_fit(
                        exp_fn, self.x[pre_inds], self.y[pre_inds]
                    )
                    self.y = self.y - exp_fn(popt) + self.DEFAULT_PRE_EDGE_LEVEL_EXP
                case _:
                    # Should never reach here, and dual normalisation excludes None type.
                    raise ValueError("Pre-edge normalisation type not defined.")

        # Perform single normalisation
        elif (
            self.post_edge_normalisation
            is not scan_normalised_edges.EDGE_NORM_TYPE.NONE
        ):
            raise NotImplemented("Incomplete function.")

        # No normalisation specified, do nothing.
        else:
            pass
        return

    @property
    def pre_edge_domain(self) -> list[int] | tuple[float, float] | None:
        """
        A property defining the pre-edge domain of normalisation.

        If setting and `pre_edge_normalisation` is `NONE`, will set new normalisation enumerate to `LINEAR`.

        Parameters
        ----------
        vals : list[int] | tuple[float, float] | None
            Can be defined using a list of included indices matching x datapoints.
            Alternatively a tuple of two floats defining the inclusive endpoints.
            Alternatively None stops normalisation being performed on the pre-edge.

        Returns
        -------
        list[int] | tuple[float, float] | None
            Same as vals parameter.
        """
        return self._pre_edge_domain

    @pre_edge_domain.setter
    def pre_edge_domain(self, vals: list[int] | tuple[float, float] | None) -> None:
        if isinstance(vals, list):
            self._pre_edge_domain = vals.copy()  # non-immutable
        elif isinstance(vals, tuple) and len(vals) == 2:
            self._pre_edge_domain = vals  # immutable, can't modify.
        elif vals is None:
            # Remove vals
            del self.pre_edge_domain
        else:
            raise ValueError(
                "The pre-edge domain needs to be defined by a list of integer indices, a tuple of inclusive endpoints or None."
            )
        # Default normalisation to linear if not already defined.
        if (
            self._pre_edge_normalisation is scan_normalised_edges.EDGE_NORM_TYPE.NONE
            and vals is not None
        ):
            self._pre_edge_normalisation = scan_normalised_edges.EDGE_NORM_TYPE.LINEAR

    @pre_edge_domain.deleter
    def pre_edge_domain(self):
        self._post_edge_domain = None

    @property
    def pre_edge_normalisation(self) -> scan_normalised_edges.EDGE_NORM_TYPE:
        """
        Property to define the type of normalisation performed on the pre-edge.

        Parameters
        ----------
        vals : scan_normalised_edges.EDGE_NORM_TYPE
            LINEAR, EXPONENTIAL or NONE.

        Returns
        -------
        scan_normalised_edges.EDGE_NORM_TYPE
            The current normalisation type.
        """
        return self._post_edge_normalisation

    @pre_edge_normalisation.setter
    def pre_edge_normalisation(
        self, vals: scan_normalised_edges.EDGE_NORM_TYPE
    ) -> None:
        if vals in scan_normalised_edges.EDGE_NORM_TYPE:
            self._pre_edge_normalisation = vals
        else:
            raise ValueError(f"{vals} not in {scan_normalised_edges.EDGE_NORM_TYPE}.")
        # Change pre-edge level by default to a reasonable value if setting exponential.
        if (
            vals is scan_normalised_edges.EDGE_NORM_TYPE.EXPONENTIAL
            and self.pre_edge_level <= 0
        ):
            self._pre_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_EXP

    @pre_edge_normalisation.deleter
    def pre_edge_normalisation(self) -> None:
        self._pre_edge_normalisation = None

    @property
    def pre_edge_level(self) -> float:
        """
        Property to define the normalisation level for the pre-edge.

        Parameters
        ----------
        vals : float
            The normalisation average.

        Returns
        -------
        float
            The normalisation level.
        """
        return self._pre_edge_level

    @pre_edge_level.setter
    def pre_edge_level(self, vals: float) -> None:
        if (
            self.pre_edge_normalisation
            is scan_normalised_edges.EDGE_NORM_TYPE.EXPONENTIAL
        ):
            if vals <= 0:
                raise ValueError(
                    "Exponential normalisation requires a positive, non-zero level."
                )
        self._pre_edge_level = vals

    @pre_edge_level.deleter
    def pre_edge_level(self) -> None:
        if (
            self.pre_edge_normalisation
            is scan_normalised_edges.EDGE_NORM_TYPE.EXPONENTIAL
        ):
            self._pre_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_EXP
        # elif self.pre_edge_normalisation is scan_normalised_edges.EDGE_NORM_TYPE.LINEAR:
        # self._pre_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
        else:
            self._pre_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_LINEAR

    @property
    def post_edge_domain(self) -> list[int] | tuple[float, float] | None:
        """
        A property defining the post-edge domain of normalisation.

        If setting and `post_edge_normalisation` is `NONE`, will set new normalisation enumerate to `LINEAR`.

        Parameters
        ----------
        vals : list[int] | tuple[float, float] | None
            Can be defined using a list of included indices matching x datapoints.
            Alternatively a tuple of two floats defining the inclusive endpoints.
            Alternatively None stops normalisation being performed on the post-edge.

        Returns
        -------
        list[int] | tuple[float, float] | None
            Same as vals parameter.
        """
        return self._post_edge_domain

    @post_edge_domain.setter
    def post_edge_domain(self, vals: list[int] | tuple[float, float] | None) -> None:
        if isinstance(vals, list):
            self._post_edge_domain = vals.copy()  # non-immutable
        elif isinstance(vals, tuple) and len(vals) == 2:
            self._post_edge_domain = vals  # immutable, can't modify.
        elif vals is None:
            # Remove vals and perform any other
            del self.post_edge_domain
        else:
            raise ValueError(
                "The post-edge domain needs to be defined by a list of integer indices, a tuple of inclusive endpoints or None."
            )
        # Default normalisation to linear if not already defined.
        if (
            self._post_edge_normalisation is scan_normalised_edges.EDGE_NORM_TYPE.NONE
            and vals is not None
        ):
            self._post_edge_normalisation = scan_normalised_edges.EDGE_NORM_TYPE.LINEAR

    @post_edge_domain.deleter
    def post_edge_domain(self):
        self._post_edge_domain = None

    @property
    def post_edge_normalisation(self) -> scan_normalised_edges.EDGE_NORM_TYPE:
        """
        Property to define the type of normalisation performed on the post-edge.

        Parameters
        ----------
        vals : scan_normalised_edges.EDGE_NORM_TYPE
            LINEAR, EXPONENTIAL or NONE.

        Returns
        -------
        scan_normalised_edges.EDGE_NORM_TYPE
            The current normalisation type.
        """
        return self._post_edge_normalisation

    @post_edge_normalisation.setter
    def post_edge_normalisation(
        self, vals: scan_normalised_edges.EDGE_NORM_TYPE
    ) -> None:
        if vals in scan_normalised_edges.EDGE_NORM_TYPE:
            self._post_edge_normalisation = vals
        else:
            raise ValueError(f"{vals} not in {scan_normalised_edges.EDGE_NORM_TYPE}.")

    @post_edge_normalisation.deleter
    def post_edge_normalisation(self) -> None:
        self._post_edge_normalisation = None

    @property
    def post_edge_level(self) -> float:
        """
        Property to define the normalisation level for the post-edge.

        Parameters
        ----------
        vals : float
            The normalisation average.

        Returns
        -------
        float
            The normalisation level.
        """
        return self._post_edge_level

    @post_edge_level.setter
    def post_edge_level(self, vals: float) -> None:
        if (
            self.post_edge_normalisation
            is scan_normalised_edges.EDGE_NORM_TYPE.EXPONENTIAL
        ):
            if vals <= 0:
                raise ValueError(
                    "Exponential normalisation requires a positive, non-zero level."
                )
        self._post_edge_level = vals

    @post_edge_level.deleter
    def post_edge_level(self) -> None:
        if (
            self.post_edge_normalisation
            is scan_normalised_edges.EDGE_NORM_TYPE.EXPONENTIAL
        ):
            self._post_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_EXP
        # elif self.pre_edge_normalisation is scan_normalised_edges.EDGE_NORM_TYPE.LINEAR:
        # self._pre_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
        else:
            self._post_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
