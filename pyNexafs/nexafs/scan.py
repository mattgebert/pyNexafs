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
from types import NoneType
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
        self._y_labels = []  # List[str]
        self._y_units = []  # List[str]

        # Load data from parser
        if self.parser is not None:
            self._load_from_parser(load_all_columns=load_all_columns)

        return

    @property
    def x(self) -> npt.NDArray:
        return self._x
    
    @property
    def y(self) -> npt.NDArray:
        return self._y
    
    @property
    def y_errs(self) -> npt.NDArray:
        return self._y_errs

    @property
    def x_errs(self) -> npt.NDArray:
        return self._x_errs    
    
    @property
    def x_label(self) -> str:
        return self._x_label
    
    @property
    def y_labels(self) -> list[str]:
        return self._y_labels
    
    @property
    def x_unit(self) -> str:
        return self._x_unit
    
    @property
    def y_units(self) -> list[str]:
        return self._y_units

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
    
class scan_normalised(scan_base):
    """
    General class for a normalised scan_base.
    
    normalisation is usually performed in reference to some flux measurement,
    such as the current of a mesh or photodiode.
    Requires `norm_channel` or `norm_data` to be defined, but not both.

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
    def __init__(self, 
                 scan: Type[scan_base] | scan_base,
                 norm_channel: str | None = None,
                 norm_data: npt.NDArray | None = None,
                 norm_data_errs: npt.NDArray | None = None) -> None:
        if norm_channel is None and norm_data is None:
            raise ValueError("A normalisation channel label or normalisation data is required for initialisation.")
        elif norm_channel is not None and norm_data is not None:
            raise ValueError("Both `norm_channel` and `norm_data` parameters are defined. Only one can be defined.")
        else:
            # Collect data if using norm_channel label.
            if norm_channel is not None:
                # Collect index of normalisation data
                ind = scan.y_labels.index(norm_channel)
                
                # Collect normalisation data
                self.norm_data = (
                    scan.y[:, ind],
                    scan.y_errs[:, ind] if scan.y_errs is not None else None
                )
                
                # Collect regular data
                self._y = np.delete(scan.y, ind, 1) #copy data, removing the index from the existing set of data.
                self._y_errs = np.delete(scan.y_errs, ind, 1) if scan.y_errs is not None else None
                self._y_labels = scan.y_labels[0:ind] + scan.y_labels[ind+1:]
                
            elif len(norm_data) == len(scan.y):
                # Copy normalisation data
                self._norm_data = norm_data.copy()
                self._norm_data_errs = norm_data_errs.copy() if norm_data_errs is not None else None
                
                # Copy data
                self._y = scan.y.copy()
                self._y_errs = scan.y_errs.copy() if scan.y_errs is not None else None
                self._y_labels = scan.y_labels.copy()
            else:
                raise ValueError(f"Normalisation data with shape {norm_data.shape} doesn't match length of scan y-data with shape {scan.y.shape}")
            
            # Set reference for original scan object
            self._origin = scan
            self._norm_channel = norm_channel
            
            ## Load other y-data.
            self._load_from_origin()
            
            ## Perform normalisation on data.
            self._scale_from_norm_data()
                
            return
    
    @property
    def norm_data(self) -> npt.NDArray:
        """
        Property for normalisation data for the normalised scan.
        
        Can also be used to define errors. Setting values creates a clone.
        
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
    def norm_data(self, values: tuple[npt.NDArray, npt.NDArray | None] | npt.NDArray) -> None:
        if isinstance(values, tuple):
            if len(values) == 2 and isinstance(values[0], np.ndarray) and isinstance(values[1], (np.ndarray, NoneType)):
                norm_data, norm_data_errs = values
                if norm_data_errs is not None:
                    if norm_data.shape == norm_data_errs.shape:
                        self._norm_data = norm_data.copy()
                        self._norm_data_errs = norm_data_errs.copy()
                    else:
                        raise ValueError(f"Shape of normalisation data {norm_data.shape} and errors {norm_data_errs.shape} to not match.")
                else:
                    self._norm_data = norm_data.copy()
                    self._norm_data_errs = None
            else:
                raise ValueError("Tuple does not match required properties; two objects of numpy arrays, the latter allowed to be None.")
        elif isinstance(values, np.ndarray):
            self._norm_data = values.copy()
        else:
            raise TypeError("`values` parameter is not tuple[] or numpy array.")
        
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
    
    def _scale_from_norm_data(self) -> None:
        """
        normalises the y data (and y_err data if present).
        
        If errors provided for normalisation data, incorporates into calculating new y_err data,
        using a sum of squares as follows:
            y = f(x, z) = x/z, x = data, z = normalisation data
            u(y) = sqrt( (df/dx * unc(x)) ^2 + (df/dz * unc(z) )^2)
            df/dx = 1/z
            df/dz = -x/z**2
        """
        
        # Scale normalisation data point #0 to 1, so it affects from unity.
        scaled_norm = self._norm_data / self._norm_data[0]
        scaled_norm_errs = self._norm_data_errs / self._norm_data[0] if self._norm_data_errs is not None else None
        # Scale y, yerr data by normalisation.
        self._y /= scaled_norm
        # Scale y_errs if defined
        if self._y_errs is not None:
            if self._norm_data_errs is not None:
                # Two error sources, use sum of square errors if errors is defined:
                square_y_errs = np.square(1/scaled_norm * self._y_errs)
                square_norm_errs = np.square(self._y / scaled_norm**2 * scaled_norm_errs)
                self._y_errs = np.sqrt(square_y_errs + square_norm_errs)
            else:
                # No norm data errors
                self._y_errs /= scaled_norm
        elif self._norm_data_errs is not None:
                # Create y_errs from norm_data_errors.
                self._y_errs = self.y / scaled_norm**2 * scaled_norm_errs
        # pass for no definition of errors.
        return
            
    
    def _load_from_origin(self) -> None:
        """
        Re-loads data, refreshing data, errors, labels and units for x,y variables.
        """
        # X Reloading
        self._x = self._origin.x.copy()
        self._x_errs = self._origin.x_errs.copy() if self._origin.x_errs is not None else None
        self._x_label = self._origin.x_label
        self._x_unit = self._origin.x_unit
        # Y Reloading
        if self._norm_channel is not None:
            # Collect index of normalisation data
            ind = self._origin.y_labels.index(self._norm_channel)
            # Collect regular data
            self._y = np.delete(self._origin.y, ind, 1) #copy data, removing the index from the existing set of data.
            self._y_errs = np.delete(self._origin.y_errs, ind, 1) if self._origin.y_errs is not None else None
            self._y_labels = self._origin.y_labels[0:ind] + self._origin.y_labels[ind+1:]
            self._y_units = self._origin.y_units[0:ind] + self._origin.y_units[ind+1:]
        else:
            # Copy data
            self._y = self._origin.y.copy()
            self._y_errs = self._origin.y_errs.copy() if self._origin.y_errs is not None else None
            self._y_labels = self._origin.y_labels.copy()
            self._y_units = self._origin.y_units.copy()
            
#class scan_normalised_subset(scan_normalised):
