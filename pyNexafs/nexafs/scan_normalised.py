from __future__ import annotations  # For type hinting within class definitions
import numpy.typing as npt
import numpy as np
import abc
import warnings
import overrides
import datetime
from scipy import optimize as sopt
from enum import Enum
from types import NoneType
from typing import Type
from pyNexafs.nexafs.scan import scan_base, scan_abstract


class scan_abstract_normalised(scan_abstract, metaclass=abc.ABCMeta):
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

    @overrides.overrides
    def reload_labels_from_parser(self) -> None:
        return self._origin.reload_labels_from_parser()

    @property
    def ctime(self) -> datetime.datetime:
        """
        Returns the creation time of the file as a datetime object.
        """
        return self._origin.ctime

    @property
    def mtime(self) -> datetime.datetime:
        """
        Returns the modification time of the file as a datetime object.
        """
        return self._origin.mtime

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
        self._y = self._origin.y.copy()
        self._y_errs = (
            self._origin.y_errs.copy() if self._origin.y_errs is not None else None
        )

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

    @scan_abstract.x_label.getter
    def x_label(self) -> str:
        """
        Property for the x-axis label, from the origin scan object.

        Returns
        -------
        str
            The x-axis label.
        """
        return self.origin.x_label

    @x_label.setter
    def x_label(self, label: str) -> None:
        """
        Property setter for the x-axis label, to the origin scan.

        Parameters
        ----------
        label : str
            The new x-axis label.
        """
        self.origin.x_label = label

    @scan_abstract.x_unit.getter
    def x_unit(self) -> str:
        """
        Property for the x-axis unit, from the origin scan object.

        Returns
        -------
        str
            The x-axis unit.
        """
        return self.origin.x_unit

    @x_unit.setter
    def x_unit(self, unit: str) -> None:
        """
        Property setter for the x-axis unit, to the origin scan.

        Parameters
        ----------
        unit : str
            The new x-axis unit.
        """
        self.origin.x_unit = unit

    @scan_abstract.y_labels.getter
    def y_labels(self) -> list[str]:
        """
        Property for the y-axis labels, from the origin scan object.

        Returns
        -------
        list[str]
            The y-axis labels.
        """
        return self.origin.y_labels

    @y_labels.setter
    def y_labels(self, labels: list[str]) -> None:
        """
        Property setter for the y-axis labels, to the origin scan.

        Parameters
        ----------
        labels : list[str]
            The new y-axis labels.
        """
        self.origin.y_labels = labels

    @scan_abstract.y_units.getter
    def y_units(self) -> list[str]:
        """
        Property for the y-axis units, from the origin scan object.

        Returns
        -------
        list[str]
            The y-axis units.
        """
        return self.origin.y_units

    @y_units.setter
    def y_units(self, units: list[str]) -> None:
        """
        Property setter for the y-axis units, to the origin scan.

        Parameters
        ----------
        units : list[str]
            The new y-axis units.
        """
        self.origin.y_units = units


class scan_background_subtraction(scan_abstract_normalised):
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
        If both `norm_channel` and `norm_data` are simultaneously `None` or not-`None`. Only one can be defined.
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


class scan_normalised_background_channel(scan_normalised):
    """
    Normalising a scan_base to a particular channel of a background scan.

    Useful for NEXAFS such as carbon K edges. If a mesh current is used to normalise the signal,
    it will not necessarily normalise to the true absorption signal. A downstream photodiode, or alternative
    intensity channel, can be used to capture a true background signal. This channel can then
    permit double normalisation (i.e., with the mesh current signal) on the measured spectra.
    """

    @overrides.overrides
    def __init__(
        self,
        scan: Type[scan_abstract],
        background_scan: Type[scan_abstract],
        norm_channel: str,
    ) -> None:
        # Save background scan and channel information
        self._background_scan = background_scan
        self._background_channel = norm_channel
        # Collect normalisation data
        norm_channel_index = background_scan.y_labels.index(norm_channel)
        norm_data = background_scan.y[:, norm_channel_index]
        norm_data_errs = (
            background_scan.y_errs[:, norm_channel_index]
            if background_scan.y_errs is not None
            else None
        )
        super().__init__(scan, norm_data=norm_data, norm_data_errs=norm_data_errs)

    @overrides.overrides
    def _load_from_origin(self):
        # Reload background_scan data
        self._background_scan.load_from_origin()
        # Reload scan data
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
