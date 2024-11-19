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
from typing import Type, Self
from pyNexafs.nexafs.scan import scan_base, scan_abstract


class scan_abstract_normalised(scan_abstract, metaclass=abc.ABCMeta):
    """
    Abstract class to define the common methods used in all normalized scans.

    Parameters
    ----------
    scan_base : Type[scan_base]
        A scan object. Could be already normalised scan or a scan_base object.
    """

    def __init__(self, scan: scan_abstract):
        self._origin = scan
        return

    @overrides.overrides
    @abc.abstractmethod
    def copy(self) -> scan_abstract_normalised:
        """
        Returns a copy of the scan object.

        Returns
        -------
        scan_abstract
            A copy of the scan object.
        """
        copy_obj = type(self)(scan=self._origin)
        return copy_obj

    @property
    def origin(self) -> scan_abstract:
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
    def x_label(self, label: str | None) -> None:
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

    @scan_abstract.y.setter
    def y(
        self, vals: npt.NDArray | list[list[int | float]] | list[int | float]
    ) -> None:
        """
        Property setter for the y data.

        Similar to @scan_abstract.y, but avoids resetting the `y_units`
        and `y_labels` attributes when setting the y data dimension, as
        labels should always be inherited from the original scan object.

        Parameters
        ----------
        values : npt.NDArray
            The new y data.
        """
        # Set Y values
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

        # Remove errs if number of yvals have changed.
        if self.y_errs is not None and self.y_errs.shape != self._y.shape:
            self.y_errs = None

        # Do not remove unit/labels if number of yvals have changed!

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
    def y_labels(self, labels: list[str] | None) -> None:
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
    def y_units(self, units: list[str] | None) -> None:
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

        super().__init__(scan)
        self._background = scan_background

        # Run the normalisation
        self.load_and_normalise()
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

    @overrides.overrides
    def copy(self) -> scan_background_subtraction:
        return scan_background_subtraction(self._origin, self._background)


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
        An alternative to y-channel; Custom data to normalise the y data. Must match y-data length.

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
        scan: Type[scan_base],
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
            """The normalisation channel label, if defined."""
            self._norm_idx: int | None = None
            """The index of the normalisation channel in the origin scan, if `norm_channel` was provided."""
            self._norm_data = norm_data
            """The normalisation data, if defined."""
            self._norm_data_errs = norm_data_errs
            """The normalisation data errors, if defined."""

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
        if len(self._y.shape) == 1:
            self._y /= scaled_norm
        else:
            self._y /= scaled_norm[:, np.newaxis]

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

        Overrides `scan_abstract_normalised._load_from_origin` depending on the `_norm_channel` attribute being None.
        """
        # Y Reloading
        if self._norm_channel is not None:
            # Collect index of normalisation data
            self._norm_idx = ind = self._origin.y_labels.index(self._norm_channel)
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

            # Collect X data normally.
            self._x = self._origin.x.copy()
            self._x_errs = (
                self._origin.x_errs.copy() if self._origin.x_errs is not None else None
            )
        else:
            # Load x and y data regularly.
            super()._load_from_origin()

    @scan_abstract_normalised.y_labels.getter
    def y_labels(self) -> list[str]:
        """
        Property for the y-axis labels, from the origin scan object.
        When the normalisation channel is defined, the labels are adjusted to remove the normalisation channel.

        Parameters
        ----------
        labels : list[str] | None
            The new y-axis labels.

        Returns
        -------
        list[str]
            The y-axis labels.
        """
        origin_labels = self._origin.y_labels
        idx = self._norm_idx
        if idx is not None:
            return origin_labels[0:idx] + origin_labels[idx + 1 :]
        else:
            return origin_labels

    @y_labels.setter
    def y_labels(self, labels: list[str] | None) -> None:
        idx = self._norm_idx
        if idx is not None:
            if labels is not None:
                # Include the normalisation channel in the labels when setting.
                self._origin.y_labels = (
                    labels[0:idx] + [self._norm_channel] + labels[idx:]
                )
        # Otherwise set labels normally.
        self._origin.y_labels = labels

    @scan_abstract_normalised.y_units.getter
    def y_units(self) -> list[str] | None:
        """
        Property for the y-axis units, from the origin scan object.
        When the normalisation channel is defined, the units are adjusted to remove the normalisation channel.

        Parameters
        ----------
        labels : list[str] | None
            The new y-axis units.

        Returns
        -------
        list[str] | None
            The y-axis units.
        """
        origin_units = self._origin.y_units
        idx = self._norm_idx
        if idx is not None and origin_units is not None:
            return origin_units[0:idx] + origin_units[idx + 1 :]
        else:
            return origin_units

    @y_units.setter
    def y_units(self, units: list[str] | None) -> None:
        idx = self._norm_idx
        if idx is not None:
            if units is not None:
                # Include the normalisation channel in the units when setting.
                self._origin.y_units = units[0:idx] + [self._norm_channel] + units[idx:]
                return
        # Otherwise, set units normally.
        self._origin.y_units = units
        return

    def copy(self) -> Self:
        """
        Returns a copy of the scan object.

        Returns
        -------
        scan_abstract
            A copy of the scan object.
        """
        return scan_normalised(self._origin, norm_channel=self._norm_channel)


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
        scan: scan_abstract,
        background_scan: scan_abstract,
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
        if hasattr(self._background_scan, "_load_from_origin"):
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
    """The constant offset for the pre-edge normalisation. I.e. Pre-edge at 0.0"""

    DEFAULT_PRE_EDGE_LEVEL_LINEAR = 0.0
    """The constant gradient for the pre-edge normalisation. I.e. Linear gradient at 0.0"""

    DEFAULT_PRE_EDGE_LEVEL_EXP = 0.0
    """The constant exponential level for the pre-edge normalisation. I.e. Exponential level at 0.0"""

    LIN_FN = lambda x, m, c: m * x + c
    """
    A function to define a linear fit.

    Parameters
    ----------
    x : ArrayLike
        The x value.
    m : float
        The gradient of the line.
    c : float
        The y-intercept of the line.
    """

    EXP_FN_OFFSET = lambda x, a, b, c: a * np.exp(b * (x - x.min())) + c
    """
    A function to define an exponential fit, adjusted to the minimum x value.

    Parameters
    ----------
    x : ArrayLike
        The x value.
    a : float
        The amplitude of the exponential.
    b : float
        The decay constant of the exponential.
    c : float
        The offset of the exponential.
    """

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
    """The constant offset for the post-edge normalisation. I.e. Post-edge at 1.0"""

    def __init__(
        self,
        scan: Type[scan_abstract],
        pre_edge_domain: list[int] | tuple[float, float] | None,
        post_edge_domain: list[int] | tuple[float, float] | None,
        pre_edge_normalisation: PREEDGE_NORM_TYPE = PREEDGE_NORM_TYPE.CONSTANT,
        post_edge_normalisation: POSTEDGE_NORM_TYPE = POSTEDGE_NORM_TYPE.CONSTANT,
        pre_edge_level: float | None = None,
        post_edge_level: float | None = None,
    ) -> None:
        # Initialize the scan_abstract_normalised class.
        super().__init__(scan)
        # Pre-init vars
        self._pre_edge_normalisation = scan_normalised_edges.PREEDGE_NORM_TYPE.NONE
        self._post_edge_normalisation = scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE
        self._pre_edge_level = None
        self._post_edge_level = None
        # Use properties to define the values.
        self.pre_edge_domain = pre_edge_domain
        self.post_edge_domain = post_edge_domain
        # Setup normalisation types
        self.pre_edge_normalisation = pre_edge_normalisation
        self.post_edge_normalisation = post_edge_normalisation
        # Level defined after normalisation type
        if pre_edge_level is not None:
            self.pre_edge_level = pre_edge_level
        if post_edge_level is not None:
            self.post_edge_level = post_edge_level
        # Define variables for edge fitting
        self.pre_edge_fit_params: list | None = None
        self.post_edge_fit_params: list | None = None

        # Perform normalisation
        self.load_and_normalise()
        return

    @overrides.overrides
    def copy(self) -> scan_normalised_edges:
        clone = scan_normalised_edges(
            self._origin,
            pre_edge_domain=self._pre_edge_domain,
            post_edge_domain=self._post_edge_domain,
            pre_edge_normalisation=self.pre_edge_normalisation,
            post_edge_normalisation=self.post_edge_normalisation,
            pre_edge_level=self._pre_edge_level,
            post_edge_level=self._post_edge_level,
        )
        return clone

    @overrides.overrides
    def _scale_from_normalisation_data(self) -> None:
        """
        Scales y data by the normalisation regions.
        """
        # Define fitting functions
        lin_fn = scan_normalised_edges.LIN_FN
        exp_fn = scan_normalised_edges.EXP_FN_OFFSET
        # Perform dual normalisation, sequentially to avoid duplicate code.

        ## PRE-EDGE
        if (
            self.pre_edge_normalisation
            is not scan_normalised_edges.PREEDGE_NORM_TYPE.NONE
            and self._pre_edge_domain is not None
        ):

            # Collect pre-edge indexes
            if isinstance(self.pre_edge_domain, list):
                # Use the list of indexes
                pre_inds = self.pre_edge_domain
                if len(pre_inds) == 0:
                    raise ValueError("Pre-edge index list is empty.")
            elif isinstance(self.pre_edge_domain, tuple):
                # Use the domain range
                pre_inds = np.asarray(
                    (self.x >= self.pre_edge_domain[0])
                    & (self.x <= self.pre_edge_domain[1])
                ).nonzero()
                # Check indexes of each tuple element
                if len(pre_inds[0]) == 0:
                    raise ValueError(
                        f"Pre-edge domain ({self.pre_edge_domain[0]} to {self.pre_edge_domain[1]}) contains no datapoints."
                    )
            else:
                raise AttributeError(
                    "Pre-edge domain is not defined correctly. Should be a list of indexes or the range in a tuple."
                )

            # Calculate pre-edge on all y values and normalise
            match self.pre_edge_normalisation:
                case scan_normalised_edges.PREEDGE_NORM_TYPE.CONSTANT:
                    mean = np.mean(self.y[pre_inds], axis=0)
                    stdev = np.std(self.y[pre_inds], axis=0)
                    self.y += -mean + self.pre_edge_level
                    self.pre_edge_fit_params = mean.tolist()
                    # if self.y_errs is not None:
                    #     # Add the standard deviation
                    #     self.y_errs = np.sqrt(np.square(self.y_errs) + stdev ** 2)

                case scan_normalised_edges.PREEDGE_NORM_TYPE.LINEAR:
                    params: list = []
                    for i in range(self.y.shape[1]):
                        popt, pcov = sopt.curve_fit(
                            lin_fn, self.x[pre_inds], self.y[:, i][pre_inds]
                        )
                        self.pre_edge_fit_params = popt.tolist()
                        self.y[:, i] += -lin_fn(self.x, *popt) + self.pre_edge_level
                        params += popt.tolist()

                    self.pre_edge_fit_params = params

                case scan_normalised_edges.PREEDGE_NORM_TYPE.EXPONENTIAL:
                    params: list = []
                    for i in range(self.y.shape[1]):
                        # Subtract offset to fit
                        xfit = self.x[pre_inds] - self.x[pre_inds].min()
                        x = self.x - self.x[pre_inds].min()
                        a0 = self.y[:, i][pre_inds].max() - self.y[:, i][pre_inds].min()
                        # Fit
                        popt, pcov = sopt.curve_fit(
                            exp_fn,
                            xfit,
                            self.y[:, i][pre_inds],
                            p0=(a0, -0.5, 0),
                            bounds=[(-np.inf, -np.inf, -np.inf), (np.inf, 0, np.inf)],
                            maxfev=10000,
                        )
                        self.y[:, i] += -exp_fn(x, *popt) + self.pre_edge_level
                        params += popt.tolist()

                    self.pre_edge_fit_params = params

                case scan_normalised_edges.PREEDGE_NORM_TYPE.NONE:
                    # Do nothing for no pre-edge normalisation
                    pass
                case _:
                    # Should never reach here, and dual normalisation excludes None type.
                    raise ValueError("Pre-edge normalisation type not defined.")

        ## POST-EDGE
        if (
            self.post_edge_normalisation
            is not scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE
            and self._post_edge_domain is not None
        ):

            # Collect post-edge indexes:
            if isinstance(self.post_edge_domain, list):
                post_inds = self.post_edge_domain
                if len(post_inds) == 0:
                    raise ValueError("Post-edge index list is empty.")
            elif isinstance(self.post_edge_domain, tuple):
                post_inds = np.where(
                    (self.x >= self.post_edge_domain[0])
                    & (self.x <= self.post_edge_domain[1])
                )
                # Check indexes of each tuple element
                if len(post_inds[0]) == 0:
                    raise ValueError(
                        f"Post-edge domain ({self.post_edge_domain[0]} to {self.post_edge_domain[1]}) contains no datapoints."
                    )
            else:
                raise AttributeError(
                    "Post-edge domain is not defined correctly. Should be a list of indexes or the range in a tuple."
                )

            if len(post_inds) == 0:
                raise ValueError("Post-edge domain is empty.")

            # Calculate post-edge and normalise. Note, y data is already normalised by pre-edge.
            match self.post_edge_normalisation:
                case scan_normalised_edges.POSTEDGE_NORM_TYPE.CONSTANT:
                    # Get the mean of the post-edge from the pre-edge level.
                    postave = np.mean(self.y[post_inds] - self.pre_edge_level, axis=0)
                    # Scale to difference from pre-edge level to post-edge level. Ignore zero values.
                    scale = np.zeros(postave.shape)
                    scale[postave != 0] = (
                        self.post_edge_level - self.pre_edge_level
                    ) / postave[postave != 0]
                    self.y = (
                        self.y - self.pre_edge_level
                    ) * scale + self.pre_edge_level
                    pass
                case scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE:
                    # Do nothing
                    pass
                case _:
                    # Should never reach here, and dual normalisation excludes None type.
                    raise ValueError("Post-edge normalisation type not defined.")

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
            self._pre_edge_normalisation is scan_normalised_edges.PREEDGE_NORM_TYPE.NONE
            and vals is not None
        ):
            self._pre_edge_normalisation = (
                scan_normalised_edges.PREEDGE_NORM_TYPE.LINEAR
            )

    @pre_edge_domain.deleter
    def pre_edge_domain(self):
        self._post_edge_domain = None

    @property
    def pre_edge_normalisation(self) -> scan_normalised_edges.PREEDGE_NORM_TYPE:
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
        return self._pre_edge_normalisation

    @pre_edge_normalisation.setter
    def pre_edge_normalisation(
        self, vals: scan_normalised_edges.PREEDGE_NORM_TYPE
    ) -> None:
        if vals in scan_normalised_edges.PREEDGE_NORM_TYPE:
            self._pre_edge_normalisation = vals
        else:
            raise ValueError(
                f"{vals} not in {scan_normalised_edges.PREEDGE_NORM_TYPE}."
            )
        # If pre-edge not defined, set to default level
        if self._pre_edge_level is None:
            match vals:
                case scan_normalised_edges.PREEDGE_NORM_TYPE.CONSTANT:
                    self._pre_edge_level = (
                        scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_CONSTANT
                    )
                case scan_normalised_edges.PREEDGE_NORM_TYPE.LINEAR:
                    self._pre_edge_level = (
                        scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
                    )
                case scan_normalised_edges.PREEDGE_NORM_TYPE.EXPONENTIAL:
                    self._pre_edge_level = (
                        scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_EXP
                    )
                case _:
                    # Do nothing if not defined / NONE.
                    pass

        # Change pre-edge level by default to a reasonable value if setting exponential.
        if (
            vals is scan_normalised_edges.PREEDGE_NORM_TYPE.EXPONENTIAL
            and self.pre_edge_level <= 0
        ):
            self._pre_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_EXP

    @pre_edge_normalisation.deleter
    def pre_edge_normalisation(self) -> None:
        self._pre_edge_normalisation = None

    @property
    def pre_edge_level(self) -> float:
        """
        Property to define the normalisation level (constant) for the pre-edge.

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
            is scan_normalised_edges.PREEDGE_NORM_TYPE.EXPONENTIAL
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
            is scan_normalised_edges.PREEDGE_NORM_TYPE.EXPONENTIAL
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
        # Default normalisation to constant if not already defined.
        if (
            self._post_edge_normalisation
            is scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE
            and vals is not None
        ):
            self._post_edge_normalisation = (
                scan_normalised_edges.POSTEDGE_NORM_TYPE.CONSTANT
            )

    @post_edge_domain.deleter
    def post_edge_domain(self):
        self._post_edge_domain = None

    @property
    def post_edge_normalisation(self) -> scan_normalised_edges.POSTEDGE_NORM_TYPE:
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
        self, vals: scan_normalised_edges.POSTEDGE_NORM_TYPE
    ) -> None:
        if vals in scan_normalised_edges.POSTEDGE_NORM_TYPE:
            self._post_edge_normalisation = vals
        else:
            raise ValueError(
                f"{vals} not in {scan_normalised_edges.POSTEDGE_NORM_TYPE}."
            )

        # Set default if None
        if self._post_edge_level is None:
            match vals:
                case scan_normalised_edges.POSTEDGE_NORM_TYPE.CONSTANT:
                    self._post_edge_level = (
                        scan_normalised_edges.DEFAULT_POST_EDGE_LEVEL_CONSTANT
                    )
                case _:
                    # Do nothing if not defined / NONE.
                    pass

    @post_edge_normalisation.deleter
    def post_edge_normalisation(self) -> None:
        self._post_edge_normalisation = None

    @property
    def post_edge_level(self) -> float:
        """
        Property to define the normalisation level (constant) for the post-edge.

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
        # if (
        #     self.post_edge_normalisation
        #     is scan_normalised_edges.POSTEDGE_NORM_TYPE.EXPONENTIAL
        # ):
        #     if vals <= 0:
        #         raise ValueError(
        #             "Exponential normalisation requires a positive, non-zero level."
        #         )
        self._post_edge_level = vals

    @post_edge_level.deleter
    def post_edge_level(self) -> None:
        if (
            # self.post_edge_normalisation
            # is scan_normalised_edges.POSTEDGE_NORM_TYPE.EXPONENTIAL
            False
        ):
            self._post_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_EXP
        # elif self.pre_edge_normalisation is scan_normalised_edges.EDGE_NORM_TYPE.LINEAR:
        # self._pre_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_LINEAR
        else:
            self._post_edge_level = scan_normalised_edges.DEFAULT_PRE_EDGE_LEVEL_LINEAR


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # Create a basic scan object form test data
    MEX2_MDA_PATH = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..\\..\\tests\\test_data\\au\\MEX2\\MEX2_5640.mda",
        )
    )
    SXR_MDA_PATH = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..\\..\\tests\\test_data\\au\\SXR\\sxr129598.mda",
        )
    )
    from pyNexafs.parsers.au import MEX2_NEXAFS, SXR_NEXAFS

    # MEX2 example
    parser_mex2 = MEX2_NEXAFS(
        MEX2_MDA_PATH, relabel=True, energy_bin_domain=(3300, 3700)
    )
    scan_mex2 = parser_mex2.to_scan()

    # SXR Oxygen example
    parser_sxr = SXR_NEXAFS(SXR_MDA_PATH, relabel=True)
    scan_sxr = parser_sxr.to_scan()
    assert scan_sxr.y_labels[2] == "I0 VF", scan_sxr.y_labels[2]

    examples = [
        {
            "scan": scan_mex2,
            "norm_channel": "I0",
            "pre_edge_domain": (2460, 2465),
            "post_edge_domain": (2520, 2530),
            "idx": 1,
        },
        {
            "scan": scan_sxr,
            "norm_channel": scan_sxr.y_labels[2],
            "pre_edge_domain": (515, 525),
            "post_edge_domain": (570, 580),
            "idx": 0,
        },
    ]

    # plt.subplots(1,2, figsize=(10,5))
    # plt.plot(scan_sxr.x, scan_sxr.y[:,2], label="I0")
    # plt.show()

    for ex in examples[1:]:
        scan = ex["scan"]
        norm_channel = ex["norm_channel"]
        pre_edge_domain = ex["pre_edge_domain"]
        post_edge_domain = ex["post_edge_domain"]
        idx = ex["idx"]

        # Normalise to a channel
        norm = scan_normalised(scan, norm_channel=norm_channel)

        # Normalise to the edges
        edge_c = scan_normalised_edges(
            scan, pre_edge_domain=pre_edge_domain, post_edge_domain=post_edge_domain
        )
        edge_l = scan_normalised_edges(
            scan,
            pre_edge_domain=pre_edge_domain,
            post_edge_domain=post_edge_domain,
            pre_edge_normalisation=scan_normalised_edges.PREEDGE_NORM_TYPE.LINEAR,
        )

        edge_e = scan_normalised_edges(
            scan,
            pre_edge_domain=pre_edge_domain,
            post_edge_domain=post_edge_domain,
            pre_edge_normalisation=scan_normalised_edges.PREEDGE_NORM_TYPE.EXPONENTIAL,
        )

        # Plot each normalisation step
        fig, ax = plt.subplots(1, 3, sharex=True, figsize=(15, 5))
        ax[0].plot(
            scan.x,
            scan.y[:, idx],
            label=f"{type(scan_mex2).__name__}\n{scan.y_labels[idx]}",
        )

        l = ax[1].plot(
            norm.x, norm.y[:, idx], label=f"{type(norm).__name__}\n{norm.y_labels[idx]}"
        )
        ax[1].plot(
            scan.x,
            scan.y[:, idx],
            "--",
            label=f"{type(scan_mex2).__name__}\n{scan.y_labels[idx]}",
        )  # c=l[0].get_color()

        l = ax[2].plot(
            edge_c.x,
            edge_c.y[:, idx],
            label=f"{type(edge_c).__name__}\nConstant\n{edge_c.y_labels[idx]}",
        )
        l = ax[2].plot(
            edge_l.x,
            edge_l.y[:, idx],
            label=f"{type(edge_l).__name__}\nLinear\n{edge_l.y_labels[idx]}",
        )
        l = ax[2].plot(
            edge_e.x,
            edge_e.y[:, idx],
            label=f"{type(edge_e).__name__}\nExponential\n{edge_e.y_labels[idx]}",
        )
        ax[1].plot(
            edge_e.x,
            scan_normalised_edges.EXP_FN_OFFSET(
                edge_e.x, *edge_e.pre_edge_fit_params[3 * idx : 3 * (idx + 1)]
            ),
            "--",
            c=l[0].get_color(),
            label="Exponential Subtraction",
        )

        ax[2].set_xlabel("Energy (eV)")
        for a in ax:
            a.set_ylabel("Intensity (a.u.)")
            a.legend()

        ax[0].set_title("Original")
        ax[1].set_title("Normalised to I0")
        ax[2].set_title("Normalised to Edges")
        # ax[2].set_yscale("log")
        fig.tight_layout()
        plt.show()
