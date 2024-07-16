"""
This module contains functions for reducing detector energy binning of raw NEXAFS data from beamlines.

For instance, as of Apr 2024, the MEX2 beamline at the Australian Synchrotron has 4 detectors,
each with 4096 energy bins (channels), all of which is recorded as the beam energy is scanned.
"""

import numpy as np
import numpy.typing as npt
import typing
import os
from pyNexafs.utils.mda import MDAFileReader
import matplotlib.pyplot as plt, matplotlib as mpl
from typing import Callable


class reducer:
    """
    A class for reducing detector energy binning of raw NEXAFS data from beamlines.

    Parameters
    ----------
    energies : array_like
        A 1D array of energy values for the scanning beam energy.
    dataset : array_like
        The NEXAFS dataset with dimensions (beam_energy, detection_energy_bin, detector_index).
        The detector_index dimension is optional, used for multiple multi-channel analysers.
    bin_energies : array_like | None, optional
        A 1D array of energy values for each detection energy bin. If not provided, the bin numbers are used.
        If 2D, the second dimension must match the number of detectors.
    """

    def __init__(
        self,
        energies: npt.NDArray,
        dataset: npt.NDArray,
        bin_energies: npt.NDArray | None = None,
    ) -> None:
        self.set_data(energies, dataset, bin_energies)

    def set_data(
        self,
        energies: npt.NDArray,
        dataset: npt.NDArray,
        bin_energies: npt.NDArray | None = None,
    ) -> None:
        """
        Set the energies, dataset and bin_energies attributes.

        Parameters
        ----------
        energies : npt.NDArray

        dataset : npt.NDArray
            _description_
        bin_energies : npt.NDArray | None, optional
            _description_, by default None
        """
        # Convert to numpy arrays
        energies = np.asarray(energies)
        dataset = np.asarray(dataset)
        if bin_energies is not None:
            bin_energies = np.asarray(bin_energies)

        # Check validity of input
        self._validify_inputs(energies, dataset, bin_energies)

        # Assign attributes
        self._energies: np.ndarray = energies
        self._dataset: np.ndarray = dataset
        self._bin_energies: np.ndarray = bin_energies

        # New boolean attribute to track if bin_energies has been set
        self._bins_assigned: bool
        """Tracks if bin_energies has been set."""

        if bin_energies is None:
            # Create bin numbers:
            bin_energies = np.linspace(
                0, self._dataset.shape[1], self._dataset.shape[1]
            )
            self._bins_assigned = False
        else:
            self._bins_assigned = True

    @property
    def energies(self) -> npt.NDArray:
        """
        The scanning beam energy values.

        Parameters
        ----------
        energies : npt.NDArray
            A 1D array of energy values for the scanning beam energy.
            Must have the same length as the first dimension of the dataset.

        Returns
        -------
        npt.NDArray
            The beam energy values.
        """
        return self._energies

    @energies.setter
    def energies(self, energies: npt.NDArray) -> None:
        # Raises an error if input is invalid.
        if reducer._validify_inputs(energies, self._dataset, self._bin_energies):
            self._energies = np.asarray(energies)

    @property
    def dataset(self) -> npt.NDArray:
        """
        The NEXAFS dataset with dimensions (beam_energy, detection_energy_bin, detector_index).

        Parameters
        ----------
        dataset : npt.NDArray
            The NEXAFS dataset indexed with dimensions (beam_energy, detection_energy_bin, detector_index).
            The detector_index dimension is optional, used for multiple multi-channel analysers.

        Returns
        -------
        npt.NDArray
            The NEXAFS dataset.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: npt.NDArray) -> None:
        # Raises an error if input is invalid.
        if reducer._validify_inputs(self._energies, dataset, self._bin_energies):
            self._dataset = np.asarray(dataset)

    @property
    def bin_energies(self) -> npt.NDArray:
        """
        The energy values (or bin indexes) for each detection energy bin.

        Parameters
        ----------
        bin_energies : npt.NDArray | None
            A 1D or 2D array of energy values for each detection energy bin.
            Must have the same length as the second dimension of the `dataset`.
            If None, generates bin numbers indexed to `dataset`.

        Returns
        -------
        npt.NDArray
            The energy values (or bin indexes) for each detection energy bin.
            Can be 1D or 2D, matching the latter dimensions of the dataset.
        """
        return self._bin_energies

    @bin_energies.setter
    def bin_energies(self, bin_energies: npt.NDArray | None) -> None:
        if bin_energies is None:
            self._bins_assigned = False
            self._bin_energies = np.linspace(
                0, self.dataset.shape[1], self.dataset.shape[1]
            )
        else:
            # Raises an error if input is invalid.
            if reducer._validify_inputs(self._energies, self._dataset, bin_energies):
                self._bin_energies = np.asarray(bin_energies)

    @property
    def has_bin_energies(self) -> bool:
        """
        A boolean property indicating whether the bin_energies attribute has been set.

        Important in determining binning behaviour.

        Returns
        -------
        bool
            True if bin_energies has been set, otherwise False.
        """
        return self._bins_assigned

    @property
    def detectors(self) -> int:
        """
        The number of detectors in the dataset.

        Returns
        -------
        int
            The number of detectors in the dataset.
        """
        if self._dataset.ndim == 2:
            return 1
        else:
            return self._dataset.shape[2]

    def domain_to_detector_bin_indexes(
        self, bin_domain: tuple[int, int] | tuple[float, float], detector_idx: int = 0
    ) -> tuple[int, int]:
        """
        Converts an energy|bin domain to bin indexes for a given detector.

        By default, the first detector is assumed.

        Parameters
        ----------
        bin_domain : tuple[int, int]
            The energy domain to convert to bin indexes.
        detector : int, optional
            Index of the detector to use, by default 0.

        Returns
        -------
        tuple[int, int]
            The lower and upper selected bin indexes for the given domain.
        """
        if not self.has_bin_energies:
            # If no labels, the range matches the indexes, convert to nearest integer
            return tuple(np.round(bin_domain).astype(int))
        else:
            if not isinstance(detector_idx, int):
                raise ValueError("`detector_idx` must be an integer.")
            # If labels and a given domain, convert the range to indexes
            if detector_idx >= self.detectors:
                raise ValueError(
                    f"Detector index {detector_idx} out of range [0-{self.detectors})."
                )
            else:
                if len(self.bin_energies.shape) == 1:
                    lb = np.argmin(np.abs(self.bin_energies - bin_domain[0]))
                    ub = np.argmin(np.abs(self.bin_energies - bin_domain[1]))
                else:
                    lb = np.argmin(
                        np.abs(self.bin_energies[:, detector_idx] - bin_domain[0])
                    )
                    ub = np.argmin(
                        np.abs(self.bin_energies[:, detector_idx] - bin_domain[1])
                    )
                return (lb, ub)

    def domain_to_indexes(
        self,
        bin_domain: (
            list[tuple[int, int] | tuple[float, float]]
            | tuple[int, int]
            | tuple[float, float]
        ),
    ) -> list[tuple[int, int]]:
        """
        Converts an energy|bin domain to bin indexes for all detectors.

        Parameters
        ----------
        bin_domain : tuple[int, int] | tuple[float, float]
            The energy|bin domain to convert to bin indexes.

        Returns
        -------
        list[tuple[int, int]]
            The lower and upper selected bin indexes for each detector.
        """
        if isinstance(bin_domain, list):
            return [
                self.domain_to_detector_bin_indexes(bin_domain=domain, detector_idx=i)
                for i, domain in enumerate(bin_domain)
            ]
        else:
            return [
                self.domain_to_detector_bin_indexes(
                    bin_domain=bin_domain, detector_idx=i
                )
                for i in range(self.detectors)
            ]

    def reduce_domain(
        self,
        bin_domain: (
            list[tuple[float, float] | tuple[int, int]]
            | tuple[float, float]
            | tuple[int, int]
            | None
        ) = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Reduce the dataset to a specified energy bin domain.

        Parameters
        ----------
        bin_domain : list[tuple[float, float] | tuple[int, int]] | tuple[float, float] | tuple[int, int] | None, optional
            The domain to reduce the dataset to. If None, the dataset is not reduced.
            If a single tuple is provided, the domain is applied to all detectors.
            If a list of tuples is provided, the domain is applied to each detector.
            The tuples should be in the form (min, max) for each detector.

        Returns
        -------
        reduced_bin_energies : npt.NDArray
            The reduced energy values for each detection energy bin.
            If bin_energies is not set, the values correspond bin indexes.
            Like `reduced_dataset`, `reduced_bin_energies` has as dimensions (N, D).
        reduced_dataset : npt.NDArray
            The reduced NEXAFS dataset, with dimensions (M, N, D),
            where M is the number of beam energies,
            N is the reduced number of detection energy bins according to the provided domain,
            and D is the number of detectors.
        """
        # Prepare the domain variable
        if bin_domain is None:
            # Return the full dataset window if no domain
            return self.bin_energies.copy(), self.dataset.copy()
        else:
            # Convert non-list to list
            if not isinstance(bin_domain, list):
                bin_domain = [bin_domain] * self.detectors
            # Check list validity
            if len(bin_domain) != self.detectors:
                raise ValueError(
                    f"`domain` must be singular i.e. `(min, max)`, or have the same length ({len(bin_domain)}) as"
                    + f" the number of detectors ({self.detectors}) i.e. `[(min1, max1), ..., (minN, maxN)]`."
                )

            # Reduce the dataset to the domain
            reduced_idxs = self.domain_to_indexes(bin_domain)

            if len(reduced_idxs) == 1 and self.dataset.ndim == 2:
                lb, ub = reduced_idxs[0]
                if self.has_bin_energies:
                    # Same logic for single or multiple detectors:
                    return self.bin_energies[lb:ub], self.dataset[:, lb:ub]
                else:
                    return np.arange(lb, ub), self.dataset[:, lb:ub]
            else:
                data = []
                bins = []
                for i, (lb, ub) in enumerate(reduced_idxs):
                    # Singular detector or not
                    if i == 1 and self.dataset.ndim == 2:
                        data.append(self.dataset[:, lb:ub])
                    else:
                        data.append(self.dataset[:, lb:ub, i])
                    if self.has_bin_energies:
                        if self.bin_energies.ndim == 1:
                            bins.append(self.bin_energies[lb:ub])
                        else:
                            bins.append(self.bin_energies[lb:ub, i])
                    else:
                        bins.append(np.arange(lb, ub))
                # Convert to arrays
                data = np.array(data)
                bins = np.array(bins)
                # Swap back the indexes of the datasets (currently detector, energies, bins):
                data = np.moveaxis(data, 0, -1)
                bins = np.moveaxis(bins, 0, -1) if bins.ndim == 2 else bins
                return bins, data

    def reduce(
        self,
        f: Callable[[npt.NDArray, npt.NDArray, npt.NDArray], npt.NDArray],
        bin_domain: (
            list[tuple[float, float] | tuple[int, int]]
            | tuple[float, float]
            | tuple[int, int]
            | None
        ) = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Reduces the bin dimension of the dataset to a singular value.

        Parameters
        ----------
        fn : Callable[[npt.NDArray, npt.NDArray, npt.NDArray], npt.NDArray]
            The function to apply to the reducer data.
            Parameters (in order) are energies, dataset, bin_energies.
            See examples for usage.

        domain : _type_, optional
            _description_, by default None

        Returns
        -------
        bin_energies : npt.NDArray
            An array of the bin energies within the bin domain.
            If bin_energies is not set, the values correspond bin indexes.
            Can be 1D if one detector, or 2D for multiple detectors.
        reduced_dataset : npt.NDArray
            The reduced NEXAFS dataset.
            Shape depends on the return of `fn`.

        Examples
        --------
        # Example 1: Sum the dataset over the bin and detector dimensions
        def sum_dataset(energies, dataset, bin_energies):
            return dataset.sum(axis=(1, 2))
        dataset = reducer.reduce_bins_to_singular(sum_dataset)

        # Example 2: Translate the dataset to be positive definite,
        # and add 0.01 so that the log is not undefined.
        def positive_definite(energies, dataset, bin_energies):
            return np.log(dataset - dataset.min() + 1e-2)

        """
        energies = self.energies
        if bin_domain is not None:
            bin_energies, dataset = self.reduce_domain(bin_domain)
        else:
            dataset, bin_energies = self.dataset.copy(), self.bin_energies.copy()
        # Apply the function to the sub_domain dataset
        reduced_dset = f(energies, dataset, bin_energies)
        # Return the reduced dataset. Don't need to return bin_energies as
        return bin_energies, reduced_dset

    def reduce_by_sum(
        self,
        bin_domain: (
            list[tuple[float, float] | tuple[int, int]]
            | tuple[float, float]
            | tuple[int, int]
            | None
        ) = None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Reduces the dataset into a single value for each beam energy by summing over dimensions.

        Parameters
        ----------
        domain : list[tuple[float, float] | tuple[int, int]] | tuple[float, float] | tuple[int, int] | None, optional
            The domain to reduce the dataset to. If None, the dataset is not reduced.
            If a single tuple is provided, the domain is applied to all detectors.
            If a list of tuples is provided, the domain is applied to each detector.
            The tuples should be in the form (min, max) for each detector.

        Returns
        -------
        reduced_bin_energies : npt.NDArray
            The reduced energy values for each detection energy bin.
            If bin_energies is not set, the values correspond bin indexes.
            Like `reduced_dataset`, `reduced_bin_energies` has as dimensions (N, D).
        reduced_dataset : npt.NDArray
            The reduced NEXAFS dataset, with dimensions (M, N, D),
            where M is the number of beam energies,
            N is the reduced number of detection energy bins according to the provided domain,
            and D is the number of detectors.
        """
        f = lambda energies, dataset, bin_energies: dataset.sum(axis=(1, 2))
        return self.reduce(f, bin_domain)

    def reduce_to_bin_features(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Reduces the dataset to find the signal dependence of energy bins with signal.

        The return signal has
        1. Summed over all beam energies.
        2. Translated to be positive definite (a base 1e-2 is added so we can log plot).
        the sum of the dataset over all beam energies.

        Returns
        -------
        bin_energies : npt.NDArray
            The energy values for each detection energy bin.
            If bin_energies is not set, the values correspond bin indexes.
        dataset_sum : npt.NDArray
            The reduced dataset, with dimensions (N, D),
            where N is the number of detection energy bins with signal,
            and D is the number of detectors.
        """
        # Sum over all energies to find the energy bins with signal.
        ds_sum: np.ndarray = self.dataset.sum(axis=0)
        # Translate every bin to be positive definite if data values < 1e-2,
        # This also allows log plotting.
        if np.any(ds_sum < 1e-2):
            # Keep relative signal amplitude between detectors
            ds_sum += 1e-2 - ds_sum.min()
        return self.bin_energies, ds_sum

    @staticmethod
    def _validify_inputs(
        energies: npt.NDArray,
        dataset: npt.NDArray,
        bin_energies: npt.NDArray | None = None,
    ) -> bool:
        """
        Validates the input dataset, energies and bin_energies.

        Raises a ValueError if the inputs are invalid, otherwise returns True.

        Parameters
        ----------
        dataset : npt.NDArray
            The NEXAFS dataset with dimensions (beam_energy, detection_energy_bin, detector_index).
            The detector_index dimension is optional, used for multiple multi-channel analysers.
        energies : npt.NDArray
            A 1D array of energy values for the scanning beam energy.
        bin_energies : npt.NDArray, optional
            A 1D array of energy values for each detection energy bin. If not provided, the bin numbers are used.
            If 2D, the second dimension must match the number of detectors.

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]
            The validated dataset, energies and bin_energies.
        """
        if not energies.ndim == 1:
            raise ValueError("`energies` must be a 1D numpy array.")
        if not (dataset.ndim == 2 or dataset.ndim == 3):
            raise ValueError(
                "`dataset` must be a 2D or 3D numpy array with indexes corresponding to"
                + " (beam_energy, detection_energy_bin, detector_index). The detector_index dimension is optional."
            )
        if not (energies.shape[0] == dataset.shape[0]):
            raise ValueError(
                f"`energies` must have the same length ({energies.shape[0]})"
                + f" as the first dimension of `dataset` ({dataset.shape[0]})."
            )
        if dataset.ndim == 2:
            dataset = dataset[:, :, np.newaxis]  # add final dimension.

        # Check bin_energies validity
        if bin_energies is not None:
            bin_energies = np.asarray(bin_energies)
            if not (bin_energies.ndim == 1 or bin_energies.ndim == 2):
                raise ValueError(
                    "`bin_energies` must be a 1D or 2D numpy array, with indexes corresponding to"
                    + " (detection_bin_energy, detector_index). The second dimension is optional."
                )
            if not (bin_energies.shape[0] == dataset.shape[1]):
                raise ValueError(
                    f"`bin_energies` must have the same length ({bin_energies.shape[0]}) "
                    + f"as the second dimension of `dataset` ({dataset.shape[1]})."
                )
            if bin_energies.ndim == 2 and not (
                bin_energies.shape[1] == dataset.shape[2]
            ):
                raise ValueError(
                    f"`bin_energies` must have the same length ({bin_energies.shape[1]}) "
                    + f"as the third dimension of `dataset` ({dataset.shape[2]})."
                )
        return True


if __name__ == "__main__":
    ###
    ### This is a demo to show how to use the reducer class.
    ###

    path = os.path.dirname(__file__)
    package_path = os.path.normpath(os.path.join(path, "../../"))
    mda_path = os.path.normpath(
        os.path.join(package_path, "tests/test_data/au/MEX2/MEX2_5643.mda")
    )
    mda_data, mda_scans = MDAFileReader(mda_path).read_scans()
    data1D, data2D = mda_data

    # Convert MEX2 Data
    energies = data1D[:, 0] * 1000  # Convert keV to eV
    span = [96, 500]
    binned_data = data2D[:, span[0] : span[1], :]
    TOTAL_BINS = span[1] - span[0]
    BIN_ENERGY = 11.935
    BIN_96_ENERGY = 1146.7
    bin_energies = np.linspace(
        BIN_96_ENERGY, BIN_96_ENERGY + TOTAL_BINS * BIN_ENERGY, TOTAL_BINS
    )

    # Setup a graph
    graph = plt.subplots(1, 3, figsize=(12, 6))
    fig: plt.Figure = graph[0]
    ax: list[plt.Axes] = graph[1]

    # Manually reduce
    reduced_data = binned_data.sum(axis=1)  # Sum over bins
    [
        ax[0].plot(energies, reduced_data[:, i], label=f"Detector {i}")
        for i in range(reduced_data.shape[1])
    ]
    ax[0].set_title("Raw data, summed over all bin energies")
    ax[0].set_xlabel("Beam Energy (eV)")
    ax[0].set_ylabel("Intensity")
    ax[0].legend()

    def see_bin_features(energies, dataset, bin_energies):
        # Subtract average of each detector along the bin axis for each energy (this removes constant background)
        # ds_sub: np.ndarray = (
        #         dataset[:, :, :] - np.mean(dataset[:, :, :], axis=1)[:, np.newaxis, :]
        # )
        ds_sub = dataset
        # Sum over all energies to find the energy bins with signal.
        ds_sum: np.ndarray = ds_sub.sum(axis=0)
        # Translate every bin to be positive definite, and add a base 1e-2 so we can log plot
        if np.any(ds_sum <= 1e-2):
            if dataset.ndim == 2:  # Single detector
                ds_sum += 1e-2 - ds_sum.min()
            else:  # Multiple detectors
                ds_sum += 1e-2 - ds_sum.min(axis=0)
        return ds_sum

    logdata = see_bin_features(energies, binned_data, bin_energies)
    [
        ax[1].plot(bin_energies, logdata[:, i], label=f"Detector {i}")
        for i in range(reduced_data.shape[1])
    ]
    ax[1].set_title("Raw data, summed over all beam energies")
    ax[1].set_yscale("log")
    # ax[1].set_ylim(1e-2, logdata.max()*5)
    ax[1].set_xlabel("Bin Energy (eV)")
    ax[1].set_ylabel("Intensity")
    ax[1].legend()

    # Now use the reducer
    red = reducer(energies, binned_data, bin_energies)
    # Demonstrate bin domain use
    domain = (3300, 3700)
    reduced_bin_energies, reduced_dset = red.reduce_domain(domain)
    [
        ax[2].plot(
            energies, np.sum(reduced_dset[:, :, i], axis=1), label=f"Detector {i}"
        )
        for i in range(red.detectors)
    ]
    ax[2].set_title("Reduced NEXAFS data")
    ax[2].set_xlabel("Beam Energy (eV)")
    ax[2].set_ylabel("Intensity")
    # Demonstrate bin reduction
    reduced_bin_energies, reduced_dset = red.reduce_by_sum(bin_domain=domain)
    ax[2].plot(energies, reduced_dset, label="Reduced")
    ax[2].legend()

    fig.tight_layout()
    plt.show()
