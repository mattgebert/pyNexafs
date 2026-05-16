"""
A module for defining detector specifications for the MEX 1 & 2 beamlines.

Defines classes for fluorescence detectors, which are implemented
with multi-channel analysers (MCA) that bin the fluorescence signal into energy bins.
"""

import numpy as np
from numpy.typing import NDArray


class FluorescenceDetector:
    """
    An abstract class to reduce energy-binned fluorescence.

    Reduces multi-channel analyser (MCA) binned fluorescence data.
    """

    FLUOR_NAMES: list[str]
    """The string names of the 2D fluorescence binned channels (length N)."""
    BIN_ENERGY_DELTA: float
    """The energy difference (eV) between each bin."""
    TOTAL_BINS: int
    """The total number of energy bins in the channel data (length M)."""
    TOTAL_BIN_ENERGIES: NDArray
    """
    The energy (eV) of each channel bin.

    Should match the dimensions (M, N) where M is the number of bins, N is the number of channels.
    """
    BIN_ENERGY_OFFSET: float
    """The offset energy (eV) of the first energy bin."""

    INTERESTING_BIN_IDX: tuple[int, int] | None = None
    """
    Indexes of interesting energy bins from the detector.

    Can be used to hide energy bins with zero signal.
    """

    @classmethod
    def INTERESTING_BIN_ENERGIES(cls) -> NDArray:
        """
        The energies corresponding to the bins between the `INTERESTING_BIN_IDX` indices.

        Returns
        -------
        NDArray
            An array of the energies between the `INTERESTING_BIN_IDX` indices.

        Raises
        ------
        AttributeError
            When the class attribute `INTERESTING_BIN_IDX` doesn't have a defined tuple of indices.
        """
        if cls.INTERESTING_BIN_IDX is not None:
            return cls.TOTAL_BIN_ENERGIES[
                cls.INTERESTING_BIN_IDX[0] : cls.INTERESTING_BIN_IDX[1]
            ]
        else:
            raise AttributeError(
                f"`INTERESTING_BIN_ENERGIES` is not defined on the {cls.__name__} detector."
            )


class DanteFluorescence(FluorescenceDetector):
    """Configuration for interpreting the Dante MCA Fluorescence data."""

    FLUOR_NAMES: list[str] = [
        "MEX2ES01DPP01:ch1:W:ArrayData",
        "MEX2ES01DPP01:ch2:W:ArrayData",
        "MEX2ES01DPP01:ch3:W:ArrayData",
        "MEX2ES01DPP01:ch4:W:ArrayData",
    ]
    """The fluorescence channel names for the Dante MCA detector."""
    BIN_ENERGY_DELTA = 11.935
    """The energy difference (eV) between each bin for the Dante MCA detector."""
    BIN_ENERGY_OFFSET = -1146.7  # Bin 96 corresponds to -0.94 eV.
    """
    The offset energy (eV) of the first energy bin for the Dante MCA detector.

    Bin 96 corresponds to -0.94 eV, which is approximately the carbon edge.
    """
    TOTAL_BINS = 4096
    """The total number of energy bins in the Dante MCA detector."""
    TOTAL_BIN_ENERGIES = np.linspace(
        start=BIN_ENERGY_OFFSET,
        stop=BIN_ENERGY_OFFSET + TOTAL_BINS * BIN_ENERGY_DELTA,
        num=TOTAL_BINS,
    )
    """The energy (eV) of each channel bin for the Dante MCA detector."""
    INTERESTING_BIN_IDX = (80, 900)
    """The indexes of bins with non-zero signal for the Dante MCA detector."""


class Xpress3Fluorescence(FluorescenceDetector):
    """Configuration for interpreting the Xpress3 MCA Fluorescence data."""

    FLUOR_NAMES = [
        "MEX2ES01DPP02:MCA1:ArrayData",  # New MCA
        "MEX2ES01DPP02:MCA2:ArrayData",
        "MEX2ES01DPP02:MCA3:ArrayData",
        "MEX2ES01DPP02:MCA4:ArrayData",
    ]
    """The fluorescence channel names for the Xpress3 MCA detector."""
    BIN_ENERGY_DELTA = 10
    """The energy difference (eV) between each bin for the Xpress3 MCA detector."""
    BIN_ENERGY_OFFSET = 0
    """The offset energy (eV) of the first energy bin for the Xpress3 MCA detector."""
    TOTAL_BINS = 4096
    """The total number of energy bins in the Xpress3 MCA detector."""
    TOTAL_BIN_ENERGIES = np.linspace(
        start=BIN_ENERGY_OFFSET,
        stop=BIN_ENERGY_OFFSET + TOTAL_BINS * BIN_ENERGY_DELTA,
        num=TOTAL_BINS,
    )
    """The energy (eV) of each channel bin for the Xpress3 MCA detector."""
    INTERESTING_BIN_IDX = (0, 800)
    """The indexes of bins with non-zero signal for the Xpress3 MCA detector."""
