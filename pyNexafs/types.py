"""
Definitions of NEXAFS data types, to allow attribute access on nexafs `scan` objects.
"""

from enum import StrEnum
from typing import Any, TypedDict
from typing_extensions import NotRequired
import numpy.typing as npt
from pyNexafs.utils.decorators import enum_member_doc


@enum_member_doc
class dtype(StrEnum):
    """
    An enumerate for the different NEXAFS data sources.
    """

    ## The direct absorption of X-ray intensity:
    T = "Transmission"  # Equivalent to absorption.
    """Transmission, where the intensity of X-rays transmitted through the sample is measured."""

    A = "Absorption"  # Equivalent to transmission.
    """Absorption, where the absorption of X-rays by the sample is measured."""

    ## Partial photo-electron yield
    PEY = "Partial Electron Yield"
    """Partial Electron Yield, where electrons are collected within a partial energy/momentum window."""

    # The current from a sample drain:
    TEY = "Total Electron Yield"
    """Total Electron Yield, where the total current from the sample drain is measured."""

    # Auger electron yield:
    AEY = "Auger Electron Yield"
    """Auger Electron Yield, where Auger electrons are collected."""

    # Fluorescence yield:
    TFY = "Total Fluorescence Yield"
    """Total Fluorescence Yield, where the total fluorescence from the sample is measured."""

    # Fluorescence yield from a partial energy window:
    PFY = "Partial Fluorescence Yield"
    """Partial Fluorescence Yield, where the fluorescence from a partial energy window is measured."""

    # Normalisation channels:
    I0 = "I0"  # Incident X-ray intensity
    """I0, where the incident X-ray intensity is measured typically via a mesh current, often used for normalisation."""

    # Photodiode scan without sample
    PHD = "Photodiode"
    """
    Photodiode, where the X-ray intensity is measured by a photodiode without the sample,
    often used for normalisation at the carbon edge.

    For general absorption/transmission through a film, use dtype.A or dtype.T.
    """

    # Reference foil scan
    REF = "Reference Foil"
    """A measurement of a reference foil, used for calibrating the energy axis."""

    # Energy channels:
    E = "Energy"
    """Energy, the measured x-ray photon energy in eV."""

    E_SET = "Energy Setpoint"
    """Energy, the setpoint of the x-ray photon energy in eV."""


# Additionally add aliases for the datatypes that are commonly used in NEXAFS
T = dtype.T
"""Alias of :attr:`dtype.T` (Transmission)."""

A = dtype.A
"""Alias of :attr:`dtype.A` (Absorption)."""

TEY = dtype.TEY
"""Alias of :attr:`dtype.TEY` (Total Electron Yield)."""

PEY = dtype.PEY
"""Alias of :attr:`dtype.PEY` (Partial Electron Yield)."""

AEY = dtype.AEY
"""Alias of :attr:`dtype.AEY` (Auger Electron Yield)."""

TFY = dtype.TFY
"""Alias of :attr:`dtype.TFY` (Total Fluorescence Yield)."""

PFY = dtype.PFY
"""Alias of :attr:`dtype.PFY` (Partial Fluorescence Yield)."""

E = dtype.E
"""Alias of :attr:`dtype.E` (Energy)."""

E_SET = dtype.E_SET
"""Alias of :attr:`dtype.E_SET` (Energy Setpoint)."""

PHD = dtype.PHD
"""Alias of :attr:`dtype.PHD` (Photodiode)."""

REF = dtype.REF
"""Alias of :attr:`dtype.REF` (Reference Foil)."""

I0 = dtype.I0
"""Alias of :attr:`dtype.I0` (Incident X-ray intensity)."""

ALL_DTYPE_MEMBERS: list[dtype] = [T, A, TEY, PEY, AEY, TFY, PFY, E, E_SET, PHD, REF, I0]
"""
A list of all the available dtype members, for easy reference and validation.
"""


# Define the column assignments dictionary typing key-value pairs
class assignments_type(TypedDict):
    """
    Required assignments types for the COLUMN_ASSIGNMENTS parser class property.

    Allows for the key assignments of
    - 'x': A single string or tuple of synonymous strings, identifying the independent variable.
    - 'y': A single string, a tuple of strings, or a list of the former two, identifying dependent variables.
    - 'x_errs': A single string or None, identifying the independent variable error.
    - 'y_errs': A single string, a tuple of strings, or a list of the former two, identifying dependent variable errors.
                If provided, must match the structure of 'y'.
    """

    x: str | tuple[str, ...]
    y: str | tuple[str, ...] | list[str | tuple[str, ...]]
    x_errs: NotRequired[str | tuple[str, ...] | None]
    y_errs: NotRequired[
        str | tuple[str, ...] | None | list[str | tuple[str, ...] | None]
    ]


parse_fn_ret_type = tuple[
    npt.NDArray | tuple[npt.NDArray, ...] | None,  # Data
    list[str | None]
    | list[str]
    | tuple[list[str | None] | list[str], ...]
    | None,  # Labels
    list[str | None]
    | list[str]
    | tuple[list[str | None] | list[str], ...]
    | None,  # Units
    dict[str, Any],  # Params
]
"""
The return type of a parser function.

A tuple that is comprised of:
- data: npt.NDArray | tuple[npt.NDArray, ...] | None
    The data array, or a tuple of data arrays, or None if no data is returned.
- labels: list[str | None] | list[str] | tuple[list[str | None] | list[str], ...] | None
    The labels of the data array(s), or None if no data is returned.
- units: list[str | None] | list[str] | tuple[list[str | None] | list[str], ...] | None
    The units of the data array(s), or None if no data is returned.
- params: dict[str, Any]
    A dictionary of parameters extracted from the file, or an empty dictionary if no parameters are returned.
"""

reduction_type = tuple[
    npt.NDArray | None, list[str | None] | None, list[str | None] | None
]
"""
The return type of a parser reduction function.

A tuple that is comprised of:
- data: npt.NDArray | None
    The reduced data array in a 1D (singular)/2D (multiple) format, or None if no data is returned.
    First index must always correspond to energy, and the second index is the channel indexes (if multiple).
- labels: list[str | None] | None
    The labels of the reduced data array, or None if no data is returned.
- units: list[str | None] | None
    The units of the reduced data array, or None if no data is returned.
"""
