"""
Definitions of NEXAFS data types, to allow attribute access on nexafs `scan` objects.
"""

from enum import StrEnum


class dtype(StrEnum):
    """
    An enumerate for the different NEXAFS data sources.
    """

    # The direct absorption of X-ray intensity:
    T = "Transmission"  # Equivalent to absorption.
    # Photo-electron yield
    PEY = "Partial Electron Yield"
    # The current from a sample drain:
    TEY = "Total Electron Yield"
    # Auger electron yield:
    AEY = "Auger Electron Yield"
    # Fluorescence yield:
    TFY = "Total Fluorescence Yield"
    # Fluorescence yield from a partial energy window:
    PFY = "Partial Fluorescence Yield"

    # Normalisation channels:
    I0 = "I0"  # Incident X-ray intensity


if __name__ == "__main__":
    print("Done")
