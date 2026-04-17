"""
This module contains the classes and functions to handle and process 1D NEXAFS scan data.

This includes treating the data, such as normalising it, and performing background subtraction.
"""

from pyNexafs.nexafs.scan import scanBase, scanAbstract, parsedScanAbstract, scanSimple

from pyNexafs.nexafs.normalisation.norm_settings import (
    configChannel,
    configExternalChannel,
    configEdges,
    configSeries,
    normMethod,
)
from pyNexafs.nexafs.normalisation.scan_normalised import (
    scanNorm,
    scanNormExt,
    scanDoubleNorm,
    scanNormEdges,
    scanEnergyNorm,
)

__all__ = [
    # Configuration classes
    "configChannel",
    "configExternalChannel",
    "configEdges",
    "configSeries",
    # Normalisation classes
    "normMethod",
    "scanNorm",
    "scanNormExt",
    "scanDoubleNorm",
    "scanNormEdges",
    # Scan data classes
    "scanBase",
    "scanAbstract",
    "parsedScanAbstract",
    "scanSimple",
    "scanEnergyNorm",
]
