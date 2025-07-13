"""
This module contains the classes and functions to handle and process 1D NEXAFS scan data.

This includes treating the data, such as normalising it, and performing background subtraction.
"""

from pyNexafs.nexafs.scan import scanBase, scanAbstract, scanSimple

from pyNexafs.nexafs.normalisation import (
    configChannel,
    configExternalChannel,
    configSeries,
    normConfigEdges,
    scanNorm,
    scanNormExt,
    scanNormBackgroundChannel,
    scanNormEdges,
)

# TODO: Fix
# from pyNexafs.nexafs.normalisation import (
#     scan_norm,
#     scan_normalised_edges,
#     scan_background_subtraction,
#     scan_normalised_background_channel,
# )
