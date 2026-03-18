"""
A module for all normalisation and background subtraction operations on the data.

Normalisation operations are designed to be repeatable and tracable, such that
the order of operations is clear and the normalisation can be reproduced between
different datasets.
"""

# Config Classes
from pyNexafs.nexafs.normalisation.norm_settings import (
    configBase,
    configSeries,
    configChannel,
    configExternalChannel,
    configEdges,
)

# Config Enumerates
from pyNexafs.nexafs.normalisation.norm_settings import (
    normMethod,
    extSelection,
)

#
from pyNexafs.nexafs.normalisation.scan_normalised import (
    scanNorm,
    scanNormExt,
    scanDoubleNorm,
    scanNormEdges,
)


# from pyNexafs.nexafs.normalisation.scan_normalised import (
#     scan_norm,
#     scan_normalised_edges,
#     # scan_background_subtraction,
#     scan_normalised_background_channel,
# )

__all__ = [
    # Config Classes
    "configBase",
    "configSeries",
    "configChannel",
    "configExternalChannel",
    "configEdges",
    # Enumerates
    "normMethod",
    "extSelection",
    # Normalisation Classes
    "scanNorm",
    "scanNormExt",
    "scanDoubleNorm",
    "scanNormEdges",
]
