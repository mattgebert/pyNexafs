"""
A module for all normalisation and background subtraction operations on the data.

Normalisation operations are designed to be repeatable and tracable, such that
the order of operations is clear and the normalisation can be reproduced between
different datasets.
"""

from pyNexafs.nexafs.normalisation.norm_settings import (
    configBase,
    configSeries,
    configChannel,
    configExternalChannel,
    normConfigEdges,
)

from pyNexafs.nexafs.normalisation.scan_normalised import (
    scanNorm,
    scanNormExt,
    scanNormBackgroundChannel,
    scanNormEdges,
)


# from pyNexafs.nexafs.normalisation.scan_normalised import (
#     scan_norm,
#     scan_normalised_edges,
#     # scan_background_subtraction,
#     scan_normalised_background_channel,
# )
