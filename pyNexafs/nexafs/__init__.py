"""
This module contains the classes and functions to handle and process 1D NEXAFS scan data.

This includes treating the data, such as normalising it, and performing background subtraction.
"""

from pyNexafs.nexafs.scan import scan_base, scan_abstract, scan_simple
from pyNexafs.nexafs.scan_normalised import (
    scan_normalised,
    scan_normalised_edges,
    scan_background_subtraction,
    scan_normalised_background_channel,
)
