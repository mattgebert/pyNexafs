""" "Tests the scan classes and their methods"""

import pytest

# Test the creation of a new parser class.
from pyNexafs.nexafs.scan import scanAbstract, scanBase, scanSimple

##############################################################################
################### Test the inner classes of parser_meta ####################
##############################################################################


class TestScanAbstract:
    """Tests the scan abstract class."""
