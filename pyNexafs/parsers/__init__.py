"""
Parser classes for NEXAFS data, including support for synchrotron specific beamlines.

Parsers are registered in the parser_loaders dictionary, which is used to register parsers
in GUI applications.
"""

# Base objects
from pyNexafs.parsers._base import parser_base, parser_meta

# Specific Parsers
from pyNexafs.parsers import au

# Define loaders with string representation.
parser_loaders = {
    "au SXR:NEXAFS": au.SXR_NEXAFS,
    "au MEX1:NEXAFS": au.MEX1_NEXAFS,
    "au MEX2:NEXAFS": au.MEX2_NEXAFS,
}

# Check that all parsers are subclasses of the base parser.
for parser in parser_loaders.values():
    assert issubclass(
        parser, parser_base
    ), f"{parser} is not a subclass of parser_base."

# Check that no parser names overlap. Important for GUIs.
parser_names = [parser_name for parser_name in parser_loaders.keys()]
assert len(parser_names) == len(set(parser_names)), "Parser names overlap."
