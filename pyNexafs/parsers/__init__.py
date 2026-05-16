"""
Parser classes for NEXAFS data, including support for synchrotron specific beamlines.

Parsers are registered in the `parser_registry` dictionary, which is used to register parsers
in GUI applications.
"""

# Base objects
from pyNexafs.parsers.base import parserBase, parserMeta

# Specific Parsers
from pyNexafs.parsers import au
from pyNexafs.parsers.au import SXR_NEXAFS, MEX1_NEXAFS, MEX2_NEXAFS

# Define registry with string representation.
parser_registry: dict[str, type[parserBase]] = {
    "au SXR:NEXAFS": au.SXR_NEXAFS,
    "au MEX1:NEXAFS": au.MEX1_NEXAFS,
    "au MEX2:NEXAFS": au.MEX2_NEXAFS,
}
"""A dictionary of parser loaders, which are used to register parsers in GUI applications."""

# Check that all parsers are subclasses of the base parser.
for parser in parser_registry.values():
    assert issubclass(parser, parserBase), f"{parser} is not a subclass of parserBase."

# Check that no parser names overlap. Important for GUIs.
parser_names: list[str] = list(parser_registry.keys())
"""A list of parser names available in the `parser_registry` dictionary."""

assert len(parser_names) == len(set(parser_names)), "Parser names overlap."

__all__ = [
    "parserBase",
    "parserMeta",
    "parser_registry",
    "au",
    "SXR_NEXAFS",
    "MEX1_NEXAFS",
    "MEX2_NEXAFS",
]
