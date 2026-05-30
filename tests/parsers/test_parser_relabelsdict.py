"""
Test the relabels dictionary in the parserMeta class.

Allows for flexible relabeling of parameters, including the use of tuples to group
synonymous keys together. The relabels dictionary should act as a regular dictionary,
but maintains elements in tuples as synonamous, and requires the `del` operator to remove keys, to ensure the integrity of the tuple relationships.
"""

import pytest
from pyNexafs.parsers import parserMeta
from pyNexafs import dtype


class TestRelabelsDict:
    """Tests the synonymous tuple functionality of the relabels dictionary."""

    class DummyParserRelabel:
        RELABELS = {"a": "b", "c": "d", ("e", "f"): "g", "i": dtype.PEY}

    @pytest.mark.parametrize(
        "key, value",
        [
            ("a", "b"),
            ("c", "d"),
            ("e", "g"),  # Individual keys from a tuple
            ("f", "g"),
            (("e", "f"), "g"),  # The tuple itself
            ("g", "g"),  # The value itself (no relabeling)
            (dtype.PEY, dtype.PEY),
            (dtype.PEY.value, "Partial Electron Yield"),
            ("i", "Partial Electron Yield"),
            ("Partial Electron Yield", "Partial Electron Yield"),
        ],
    )
    def test_get(self, key, value):
        """Tests the relabels dictionary can be set and returns the correct value,
        including the use of tuples."""

        # Create the relabel dictionary.
        relabel_dict = parserMeta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test its access
        assert relabel_dict[key] == value

    @pytest.mark.parametrize(
        "key, value",
        [
            ("a", "1"),
            ("c", "2"),
            ("e", "3"),
            ("f", "4"),
            (("e", "f"), "5"),
            ("g", "6"),
            (dtype.PEY, "Partial Electron Yield"),
            ("Partial Electron Yield", "Partial Electron Yield"),
            ("i", "Partial Electron Yield"),
        ],
    )
    def test_set(self, key, value):
        # Create the relabel dictionary.
        print("Type:", type(value))
        relabel_dict = parserMeta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test setting
        relabel_dict[key] = value
        # Check it was set
        assert relabel_dict[key] == value

    @pytest.mark.parametrize(
        "key, contained",
        [
            ("a", True),
            ("b", True),
            ("c", True),
            ("d", True),
            ("e", True),
            ("f", True),
            (("e", "f"), True),
            ("g", True),
            ("h", False),
            (("e", "g"), False),
        ],
    )
    def test_contains(self, key, contained):
        """Tests the relabels dictionary can be set and returns the correct value,
        including the use of tuples."""

        # Create the relabel dictionary.
        relabel_dict = parserMeta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test its access
        assert (key in relabel_dict) == contained

    def test_tuples(self):
        """Tests the synonymous tuple functionality of the relabels dictionary."""
        relabel_dict = parserMeta.relabels_dict(self.DummyParserRelabel.RELABELS)
        assert relabel_dict[("e", "f")] == "g"  # type: ignore ##pylance doesnt like this.
        assert relabel_dict["e"] == "g"
        # Test propogation
        relabel_dict["f"] = "h"  # Should also update `e` and `("e", "f")`
        assert relabel_dict["e"] == "h"
        assert relabel_dict[("e", "f")] == "h"  # type: ignore
