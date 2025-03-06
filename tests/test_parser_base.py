""""Tests the base parser classes and their methods"""

import pytest

# Test the creation of a new parser class.
from pyNexafs.parsers._base import parser_meta, parser_base

##############################################################################
################### Test the inner classes of parser_meta ####################
##############################################################################


class TestRelabelsDict:
    """Tests the synonymous tuple functionality of the relabels dictionary."""

    class DummyParserRelabel:
        RELABELS = {"a": "b", "c": "d", ("e", "f"): "g"}

    @pytest.mark.parametrize(
        "key, value",
        [("a", "b"), ("c", "d"), ("e", "g"), ("f", "g"), (("e", "f"), "g")],
    )
    def test_get(self, key, value):
        """Tests the relabels dictionary can be set and returns the correct value,
        including the use of tuples."""

        # Create the relabel dictionary.
        relabel_dict = parser_meta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test its access
        assert relabel_dict[key] == value

    @pytest.mark.parametrize(
        "key, value",
        [("a", "1"), ("c", "2"), ("e", "3"), ("f", "4"), (("e", "f"), "5")],
    )
    def test_set(self, key, value):

        # Create the relabel dictionary.
        relabel_dict = parser_meta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test setting
        relabel_dict[key] = value
        assert relabel_dict[key] == value

    @pytest.mark.parametrize(
        "key, contained",
        [
            ("a", True),
            ("c", True),
            ("e", True),
            ("f", True),
            (("e", "f"), True),
            ("g", False),
            ("h", False),
            (("e", "g"), False),
        ],
    )
    def test_contains(self, key, contained):
        """Tests the relabels dictionary can be set and returns the correct value,
        including the use of tuples."""

        # Create the relabel dictionary.
        relabel_dict = parser_meta.relabels_dict(self.DummyParserRelabel.RELABELS)
        # Test its access
        assert (key in relabel_dict) == contained

    def test_tuples(self):
        relabel_dict = parser_meta.relabels_dict(self.DummyParserRelabel.RELABELS)
        assert relabel_dict[("e", "f")] == "g"  # type: ignore ##pylance doesnt like this.
        assert relabel_dict["e"] == "g"
        # Test propogation
        relabel_dict["f"] = "h"  # Should also update `e` and `("e", "f")`
        assert relabel_dict["e"] == "h"
        assert relabel_dict[("e", "f")] == "h"  # type: ignore


class TestSummaryParamList:
    """Tests the summary parameters list functionality."""

    @pytest.mark.parametrize(
        "elements, query, contains",
        [
            (["a", "b", "c"], "a", True),
            ([1, 2, 3], 1, True),
            (["a", ("b", "c"), "d"], "b", True),
            (["a", ("b", "c"), "d"], ("b", "c"), True),
            (["a", "b", "c"], "d", False),
        ],
    )
    def test_contains(self, elements, query, contains):
        """Tests the summary parameters list can be set and returns the correct value."""

        # Create the summary parameters list.
        summary_params = parser_meta.summary_param_list(elements)
        # Test its access
        assert (query in summary_params) == contains

    @pytest.mark.parametrize(
        "elements, query, index_or_exception",
        [
            (["a", "b", "c"], "a", 0),
            ([1, 2, 3], 1, 0),
            (["a", ("b", "c"), "d"], "b", 1),
            (["a", ("b", "c"), "d"], ("b", "c"), 1),
            (["a", "b", "c"], "e", ValueError),
        ],
    )
    def test_index(self, elements, query, index_or_exception):
        """Tests the summary parameters list can be indexed, and tuples behave as individual elements also."""
        # Create the summary parameters list.
        summary_params = parser_meta.summary_param_list(elements)
        # Test its access
        if isinstance(index_or_exception, type) and issubclass(
            index_or_exception, Exception
        ):
            with pytest.raises(
                index_or_exception,
                match=f"{query} is not in the list, or within a tuple element.",
            ) as e:
                summary_params.index(query)
        elif isinstance(index_or_exception, int):
            assert summary_params.index(query) == index_or_exception
        else:
            raise ValueError("Invalid test case.")


##############################################################################
############################ Test parser_meta ################################
##############################################################################


class TestParserMeta:
    """Tests the metaclass functionality on creating new Parser classes."""

    def test_missing_ALLOWED_EXTENSIONS(self):
        """Tests ValueError is raised when ALLOWED_EXTENSIONS is not defined."""
        try:

            class Test_Parser(parser_base):
                pass

        except ValueError as e:
            assert "Class Test_Parser does not define ALLOWED_EXTENSIONS." in str(e)

    def test_missing_COLUMN_ASSIGNMENTS(self):
        """Tests ValueError is raised when COLUMN_ASSIGNMENTS is not defined."""
        try:

            class Test_Parser(parser_base):
                ALLOWED_EXTENSIONS = [".txt"]
                pass

        except ValueError as e:
            assert "Class Test_Parser does not define COLUMN_ASSIGNMENTS." in str(e)


##############################################################################
#################### Test the inner classes of parser_base ###################
##############################################################################


class TestParamDict:
    """Tests the parameter dictionary functionality, which stores a set of parameters,
    but allows relabelled indexing and contains."""

    class DummyLoadedParser:
        RELABELS = {"a": "b", "c": "d", ("e", "f"): "g"}
        params = {"a": 1, "c": 3, "f": 6, "g": 8}

    @pytest.mark.parametrize(
        "key, contains",
        [
            ("a", True),
            ("b", False),
            ("c", True),
            ("d", False),
            ("e", True),
            (("e", "f"), False),
            ("h", False),
        ],
    )
    def test_contains(self, key, contains):
        """Tests the parameter dictionary can be set and returns the correct value,
        including the use of tuples."""

        # Create the parameter dictionary.
        param_dict = parser_base.param_dict(parent=self.DummyLoadedParser, vals=self.DummyLoadedParser.params)  # type: ignore
        # Test its access
        assert (key in param_dict) == contains

    @pytest.mark.parametrize(
        "key, value_or_exception",
        [
            ("a", 1),
            ("b", KeyError),
            ("c", 3),
            ("d", KeyError),
            #    ("e", 8), #TODO: Fix this test... pytest catches a try/except block.
            (("e", "f"), 8),
            ("h", KeyError),
        ],
    )
    def test_get(self, key, value_or_exception):
        """Tests the values returned by the parameter dictionary."""

        # Create the parameter dictionary.
        param_dict = parser_base.param_dict(parent=self.DummyLoadedParser, vals=self.DummyLoadedParser.params)  # type: ignore
        # Test its access
        if isinstance(value_or_exception, type) and issubclass(
            value_or_exception, Exception
        ):
            with pytest.raises(value_or_exception) as e:
                param_dict[key]
        else:
            assert param_dict[key] == value_or_exception


##############################################################################
############################ Test parser_base ################################
##############################################################################


class TestParserBase:
    """Tests the base parser class and its methods."""

    pass
