"""
Test the summary parameters list functionality in the parser meta class.

Summary params are mean to be a list of parameters to summarize the file,
and this implementation should act as a "relabelled" list when the parser has
`relabels` enabled, particularly for get / indexing / contains / __getitem__ methods.
"""

import pytest
from pyNexafs.parsers import parserMeta


class TestSummaryParamList:
    """Tests the summary parameters list functionality."""

    class ParserNoRelabels:
        RELABELS = {}
        RELABELS_REVERSE = {}
        relabel = False

    class ParserRelabels:
        RELABELS = {"a": "b", "c": "d", ("e", "f"): "g"}
        RELABELS_REVERSE = {"b": "a", "d": "c", "g": ("e", "f")}
        relabel = True

    @pytest.mark.parametrize(
        "elements, err, overlapping_param",
        [
            (["a", "c"], None, None),
            (["a", "c", "c"], ValueError, "c"),
            (["a", "b", "c"], None, None),
            (["1", "2", "3"], None, None),
            (["a", "b", "d"], None, None),
            (["a", "b", "b"], ValueError, "b"),
        ],
    )
    def test_init_valid(self, elements, err, overlapping_param):
        """Tests the summary parameters list can be created and initialised, particularly the `__check_valid` method."""
        # Create the summary parameters list.
        if err:
            with pytest.raises(err) as e:
                parserMeta.summary_param_list(elements, parent=self.ParserNoRelabels)  # type: ignore
            if overlapping_param:
                assert f"Summary Parameter `{overlapping_param}`" in str(e.value)
        else:
            parserMeta.summary_param_list(elements, parent=self.ParserNoRelabels)  # type: ignore

    @pytest.mark.parametrize(
        "elements, err, overlapping_param",
        [
            (["a", "c"], None, None),
            (["a", "c", "c"], ValueError, "c"),
            (["a", "b", "c"], ValueError, "b"),
            (["1", "2", "3"], None, None),
            (["e", "f"], None, None),
            (["a", "d", "b"], ValueError, "b"),
            (["0", "b", "b"], ValueError, "b"),
            (["a", "d", "c"], ValueError, "c"),
        ],
    )
    def test_init_valid_relabelled(self, elements, err, overlapping_param):
        """Tests the summary parameters list can be created and initialised, particularly the `__check_valid` method when relabelling."""
        # Create the summary parameters list.
        if err:
            with pytest.raises(err) as e:
                parserMeta.summary_param_list(elements, parent=self.ParserRelabels)  # type: ignore
            if overlapping_param:
                assert f"Summary Parameter `{overlapping_param}`" in str(e.value)
        else:
            parserMeta.summary_param_list(elements, parent=self.ParserRelabels)  # type: ignore

    @pytest.mark.parametrize(
        "elements, query, contains",
        [
            (["a", "b", "c"], "a", True),
            (["1", "2", "3"], "1", True),
            (["a", "b", "d"], "b", True),
            (["a", "b", "c"], "d", False),
        ],
    )
    def test_contains(self, elements, query, contains):
        """Tests the summary parameters list can be set and returns the correct value."""

        # Create the summary parameters list.
        summary_params = parserMeta.summary_param_list(
            elements, parent=self.ParserNoRelabels
        )  # type: ignore
        # Test its access
        assert (query in summary_params) == contains

    @pytest.mark.parametrize(
        "elements, query, relabelled_contains",
        [
            (["a", "c"], "a", True),
            (["a", "c"], "b", True),
            (["a", "c"], "e", False),
            (["1", "2", "3"], "1", True),
            (["e", "o"], "g", True),
            (["f", "o"], "g", True),
            (["f", "o"], "h", False),
        ],
    )
    def test_contains_relabelled(self, elements, query, relabelled_contains):
        """Tests the summary parameters list can be set and returns the correct value."""
        # Create the summary parameters list.
        summary_params = parserMeta.summary_param_list(
            elements, parent=self.ParserRelabels
        )  # type: ignore
        # Test its access
        assert (query in summary_params) == relabelled_contains

    @pytest.mark.parametrize(
        "elements, query, index_or_exception",
        [
            (["a", "b", "c"], "a", 0),
            (["1", "2", "3"], "1", 0),
            (["a", "b", "d"], "b", 1),
            (["a", "b", "d"], "d", 2),
            (["a", "b", "d"], "a", 0),
            (["a", "b", "c"], "e", ValueError),
        ],
    )
    def test_index(self, elements, query, index_or_exception):
        """Tests the summary parameters list can be indexed for string-only entries."""
        # Create the summary parameters list.
        summary_params = parserMeta.summary_param_list(
            elements, parent=self.ParserNoRelabels
        )  # type: ignore
        # Test its access
        if isinstance(index_or_exception, type) and issubclass(
            index_or_exception, Exception
        ):
            with pytest.raises(
                index_or_exception,
                match=f"{query} is not in the list.",
            ):
                summary_params.index(query)
        elif isinstance(index_or_exception, int):
            assert summary_params.index(query) == index_or_exception
        else:
            raise ValueError("Invalid test case.")

    @pytest.mark.parametrize(
        "elements, query, index_or_exception",
        [
            (["a", "c"], "a", 0),
            (["a", "c"], "b", 0),
            (["1", "2", "3"], "3", 2),
            (["0", "b", "e"], "b", 1),
            (["0", "c", "e"], "d", 1),
            (["e", "o"], "g", 0),
            (["f", "o"], "g", 0),
            (["a", "0", "c"], "e", ValueError),
        ],
    )
    def test_index_relabelled(self, elements, query, index_or_exception):
        """Tests the summary parameters list can be indexed and relabels apply for string-only entries."""
        # Create the summary parameters list.
        summary_params = parserMeta.summary_param_list(
            elements, parent=self.ParserRelabels
        )  # type: ignore
        # Test its access
        if isinstance(index_or_exception, type) and issubclass(
            index_or_exception, Exception
        ):
            with pytest.raises(
                index_or_exception,
                match=f"{query} is not in the list.",
            ):
                summary_params.index(query)
