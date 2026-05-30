"""
Test the parser parameter dictionary, which allows for relabelled access to parameters.

The parameter dictionary should allow for access to parameters using both their original
keys and any relabelled keys defined in the parser's relabels dictionary.
"""

from typing import Any, cast

import pytest

from pyNexafs.parsers import parserBase, parserMeta


class TestParamDict:
    """Tests the parameter dictionary functionality, which stores a set of parameters,
    but allows relabelled indexing and contains."""

    class DummyLoadedParser:
        RELABELS = parserMeta.relabels_dict({"a": "b", "c": "d", ("e", "f"): "g"})
        RELABELS_REVERSE = {"b": "a", "d": "c", "g": ("e", "f")}
        params = {"a": 1, "c": 3, "f": 6, "g": 8}

    class DummyLoadedParserNoRelabel(DummyLoadedParser):
        relabel = False

    class DummyLoadedParserRelabel(DummyLoadedParser):
        relabel = True

    @staticmethod
    def _param_dict(parent: type[Any]) -> parserBase.param_dict:
        return parserBase.param_dict(map=parent.params, parent=cast(Any, parent))

    @pytest.mark.parametrize(
        "key, contains, contains_relabelled",
        [
            ("a", True, True),
            ("b", False, True),
            ("c", True, True),
            ("d", False, True),
            ("e", False, True),
            (("e", "f"), False, False),
            ("h", False, False),
        ],
    )
    def test_contains(self, key, contains, contains_relabelled):
        """Tests the parameter dictionary can be set and returns the correct value,
        including the use of tuples.

        Note, currently `in` keyword does not use the `__contains__` method. https://stackoverflow.com/a/38542777/1717003
        This may be confusing to users, but is not a bug in the code.
        """

        param_dict = self._param_dict(self.DummyLoadedParserNoRelabel)
        assert param_dict.__contains__(key) == contains, "Contains failed."

        param_dict = self._param_dict(self.DummyLoadedParserRelabel)
        assert param_dict.__contains__(key) == contains_relabelled, (
            "Contains relabelled failed."
        )

    @pytest.mark.parametrize(
        "key, value_or_exception, value_or_exception_relabelled",
        [
            ("a", 1, 1),
            ("b", KeyError, 1),
            ("c", 3, 3),
            ("d", KeyError, 3),
            (
                "e",
                KeyError,
                8,
            ),
            (("e", "f"), KeyError, 8),
            ("h", KeyError, KeyError),
        ],
    )
    def test_get(self, key, value_or_exception, value_or_exception_relabelled):
        """Tests the values returned by the parameter dictionary."""

        param_dict = self._param_dict(self.DummyLoadedParserNoRelabel)
        if isinstance(value_or_exception, type) and issubclass(
            value_or_exception, Exception
        ):
            with pytest.raises(value_or_exception):
                param_dict[key]
            with pytest.raises(value_or_exception):
                param_dict.__getitem__(key)
        else:
            assert param_dict[key] == value_or_exception
            assert param_dict.__getitem__(key) == value_or_exception

        param_dict = self._param_dict(self.DummyLoadedParserRelabel)
        if isinstance(value_or_exception_relabelled, type) and issubclass(
            value_or_exception_relabelled, Exception
        ):
            with pytest.raises(value_or_exception_relabelled):
                param_dict[key]
            with pytest.raises(value_or_exception_relabelled):
                param_dict.__getitem__(key)
        else:
            assert param_dict[key] == value_or_exception_relabelled
            assert param_dict.__getitem__(key) == value_or_exception_relabelled

    @pytest.mark.parametrize(
        "key, expected_get_value",
        [
            ("a", 1),
            ("b", None),
            ("c", 3),
            ("d", None),
            ("e", None),
            (("e", "f"), None),
            ("f", 6),
            ("g", 8),
            ("h", None),
        ],
    )
    def test_get_method(self, key, expected_get_value):
        """Tests the inherited `get` method on the parameter dictionary."""

        default = object()

        param_dict = self._param_dict(self.DummyLoadedParserNoRelabel)
        assert param_dict.get(key) == expected_get_value
        if expected_get_value is None:
            assert param_dict.get(key, default) is default
        else:
            assert param_dict.get(key, default) == expected_get_value

        param_dict = self._param_dict(self.DummyLoadedParserRelabel)
        assert param_dict.get(key) == expected_get_value
        if expected_get_value is None:
            assert param_dict.get(key, default) is default
        else:
            assert param_dict.get(key, default) == expected_get_value
