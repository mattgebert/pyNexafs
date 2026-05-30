""" "Tests the base parser classes and their methods"""

import tempfile
import pytest
import numpy as np

# Test the creation of a new parser class.
from pyNexafs.parsers.base import parserBase
from pyNexafs.types import dtype


##############################################################################
############################ Test parserMeta ################################
##############################################################################


class TestParserMeta:
    """Tests the metaclass functionality on creating new Parser classes."""

    def test_missing_ALLOWED_EXTENSIONS(self):
        """Tests ValueError is raised when ALLOWED_EXTENSIONS is not defined."""
        with pytest.raises(ValueError) as e:

            class Test_Parser(parserBase):
                pass

        assert "Class Test_Parser does not define ALLOWED_EXTENSIONS." in str(e)

    def test_missing_COLUMN_ASSIGNMENTS(self):
        """Tests ValueError is raised when COLUMN_ASSIGNMENTS is not defined."""
        with pytest.raises(ValueError) as e:

            class Test_Parser(parserBase):
                ALLOWED_EXTENSIONS = [".txt"]
                pass

        assert "Class Test_Parser does not define COLUMN_ASSIGNMENTS." in str(e)

    @pytest.mark.parametrize(
        "summary_params, should_fail, error_msg",
        [
            # Valid cases - only strings allowed
            (["a", "b", "c"], False, None),
            (["param1"], False, None),
            ([], False, None),  # Empty list is valid
            (["x", "y", "z"], False, None),
            # Invalid cases - non-string items
            ([1, 2, 3], True, "Invalid type for item 0 in SUMMARY_PARAMS"),
            (["a", 1, "c"], True, "Invalid type for item 1 in SUMMARY_PARAMS"),
            (["a", ["b", "c"], "d"], True, "Invalid type for item 1 in SUMMARY_PARAMS"),
            (
                ["a", ("b", "c"), "d"],
                True,
                "Invalid type for item 1 in SUMMARY_PARAMS",
            ),  # Tuples not allowed
            (["a", None, "c"], True, "Invalid type for item 1 in SUMMARY_PARAMS"),
            (["a", 3.14, "c"], True, "Invalid type for item 1 in SUMMARY_PARAMS"),
        ],
    )
    def test_SUMMARY_PARAMS_validation(self, summary_params, should_fail, error_msg):
        """Tests that SUMMARY_PARAMS is required to be a list of strings only."""
        if should_fail:
            with pytest.raises(ValueError) as e:

                def parse_fn(file):
                    return file

                class Test_Parser(parserBase):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                    SUMMARY_PARAMS = summary_params
                    parse_test = parse_fn

            assert error_msg in str(e.value)
        else:
            # Should succeed without raising an exception
            def parse_fn(file):
                return file

            class Test_Parser(parserBase):
                ALLOWED_EXTENSIONS = [".txt"]
                COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                SUMMARY_PARAMS = summary_params
                parse_test = parse_fn

            # Verify the parser class was created successfully
            assert hasattr(Test_Parser, "SUMMARY_PARAMS")


class TestParserBase:
    """Tests the base parser class and its methods."""

    class TestInitialisation:
        """Tests the initialisation of the parserBase class."""

        def test_abstract(self):
            """Tests the parser base class is abstract"""
            with pytest.raises(TypeError) as e:
                parserBase(None)
            assert "Cannot instantiate abstract class `parserBase`." in str(e)

        def test_no_parse_fn(self):
            """Tests an invalid implementations of the parserBase class with no parse method."""

            # No parser methods
            with pytest.raises(AttributeError) as e:

                class NoParserMethods(parserBase):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}

            assert "No parser methods found in `NoParserMethods` class." in str(e)

        def test_parse_fn_params(self):
            # One parameter
            with pytest.raises(NameError) as e:

                def parse_fn(x):
                    return x

                class ParserRelabelOverlapMono(parserBase):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                    RELABELS = {}
                    parse_test = parse_fn

            assert (
                "First argument of static parser method must be 'file'. Is x." in str(e)
            )
            e = None

            # Two parameters
            def parse_fn_duo(file, y):
                return y

            class ParserRelabelOverlapDuo(parserBase):
                ALLOWED_EXTENSIONS = [".txt"]
                COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                RELABELS = {}
                parse_test = parse_fn_duo

            # With Header only
            def parser_fn_header(file, y, header_only):
                return header_only

            class ParserRelabelOverlapHeader(parserBase):
                ALLOWED_EXTENSIONS = [".txt"]
                COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                RELABELS = {}
                parse_test = parser_fn_header

        # def test_parser_fn_reuse_params(self):
        #     # Test that settings are used in the next parser function call.

        @pytest.mark.parametrize(
            "relabels, assertion",
            [
                (
                    {"a": "b", "c": "b"},
                    r"Duplicate - value 'b' in class `ParserRelabelOverlap.RELABELS` dictionary with keys 'a' and 'c'.",
                ),
                (
                    {"a": "b", "b": "c"},
                    r"Overlap - key:value pair 'b':'c' "
                    + r"in class `ParserRelabelOverlap.RELABELS` dictionary overlaps with key:value pair 'a':'b'.",
                ),
                (
                    {"a": "b", ("b", "c"): "d"},
                    r"Overlap - key:value pair '('b', 'c')':'d' with subkey 'b' as an overlapping entry "
                    + r"in class'ParserRelabelOverlap.RELABELS' dictionary with key:value pair 'a':'b'",
                ),
                (
                    {("a", "b"): "c", ("e", "c"): "d"},
                    r"Overlap - key:value pair '('e', 'c')':'d' with subkey 'c' as an overlapping entry "
                    + r"in class'ParserRelabelOverlap.RELABELS' dictionary with key:value pair '('a', 'b')':'c'.",
                ),
                (
                    {("a", "b"): "c", "b": "f"},
                    r"Duplicate - key 'b' is a duplicate key entry in `ParserRelabelOverlap.RELABELS` dictionary.",
                ),
                (
                    {("a", "b"): "c", ("b", "e"): "f"},
                    r"Duplicate - key 'b' from tuple '('b', 'e')' is a duplicate key entry in `ParserRelabelOverlap.RELABELS` dictionary.",
                ),
                (
                    {("a", dtype.PEY): "c", dtype.PEY: "f"},
                    r"Duplicate - dtype key 'Partial Electron Yield' in class `ParserRelabelOverlap.RELABELS` dictionary with value 'f'.",
                ),
            ],
        )
        def test_RELABEL_overlap(self, relabels, assertion):
            """Tests the handling of overlapping values in the RELABELS dictionary"""
            # Test string string overlap
            with pytest.raises(ValueError) as e:

                def parse_fn(file):
                    return file

                class ParserRelabelOverlap(parserBase):
                    ALLOWED_EXTENSIONS = [".txt"]
                    COLUMN_ASSIGNMENTS = {"x": "a", "y": "b"}
                    RELABELS = relabels
                    parse_test = parse_fn

            # Trim the type and generic messages.
            substr = (
                repr(e.value)
                .split("\n")[0]
                .replace("ValueError(", "")
                .replace("'", "'")
            )
            print(substr)
            assert assertion in substr

    class TestGetItem:
        """Tests the `__getitem__` method of the parserBase class."""

        class ParserGetitem(parserBase):
            ALLOWED_EXTENSIONS = [".txt"]
            COLUMN_ASSIGNMENTS = {"x": "a", "y": ["b", dtype.TEY]}
            RELABELS = {
                "a": "b",
                "c": dtype.E,
                "d": "e",
                "DrainCurrent": dtype.TEY,
            }

            @classmethod
            def parse_txt(cls, file):
                fake_data = np.random.rand(10, 4)  # 10 rows, 4 channels
                labels = [
                    "a",
                    "c",
                    "f",
                    "DrainCurrent",
                ]  # Original labels for the channels
                units = ["", "", "", ""]  # Units for the channels
                params = {"filename": file}  # Example parameters
                return fake_data, labels, units, params

        @pytest.mark.parametrize(
            "key,err",
            [
                ("a", None),
                ("b", None),
                ("g", KeyError),
                ("c", None),
                (dtype.E, None),
                (dtype.PFY, KeyError),
            ],
        )
        def test_getitem(self, key, err):
            """Tests the `__getitem__` method of the parserBase class,
            which should return a 1D channel, using relabels if necessary."""

            # Create a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write("Fake data for testing.")
                obj = self.ParserGetitem(f)

                # Test the getitem method
                if err:
                    with pytest.raises(err):
                        _ = obj[key]
                else:
                    channel = obj[key]
                    assert isinstance(channel, np.ndarray), (
                        "Returned channel is not a numpy array."
                    )
                    assert channel.ndim == 1, "Returned channel is not 1D."
