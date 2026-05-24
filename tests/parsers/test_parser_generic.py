"""
Standard tests to perform on all parsers.

Ensures all parsers meet the minimum requirements of pyNexafs implementation.
"""

import pytest
from pyNexafs.parsers.base import parserBase
from typing import Sequence
from types import MethodType, FunctionType
import numpy as np
import os


class ParserTestsMeta(type):
    """
    Metaclass for ParserTests to ensure that subclasses set required class attributes.
    """

    def __new__(cls, name, bases, attrs):
        if name != "ParserTests":  # Skip the base class itself
            # Ensure that PARSER_CLASS is defined and is a subclass of parserBase
            if "PARSER_CLASS" not in attrs or attrs["PARSER_CLASS"] is None:
                raise TypeError(f"{name} must define a PARSER_CLASS attribute.")
            if not issubclass(attrs["PARSER_CLASS"], parserBase):
                raise TypeError(
                    f"{name}.PARSER_CLASS must be a subclass of parserBase."
                )

            # Ensure that TEST_FILES is defined and is a sequence of tuples or strings
            if "TEST_FILES" not in attrs or attrs["TEST_FILES"] is None:
                raise TypeError(f"{name} must define a TEST_FILES attribute.")
            if not isinstance(attrs["TEST_FILES"], Sequence):
                raise TypeError(
                    f"{name}.TEST_FILES must be a sequence of file paths or tuples of (file path, pytest mark)."
                )
                for item in attrs["TEST_FILES"]:
                    if isinstance(item, str):
                        continue
                    elif (
                        isinstance(item, tuple)
                        and len(item) == 2
                        and isinstance(item[0], str)
                        and isinstance(item[1], pytest.MarkDecorator)
                    ):
                        continue
                    else:
                        raise TypeError(
                            f"Each item in `{name}.TEST_FILES` must be a string file path or a tuple of (file_path, pytest.MarkDecorator)."
                        )

        new_cls = super().__new__(cls, name, bases, attrs)

        if name != "ParserTests":
            params: list[object] = []
            ids: list[str] = []
            test_files = getattr(new_cls, "TEST_FILES", ())
            for test_file in test_files:
                if isinstance(test_file, tuple) and len(test_file) == 3:
                    filepath, func, mark = test_file
                    if mark is None:
                        [
                            params.append((filepath, func, header_only))
                            for header_only in (True, False)
                        ]
                    else:
                        [
                            params.append(
                                pytest.param(filepath, func, header_only, marks=mark)
                            )
                            for header_only in (True, False)
                        ]
                elif isinstance(test_file, tuple) and len(test_file) == 2:
                    filepath, func = test_file
                    [
                        params.append((filepath, func, header_only))
                        for header_only in (True, False)
                    ]
                else:
                    if isinstance(test_file, str):
                        filepath = test_file
                    else:
                        raise TypeError(
                            f"Each item in `{name}.TEST_FILES` must be a string file path or a tuple of (file_path, pytest.MarkDecorator). Found: {test_file}"
                        )
                    [
                        params.append((filepath, None, header_only))
                        for header_only in (True, False)
                    ]
                [
                    ids.append(
                        os.path.basename(filepath) + f" (header_only={header_only})"
                    )
                    for header_only in (True, False)
                ]

            test_fn = getattr(new_cls, "test_parser_initialization", None)
            if test_fn is not None:
                decorated = pytest.mark.parametrize(
                    "test_file, parser_fn, header_only", params, ids=ids
                )(test_fn)
                setattr(new_cls, "test_parser_initialization", decorated)

        return new_cls


class ParserTests(metaclass=ParserTestsMeta):
    """Tests that all parsers meet the minimum requirements of pyNexafs implementation."""

    # To be set by subclasses.
    PARSER_CLASS: type[parserBase] = parserBase
    """The parser class to be tested. Must be set by subclasses."""
    TEST_FILES: Sequence[
        tuple[str, FunctionType | MethodType, pytest.MarkDecorator | None]
        | tuple[str, FunctionType | MethodType]
        | str
    ] = tuple()
    """
    A sequence of items describing the expected testing results for each datafile.
    The more comprehensive the file information the better - ideally a filepath and an expected parse function (which succeeds). A pytest mark can also be included to indicate expected failure or other conditions. Each item can be one of the following:
    - A tuple of (file path, function, pytest mark)
    - A tuple of (file path, function)
    - A string file path.
    """

    @staticmethod
    def parser_initialization_filepath(
        parser: type[parserBase],
        filepath: str,
        header_only: bool,
    ) -> parserBase:
        """
        Test that the parser can initialise with provided test files.

        Parameters
        ----------
        parser : type[parserBase]
            The parser class to be tested.
        filepath : str
            The path to the test file to be used for initialization.

        Returns
        -------
        parserBase
            An instance of the parser class initialized with the provided file.
        """
        # Tests that the parser can be initialized with the given filepath
        assert os.path.exists(filepath), f"File not found: {filepath}"
        parser_instance = parser(filepath, header_only=header_only)
        return parser_instance

    @staticmethod
    def parser_initialization_filebuffer(
        parser: type[parserBase],
        filepath: str,
        header_only: bool,
    ) -> parserBase:
        """
        Test that the parser can initialise with a file buffer `open(path, 'rb')`.

        Parameters
        ----------
        parser : type[parserBase]
            The parser class to be tested.
        filepath : str
            The path to the test file to be used for initialization.

        Returns
        -------
        parserBase
            An instance of the parser class initialized with a file buffer of the provided file.
        """
        # Tests that the parser can be initialized with a file buffer
        assert os.path.exists(filepath), f"File not found: {filepath}"
        with open(filepath, "rb", buffering=0) as f:
            parser_instance = parser(f, header_only=header_only)
        return parser_instance

    @staticmethod
    def parser_initialization_textwrapper(
        parser: type[parserBase],
        filepath: str,
        header_only: bool,
    ) -> parserBase:
        """
        Test that the parser can successfully initialise with a standard `open(path, 'r')`.

        Parameters
        ----------
        parser : type[parserBase]
            The parser class to be tested.
        filepath : str
            The path to the test file to be used for initialization.

        Returns
        -------
        parserBase
            An instance of the parser class initialized with a text wrapper of the provided file.
        """
        # Tests that the parser can be initialized with a text wrapper
        assert os.path.exists(filepath), f"File not found: {filepath}"
        with open(filepath, "r", encoding="utf-8") as f:
            parser_instance = parser(f, header_only=header_only)
        return parser_instance

    def test_parser_initialization(
        self,
        test_file: str,
        parser_fn: FunctionType | MethodType | None,
        header_only: bool,
    ):
        """
        Test that the parser can successfully initialise for one file.

        Each file is one pytest item; within that item, all supported
        initialization paths and header modes are validated.

        Parameters
        ----------
        test_file : str
            A file path for the test item.
        parser_fn : FunctionType | MethodType | None
            The expected 'successful' parser function for the file.
        header_only : bool
            Whether to initialize the parser in header-only mode.
        """
        init_methods = [
            self.parser_initialization_filepath,
            self.parser_initialization_filebuffer,
            self.parser_initialization_textwrapper,
        ]
        for init_method in init_methods:
            parser_instance = init_method(
                self.PARSER_CLASS,
                test_file,
                header_only=header_only,
            )
            assert isinstance(parser_instance, self.PARSER_CLASS), (
                f"Parser instance is not of type {self.PARSER_CLASS.__name__}"
            )

            assert parser_instance.parser_fn == parser_fn, (
                f"Expected parser function {parser_fn}, but got {parser_instance.parser_fn} (header_only={header_only})"
            )
            if header_only:
                assert parser_instance.data is None, (
                    "Data should be None in header-only mode."
                )
            else:
                assert parser_instance.data is not None, (
                    "Data should not be None when not in header-only mode."
                )
                if isinstance(parser_instance.data, (tuple, list)):
                    assert all(
                        isinstance(d, np.ndarray) for d in parser_instance.data
                    ), "All data elements should be numpy arrays."
                else:
                    assert isinstance(parser_instance.data, np.ndarray), (
                        "Data should be a numpy array."
                    )
