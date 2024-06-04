"""
Parser classes for the Generic data.

Utilises the generic numpy text loader to load data from a file.
"""

from pyNexafs.parsers import parser_base
from pyNexafs.nexafs.scan import scan_base
from io import TextIOWrapper
from typing import Any
from numpy.typing import NDArray
import numpy as np
import ast
import overrides


class MEX2_NEXAFS(parser_base):
    """
    Australian Synchrotron Soft X-ray (SXR) NEXAFS parser.

    Parses data formats including '.asc' and '.mda' formats from the SXR
    Near Edge X-ray Absorption Fine Structure (NEXAFS) tool.

    Attributes
    ----------
    ALLOWED_EXTENSIONS
    SUMMARY_PARAM_RAW_NAMES
    COLUMN_ASSIGNMENTS
    RELABELS

    Notes
    -----
    Implemented for data as of 2024-Mar.
    """

    ALLOWED_EXTENSIONS = [".xdi", ".mda"]
    SUMMARY_PARAM_RAW_NAMES = [
        "Sample",
        "ROI.start_bin",
        "ROI.end_bin",
        "Element.symbol",
        "Element.edge",
    ]
    COLUMN_ASSIGNMENTS = {
        "x": "energy",
        "y": [
            "bragg",
            "ifluor|ifluor_sum",
            "count_time",
            "i0",
            "SampleDrain",
            "ICR_AVG",
            "OCR_AVG",
        ],
        "y_errs": None,
        "x_errs": None,
    }
    RELABELS = {
        "ROI.start_bin": r"$E_1$",
        "ROI.end_bin": r"$E_2$",
        "Element.symbol": "Element",
        "Element.edge": "Edge",
        "OCR_AVG": "Output Count Rate",
        "ICR_AVG": "Input Count Rate",
        "ifluor": "Fluorescence",
    }

    # @classmethod
    # @overrides.overrides
    # def file_parser(
    #     cls, file: TextIOWrapper, header_only: bool = False
    # ) -> tuple[NDArray | None, list[str], list[str], dict[str, Any]]:
    #     """Reads Australian Synchrotron Medium Energy Xray2 (MEX2) Spectroscopy files.

    #     Parameters
    #     ----------
    #     file : TextIOWrapper
    #         TextIOWrapper of the datafile (i.e. open('file.asc', 'r'))
    #     header_only : bool, optional
    #         If True, then only the header of the file is read and NDArray is returned as None, by default False

    #     Returns
    #     -------
    #     tuple[NDArray | None, list[str], dict[str, Any]]
    #         Returns a set of data as a numpy array,
    #         labels as a list of strings,
    #         units as a list of strings,
    #         and parameters as a dictionary.

    #     Raises
    #     ------
    #     ValueError
    #         If the file is not a valid filetype.
    #     """
    #     # Init vars, check file type using super method.
    #     data, labels, units, params = super().file_parser(file)

    #     # Use specific parser based on file extension.
    #     if file.name.endswith(".xdi"):
    #         data, labels, units, params = cls.parse_xdi(file, header_only=header_only)
    #     elif file.name.endswith(".mda"):
    #         data, labels, units, params = cls.parse_mda(file, header_only=header_only)
    #     else:
    #         raise NotImplementedError(
    #             f"File {file.name} is not yet supported by the {cls.__name__} parser."
    #         )

    #     # Add filename to params at the end, to avoid incorrect filename from copy files internal params.
    #     params["filename"] = file.name

    #     return data, labels, units, params

    @classmethod
    def parse_xdi(
        cls, file: TextIOWrapper, header_only: bool = False
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """Reads Australian Synchrotron .xdi files.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.xdi', 'r'))
        header_only : bool, optional
            If True, then only the header of the file is read and NDArray is returned as None, by default False

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array,
            labels as a list of strings,
            units as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .xdi file.
        """
        # Initialise structures
        params = {}
        labels = []
        units = None

        # Check valid format.
        if not file.name.endswith(".xdi"):
            raise ValueError(f"File {file.name} is not a valid .xdi file.")

        ### Read file
        # Check header is correct
        assert file.readline() == "# XDI/1.0\n"

        ## 1 Initial Parameters
        # Read first param line.
        line = file.readline()
        while "# " == line[0:2] and line != "# ///\n":
            line = line[2:].strip().split(": ", 1)  # split first colon
            param, value = tuple(line)

            # Categorise information
            if "Column." in param:
                if "Describe." in param:
                    # Label description. No dedicated structure for this, add to params.
                    params[param] = value
                else:
                    # Label name
                    labels.append(value)
            else:
                if param == "Samples":
                    # Parse the datapoint length list
                    samples = ast.literal_eval(value)
                    samples = [x.strip() if isinstance(x, str) else x for x in samples]
                    # Add to params
                    for i in range(len(samples)):
                        params[f"Datapoints.Column.{i}"] = samples[i]
                else:
                    # General parameter, add to params.
                    params[param] = value
            # Load new param line
            line = file.readline()

        assert line == "# ///\n"  # Check end of initial parameters

        # Get samplename
        line = file.readline()
        sample_name = line[2:].strip()
        params["Sample"] = sample_name

        # Skip header lines before data
        assert file.readline() == "# \n"
        line = file.readline()
        # Some xdi conversion have a "default mda2xdi" line.
        try:
            assert line == "# xdi from default mda2xdi preset for mex2.\n"
            assert file.readline() == "#--------\n"
        except AssertionError:
            assert line == "#--------\n"

        # Read data columns
        header_line = file.readline()
        assert header_line[0:2] == "# "
        labels = (
            header_line[2:].strip().split()
        )  # split on whitespace, even though formatting seems to use "   " (i.e. three spaces).
        labels = [label.strip() if type(label) == str else label for label in labels]

        if header_only:
            # Do not process remaining lines
            return None, labels, units, params

        # Read data
        lines = file.readlines()  # read remaining lines efficiently

        # Convert data to numpy array.
        data = np.loadtxt(lines)
        data = np.array(data)

        return data, labels, units, params
