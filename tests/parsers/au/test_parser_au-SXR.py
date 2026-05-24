"""Tests for the MEX2 parser - check that example files load correctly for each parser function version"""

import pytest
import os
from pyNexafs.parsers.au.aus_sync.SXR import SXR_NEXAFS
from tests.parsers.test_parser_generic import ParserTests

# Relative directories for the test data.
LOCAL_DIR = "tests/test_data/au/SXR/"
PATH_202403 = os.path.join(
    LOCAL_DIR, "2024-03"
)  # Not sure why I can't put this as a class attr.

# 128577 is the Photodiode scan (double normalisation)
# 129578 is the 90 deg (normal) scan
# 129582 is the 20 deg (grazing) scan


class TestParserSXR(ParserTests):
    """
    SXR_NEXAFS parser tests.
    """

    PARSER_CLASS = SXR_NEXAFS
    TEST_FILES = [
        (
            # Filepath
            os.path.join(PATH_202403, file),
            # Parser function to use
            SXR_NEXAFS.parse_asc_202403
            if file.endswith(".asc")
            else SXR_NEXAFS.parse_mda_2024_03
            if file.endswith(".mda")
            else SXR_NEXAFS.parse_txt_qant_2025_12,  # Assuming .txt files
            # Marks
            None,
        )
        for file in os.listdir(PATH_202403)
        if file.endswith(tuple(SXR_NEXAFS.ALLOWED_EXTENSIONS))
    ]


class TestScanSXR:
    """
    Test that the SXR_NEXAFS parser can load the example files, and that the data is correct.
    """

    class Test_SXR_2024_03:
        @pytest.mark.parametrize("header_only", [False])
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202403, file)
                    # if "5641" not in file
                    # else pytest.param(
                    #     os.path.join(PATH_202403, file), marks=pytest.mark.xfail
                    # )
                )  # 5641 is a partial scan. It should fail.
                for file in os.listdir(PATH_202403)
                if file.endswith(".asc")
            ],
        )
        def test_scan_sxr_asc_2024_03(self, filepath, header_only):
            # Test the class constructor success
            parser = SXR_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )

            scan = parser.to_scan()
            assert scan is not None
            assert scan.x is not None
            assert scan.y is not None
            assert scan.x_label is not None
            assert scan.y_labels is not None
            assert scan.x_unit is not None
            assert scan.y_units is not None

        @pytest.mark.parametrize("header_only", [False])
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202403, file)
                    # if "5641" not in file
                    # else pytest.param(
                    #     os.path.join(PATH_202403, file), marks=pytest.mark.xfail
                    # )
                )  # 5641 is a partial scan. It should fail.
                for file in os.listdir(PATH_202403)
                if file.endswith(".mda")
            ],
        )
        def test_parser_sxr_mda_2024_03(self, filepath, header_only):
            # Test the class constructor success
            parser = SXR_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )

            scan = parser.to_scan()
            assert scan is not None
            assert scan.x is not None
            assert scan.y is not None
            assert scan.x_label is not None
            assert scan.y_labels is not None
            assert scan.x_unit is not None
            assert scan.y_units is not None

        @pytest.mark.parametrize("header_only", [False])
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202403, file)
                    # if "5641" not in file
                    # else pytest.param(
                    #     os.path.join(PATH_202403, file), marks=pytest.mark.xfail
                    # )
                )  # 5641 is a partial scan. It should fail.
                for file in os.listdir(PATH_202403)
                if file.endswith(".txt")
            ],
        )
        def test_scan_sxr_txt_2024_03(self, filepath, header_only):
            # Test the class constructor success
            parser = SXR_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )

            scan = parser.to_scan()
            assert scan is not None
            assert scan.x is not None
            assert scan.y is not None
            assert scan.x_label is not None
            assert scan.y_labels is not None
            assert scan.x_unit is not None
            assert scan.y_units is not None
