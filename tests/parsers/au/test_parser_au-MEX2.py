"""Tests for the MEX2 parser - check that example files load correctly for each parser function version"""

import pytest
import os
from pyNexafs.parsers.au.aus_sync.MEX2 import MEX2_NEXAFS
from tests.parsers.test_parser_generic import ParserTests

# Relative directories for the test data.
LOCAL_DIR = "tests/test_data/au/MEX2/"
PATH_202503 = os.path.join(
    LOCAL_DIR, "2025-03"
)  # Not sure why I can't put this as a class attr.
PATH_202403 = os.path.join(LOCAL_DIR, "2024-03")


class TestParserMEX2(ParserTests):
    """Tests that the MEX2 parser can be initialized with example files and that the correct parsing method is called."""

    PARSER_CLASS = MEX2_NEXAFS
    TEST_FILES = [
        (
            os.path.join(PATH_202403, file),
            MEX2_NEXAFS.parse_mda_2024_11
            if file.endswith(".mda")
            else MEX2_NEXAFS.parse_xdi,
            pytest.mark.xfail(reason="5641 is a partial scan. It should fail.")
            if "5641" in file
            else None,
        )
        for file in os.listdir(PATH_202403)
        if file.endswith(tuple(MEX2_NEXAFS.ALLOWED_EXTENSIONS))
    ] + [
        (
            os.path.join(PATH_202503, file),
            MEX2_NEXAFS.parse_mda_2025_02
            if file.endswith(".mda")
            else MEX2_NEXAFS.parse_xdi,
            pytest.mark.xfail(reason="13366 is a partial scan. It should fail.")
            if "13366" in file
            else None,
        )
        for file in os.listdir(PATH_202503)
        if file.endswith(tuple(MEX2_NEXAFS.ALLOWED_EXTENSIONS))
    ]


class TestScanMEX2:
    """Tests the loading of example files"""

    class Test_MDA_2024_11:
        # Test scan conversion for reduction of multi-dimensional fluorescence data
        @pytest.mark.parametrize(
            "header_only, energy_bin_domain", [(True, None), (False, (2200, 2600))]
        )
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202403, file)
                    if "5641" not in file
                    else pytest.param(
                        os.path.join(PATH_202403, file), marks=pytest.mark.xfail
                    )
                )  # 5641 is a partial scan. It should fail.
                for file in os.listdir(PATH_202403)
                if file.endswith(".mda")
            ],
        )
        def test_scan_mda_2024_11(self, filepath, header_only, energy_bin_domain):
            # Test the class constructor success
            parser = MEX2_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )

            data = parser.data
            if header_only:
                with pytest.raises(
                    ValueError, match=r"(No data loaded into the parser object)(.*+)"
                ):
                    scan = parser.to_scan(energy_bin_domain=energy_bin_domain)
                return

            assert data is not None and isinstance(data, tuple) and len(data) == 2
            data1D, data2D = data
            assert data1D is not None and data2D is not None

            scan = parser.to_scan(
                energy_bin_domain=energy_bin_domain, load_all_columns=True
            )
            # Add fluorescence channels
            assert scan.y is not None
            assert (
                scan.y.shape[-1] == data1D.shape[-1] + data2D.shape[-1] + 1 - 1
            )  # +1 for sum fluor, -1 for energy (x).

    class Test_MDA_2025_02:
        # Test scan conversion for reduction of multi-dimensional fluorescence data
        @pytest.mark.parametrize(
            "header_only, energy_bin_domain", [(True, None), (False, (2200, 2600))]
        )
        @pytest.mark.parametrize(
            "filepath",
            [
                (
                    os.path.join(PATH_202503, file)
                    if "13366" not in file
                    else pytest.param(
                        os.path.join(PATH_202503, file), marks=pytest.mark.xfail
                    )
                )  # 13366 is a partial scan. It should fail.
                for file in os.listdir(PATH_202503)
                if file.endswith(".mda")
            ],
        )
        def test_scan_mda_2025_02(self, filepath, header_only, energy_bin_domain):
            # Test the class constructor success
            parser = MEX2_NEXAFS(
                filepath,
                header_only=header_only,
                relabel=False,
            )

            data = parser.data
            if header_only:
                with pytest.raises(
                    ValueError, match=r"(No data loaded into the parser object)(.*+)"
                ):
                    scan = parser.to_scan(energy_bin_domain=energy_bin_domain)
                return

            assert data is not None and isinstance(data, tuple) and len(data) == 2
            data1D, data2D = data
            assert data1D is not None and data2D is not None

            scan = parser.to_scan(
                energy_bin_domain=energy_bin_domain, load_all_columns=True
            )
            # Add fluorescence channels
            assert scan.y is not None
            assert (
                scan.y.shape[-1] == data1D.shape[-1] + data2D.shape[-1] + 1 - 1
            )  # +1 for sum fluor, -1 for energy (x).


if __name__ == "__main__":
    # Test each of the parser functions with the test data.
    # MEX2 2024-03
    folder = "2025-03"
    # fname = "MEX2_13366.mda"
    fname = "MEX2_13385.mda"
    # fname = "MEX2_13366_processed.xdi"
    # MEX2 2025-03
    folder = "2024-03"
    fname = "MEX2_5641.mda"
    # fname = "MEX2_5641_processed.xdi"

    #
    import pkgutil
    import io

    bdata = pkgutil.get_data("pyNexafs", f"../tests/test_data/au/MEX2/{folder}/{fname}")
    assert bdata is not None
    reader = io.BytesIO(bdata)
    reader.name = fname
    if fname.endswith(".xdi"):
        reader = io.TextIOWrapper(reader, encoding="utf-8")

    if "2024-03" in folder:
        if fname.endswith(".mda"):
            result = MEX2_NEXAFS.parse_mda_2024_11(reader, header_only=False)
        elif fname.endswith(".xdi"):
            result = MEX2_NEXAFS.parse_xdi(reader, header_only=False)
        else:
            raise ValueError("File extension not recognised.")
    elif "2025-03" in folder:
        if fname.endswith(".mda"):
            result = MEX2_NEXAFS.parse_mda_2025_02(reader, header_only=False)
        elif fname.endswith(".xdi"):
            result = MEX2_NEXAFS.parse_xdi(reader, header_only=False)
        else:
            raise ValueError("File extension not recognised.")
    else:
        raise ValueError("Folder not recognised.")

    print(result)
