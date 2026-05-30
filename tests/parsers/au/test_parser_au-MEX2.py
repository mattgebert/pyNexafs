"""Tests for the MEX2 parser - check that example files load correctly for each parser function version"""

import pytest
import os
from pyNexafs.parsers.au.aus_sync.MEX2 import MEX2_NEXAFS
from tests.parsers.test_parser_generic import ParserTests

pytest_plugins = ["tests.gui.fixtures"]

# Relative directories for the test data.
LOCAL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "au", "MEX2")
)

MDA_PARSE_FN_BY_FOLDER = {
    "2024-03": MEX2_NEXAFS.parse_mda_2024_11,
    "2025-03": MEX2_NEXAFS.parse_mda_2025_02,
}

PARTIAL_SCAN_MARKERS_BY_FOLDER = {
    "2024-03": {"5641"},
    "2025-03": {"13366"},
}


def _is_partial_scan(filepath: str, folder_name: str) -> bool:
    """Return True when file is a known partial scan for this data vintage."""
    markers = PARTIAL_SCAN_MARKERS_BY_FOLDER.get(folder_name, set())
    return any(marker in os.path.basename(filepath) for marker in markers)


def _build_test_files() -> list[tuple[str, object, pytest.MarkDecorator | None]]:
    """Build parser-generic test cases across all known MEX2 data folders."""
    test_files: list[tuple[str, object, pytest.MarkDecorator | None]] = []
    for folder_name in sorted(os.listdir(LOCAL_DIR)):
        folder_path = os.path.join(LOCAL_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        mda_parser_fn = MDA_PARSE_FN_BY_FOLDER.get(folder_name)
        if mda_parser_fn is None:
            continue

        for file in sorted(os.listdir(folder_path)):
            if not file.endswith(tuple(MEX2_NEXAFS.ALLOWED_EXTENSIONS)):
                continue

            filepath = os.path.join(folder_path, file)
            parser_fn = (
                mda_parser_fn if file.endswith(".mda") else MEX2_NEXAFS.parse_xdi
            )
            mark = (
                pytest.mark.xfail(reason=f"Known partial scan in {folder_name}.")
                if _is_partial_scan(filepath, folder_name)
                else None
            )
            test_files.append((filepath, parser_fn, mark))
    return test_files


class TestParserMEX2(ParserTests):
    """Tests that the MEX2 parser can be initialized with example files and that the correct parsing method is called."""

    PARSER_CLASS = MEX2_NEXAFS
    TEST_FILES = _build_test_files()


class TestScanMEX2:
    """Tests scan conversion for MEX2 data folders via the mex2_data_dir fixture."""

    @pytest.mark.parametrize(
        "header_only, energy_bin_domain", [(True, None), (False, (2200, 2600))]
    )
    def test_scan_mda(self, mex2_data_dir, header_only, energy_bin_domain):
        folder_name = os.path.basename(mex2_data_dir)
        expected_mda_parser = MDA_PARSE_FN_BY_FOLDER.get(folder_name)
        if expected_mda_parser is None:
            pytest.skip(f"No expected parser mapping for folder '{folder_name}'.")

        mda_files = sorted(
            file for file in os.listdir(mex2_data_dir) if file.endswith(".mda")
        )
        assert mda_files, f"No MDA files found in {mex2_data_dir}."

        for file in mda_files:
            filepath = os.path.join(mex2_data_dir, file)
            try:
                parser = MEX2_NEXAFS(
                    filepath,
                    header_only=header_only,
                    relabel=False,
                )
            except Exception:
                if _is_partial_scan(filepath, folder_name):
                    # Known partial scans can legitimately fail to parse.
                    continue
                raise

            assert parser.parser_fn == expected_mda_parser

            data = parser.data
            if header_only:
                with pytest.raises(
                    ValueError, match=r"(No data loaded into the parser object)(.*+)"
                ):
                    parser.to_scan(energy_bin_domain=energy_bin_domain)
                continue

            assert data is not None and isinstance(data, tuple) and len(data) == 2
            data1d, data2d = data
            assert data1d is not None and data2d is not None

            scan = parser.to_scan(
                energy_bin_domain=energy_bin_domain,
                load_all_columns=True,
            )
            assert scan.y is not None
            assert scan.y.shape[-1] == data1d.shape[-1] + data2d.shape[-1]


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
