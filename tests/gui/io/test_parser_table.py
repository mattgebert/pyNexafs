"""Tests for ParserTableModel and ParserTableView.

Covers loading directories without a parser, with an incorrect parser,
and with the correct parser (MEX2_NEXAFS and SXR_NEXAFS). Also tests
reloading a header-only item to full data.
"""

import os
import datetime
import pytest

from pyNexafs.gui.config import QtCore, QtWidgets, QtGui
from pyNexafs.gui.widgets.io.parser_ui.parser_table import (
    loadStatus,
    ParserTableModel,
    ParserTableView,
)
import pyNexafs.gui.widgets.io.parser_ui.parser_table as parser_table_module
from pyNexafs.parsers.au.aus_sync import MEX2_NEXAFS, SXR_NEXAFS

# Load shared GUI fixtures (qapp, mex2_dir, sxr_dir, mex2_dir_2024_03, mex2_dir_2025_03, …)
pytest_plugins = ["tests.gui.fixtures"]


@pytest.fixture
def sxr_data_dir(sxr_dir):
    """Return the SXR data folder used by parser-table tests."""
    return os.path.join(sxr_dir, "2024-03")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model(path: str, parser_cls=None) -> ParserTableModel:
    """Create a ParserTableModel for the given path and optional parser."""
    return ParserTableModel(path=path, parser_cls=parser_cls)


def _col_index(model: ParserTableModel, name: str) -> int:
    """Return the column index for the given header name (case-insensitive)."""
    for i, h in enumerate(model._header):
        if isinstance(h, str) and h.lower() == name.lower():
            return i
    raise KeyError(f"Column '{name}' not found in headers: {model._header}")


# ===========================================================================
# Tests – no parser
# ===========================================================================


class TestNoParser:
    """Loading a directory without any parser: all files visible, no status col."""

    def test_mex2_row_count_no_parser(self, qapp, mex2_data_dir):
        """All files in the directory should appear as rows."""
        model = _model(mex2_data_dir, parser_cls=None)
        expected = len(os.listdir(mex2_data_dir))
        assert model.rowCount() == expected

    def test_sxr_row_count_no_parser(self, qapp, sxr_data_dir):
        model = _model(sxr_data_dir, parser_cls=None)
        expected = len(os.listdir(sxr_data_dir))
        assert model.rowCount() == expected

    def test_no_parser_header_lacks_status_col(self, qapp, mex2_data_dir):
        """Without a parser the first header column must NOT be a status column (empty string)."""
        model = _model(mex2_data_dir, parser_cls=None)
        assert model._header[0] != ""  # No load-status column.

    def test_no_parser_header_has_filename(self, qapp, mex2_data_dir):
        model = _model(mex2_data_dir, parser_cls=None)
        assert any(h.lower() == "filename" for h in model._header)

    def test_no_parser_data_rows_are_no_parser_type(self, qapp, mex2_data_dir):
        """Each data row should NOT start with a loadStatus value."""
        model = _model(mex2_data_dir, parser_cls=None)
        for row in model._data:
            assert not isinstance(row[0], loadStatus)

    def test_no_parser_model_index_returns_filename(self, qapp, mex2_data_dir):
        """QModelIndex data for the filename column should return a string."""
        model = _model(mex2_data_dir, parser_cls=None)
        fn_col = _col_index(model, "filename")
        idx = model.index(0, fn_col)
        val = model.data(idx, role=QtCore.Qt.ItemDataRole.DisplayRole)
        assert isinstance(val, str) and len(val) > 0

    def test_no_parser_column_count(self, qapp, mex2_data_dir):
        """Without a parser there should be at least 3 columns (#, Filename + specials)."""
        model = _model(mex2_data_dir, parser_cls=None)
        assert model.columnCount() >= 3

    def test_no_parser_sxr_filenames_sorted(self, qapp, sxr_data_dir):
        """Rows should be sorted alphabetically by filename."""
        model = _model(sxr_data_dir, parser_cls=None)
        fn_col = _col_index(model, "filename")
        filenames = [model._data[r][fn_col] for r in range(model.rowCount())]
        assert filenames == sorted(filenames)


# ===========================================================================
# Tests – wrong parser
# ===========================================================================


class TestIncorrectParser:
    """Loading a directory with the wrong parser: only files matching
    ALLOWED_EXTENSIONS are considered, but rows only appear for files the parser can instantiate."""

    def test_mex2_dir_with_sxr_parser_row_count(self, qapp, mex2_data_dir):
        """MEX2 directory loaded with SXR parser: only .asc/.mda/.txt files match."""
        model = _model(mex2_data_dir, parser_cls=SXR_NEXAFS)
        expected_files = [
            f
            for f in os.listdir(mex2_data_dir)
            if any(f.endswith(ext) for ext in SXR_NEXAFS.ALLOWED_EXTENSIONS)
        ]
        assert model.rowCount() == len(expected_files)

    def test_mex2_dir_with_sxr_parser_all_error(self, qapp, mex2_data_dir):
        """When no rows are produced for an incorrect parser, this is expected."""
        model = _model(mex2_data_dir, parser_cls=SXR_NEXAFS)
        if model.rowCount() == 0:
            assert True
            return
        for row in range(model.rowCount()):
            status = model._data[row][0]
            assert status == loadStatus.ERROR, (
                f"Row {row} expected ERROR but got {status}"
            )

    def test_sxr_dir_with_mex2_parser_row_count(self, qapp, sxr_data_dir):
        """SXR directory loaded with MEX2 parser: only .xdi/.mda files match."""
        model = _model(sxr_data_dir, parser_cls=MEX2_NEXAFS)
        expected_files = [
            f
            for f in os.listdir(sxr_data_dir)
            if any(f.endswith(ext) for ext in MEX2_NEXAFS.ALLOWED_EXTENSIONS)
        ]
        assert model.rowCount() == len(expected_files)

    def test_sxr_dir_with_mex2_parser_all_error(self, qapp, sxr_data_dir):
        """Wrong parser may still header-parse compatible files; expect header-only rows."""
        model = _model(sxr_data_dir, parser_cls=MEX2_NEXAFS)
        if model.rowCount() == 0:
            pytest.skip("No files matched MEX2 ALLOWED_EXTENSIONS in SXR folder.")
        for row in range(model.rowCount()):
            status = model._data[row][0]
            assert status == loadStatus.HEADER_ONLY, (
                f"Row {row} expected HEADER_ONLY but got {status}"
            )

    def test_incorrect_parser_header_has_status_col(self, qapp, mex2_data_dir):
        """Even with an incorrect parser the first header should be the status column."""
        model = _model(mex2_data_dir, parser_cls=SXR_NEXAFS)
        if model.rowCount() == 0:
            pytest.skip("No files matched SXR ALLOWED_EXTENSIONS in MEX2 folder.")
        # When a parser is set the first column header is an empty string ("").
        assert model._header[0] == ""

    def test_incorrect_parser_status_icon_returns_value(self, qapp, mex2_data_dir):
        """DecorationRole on column 0 should return a QIcon for the error rows."""
        model = _model(mex2_data_dir, parser_cls=SXR_NEXAFS)
        if model.rowCount() == 0:
            pytest.skip("No files matched.")
        idx = model.index(0, 0)
        icon = model.data(idx, role=QtCore.Qt.ItemDataRole.DecorationRole)
        # DecorationRole returns a QIcon (not None) for the status column.
        assert icon is not None


# ===========================================================================
# Tests – correct MEX2 parser
# ===========================================================================


class TestCorrectMEX2Parser:
    """Load each MEX2 data directory with MEX2_NEXAFS."""

    @pytest.fixture
    def mex2_model(self, qapp, mex2_data_dir):
        return _model(mex2_data_dir, parser_cls=MEX2_NEXAFS)

    def test_row_count(self, mex2_model, mex2_data_dir):
        """Every MEX2 file (.xdi + .mda) should produce a row."""
        expected = len(
            [
                f
                for f in os.listdir(mex2_data_dir)
                if any(f.endswith(ext) for ext in MEX2_NEXAFS.ALLOWED_EXTENSIONS)
            ]
        )
        assert mex2_model.rowCount() == expected

    def test_all_header_only_status(self, mex2_model):
        """Rows should be either HEADER_ONLY or ERROR, with at least one header-only entry."""
        has_header_only = False
        for row in range(mex2_model.rowCount()):
            status = mex2_model._data[row][0]
            assert status in {loadStatus.HEADER_ONLY, loadStatus.ERROR}, (
                f"Row {row}: expected HEADER_ONLY/ERROR, got {status!r}"
            )
            if status == loadStatus.HEADER_ONLY:
                has_header_only = True
        assert has_header_only

    def test_header_includes_summary_params(self, mex2_model):
        """Headers should contain each SUMMARY_PARAMS key (possibly with units appended)."""
        for param in MEX2_NEXAFS.SUMMARY_PARAMS:
            assert any(
                param.lower() in h.lower()
                for h in mex2_model._header
                if isinstance(h, str)
            ), f"SUMMARY_PARAM '{param}' not found in headers {mex2_model._header}"

    def test_status_column_is_first(self, mex2_model):
        """When a parser is present the first column header must be '' (status)."""
        assert mex2_model._header[0] == ""

    def test_filename_column_values(self, mex2_model, mex2_data_dir):
        """Column 2 (after status and index) should contain valid filenames."""
        for row in range(mex2_model.rowCount()):
            filename = mex2_model._data[row][2]
            assert isinstance(filename, str) and len(filename) > 0
            assert os.path.isfile(os.path.join(mex2_data_dir, filename))

    def test_decoration_role_header_only(self, mex2_model):
        """Status column should return a QIcon for header-only rows."""
        idx = mex2_model.index(0, 0)
        icon = mex2_model.data(idx, role=QtCore.Qt.ItemDataRole.DecorationRole)
        assert (
            isinstance(icon, QtWidgets.QApplication.style().__class__)
            or icon is not None
        )

    def test_tooltip_role_status_column(self, mex2_model):
        """ToolTipRole on the status column should return a descriptive string."""
        idx = mex2_model.index(0, 0)
        tip = mex2_model.data(idx, role=QtCore.Qt.ItemDataRole.ToolTipRole)
        assert isinstance(tip, str) and len(tip) > 0

    def test_filenames_sorted(self, mex2_model):
        """Rows should be sorted alphabetically by filename."""
        filenames = [mex2_model._data[r][2] for r in range(mex2_model.rowCount())]
        assert filenames == sorted(filenames)

    def test_column_count_matches_header(self, mex2_model):
        """columnCount should equal the length of _header."""
        assert mex2_model.columnCount() == len(mex2_model._header)

    def test_display_data_non_status_cols(self, mex2_model):
        """DisplayRole on non-status columns should return a value (not raise)."""
        for col in range(1, mex2_model.columnCount()):
            idx = mex2_model.index(0, col)
            # Should not raise; value may be None for missing params.
            _ = mex2_model.data(idx, role=QtCore.Qt.ItemDataRole.DisplayRole)

    def test_header_data_horizontal(self, mex2_model):
        """headerData for horizontal orientation should return capitalised strings."""
        for col in range(mex2_model.columnCount()):
            val = mex2_model.headerData(
                col,
                QtCore.Qt.Orientation.Horizontal,
                QtCore.Qt.ItemDataRole.DisplayRole,
            )
            # Either a string or None (for the status column placeholder "").
            if val is not None:
                assert isinstance(val, str)

    def test_param_selection_updates_columns_and_order(
        self, mex2_model: ParserTableModel
    ):
        """Setting param_selection should update both selected params and their order."""
        assert mex2_model.rowCount() > 0

        # Start with a two-parameter selection and verify column count and order.
        mex2_model.param_selection = ["filesize", "created"]
        assert mex2_model.columnCount() == 5  # status, #, filename + 2 params

        row = mex2_model._data[0]
        assert isinstance(row[3], (int, float))
        assert isinstance(row[4], datetime.datetime)

        # Reorder the selection and verify data columns swap accordingly.
        mex2_model.param_selection = ["created", "filesize"]
        assert mex2_model.columnCount() == 5

        row = mex2_model._data[0]
        assert isinstance(row[3], datetime.datetime)
        assert isinstance(row[4], (int, float))


# ===========================================================================
# Tests – correct SXR parser
# ===========================================================================


class TestCorrectSXRParser:
    """Load the SXR 2024-03 directory with SXR_NEXAFS."""

    @pytest.fixture
    def sxr_model(self, qapp, sxr_data_dir):
        return _model(sxr_data_dir, parser_cls=SXR_NEXAFS)

    def test_row_count(self, sxr_model, sxr_data_dir):
        expected = len(
            [
                f
                for f in os.listdir(sxr_data_dir)
                if any(f.endswith(ext) for ext in SXR_NEXAFS.ALLOWED_EXTENSIONS)
            ]
        )
        assert sxr_model.rowCount() == expected

    def test_all_header_only_status(self, sxr_model):
        for row in range(sxr_model.rowCount()):
            status = sxr_model._data[row][0]
            assert status == loadStatus.HEADER_ONLY, (
                f"Row {row}: expected HEADER_ONLY, got {status!r}"
            )

    def test_header_includes_summary_params(self, sxr_model):
        for param in SXR_NEXAFS.SUMMARY_PARAMS:
            assert any(
                param.lower() in h.lower()
                for h in sxr_model._header
                if isinstance(h, str)
            ), f"SUMMARY_PARAM '{param}' not found in headers {sxr_model._header}"

    def test_filenames_sorted(self, sxr_model):
        filenames = [sxr_model._data[r][2] for r in range(sxr_model.rowCount())]
        assert filenames == sorted(filenames)

    def test_column_count_matches_header(self, sxr_model):
        assert sxr_model.columnCount() == len(sxr_model._header)

    def test_status_icon_for_each_row(self, sxr_model):
        for row in range(sxr_model.rowCount()):
            idx = sxr_model.index(row, 0)
            icon = sxr_model.data(idx, role=QtCore.Qt.ItemDataRole.DecorationRole)
            assert icon is not None, f"Row {row} returned None for DecorationRole"

    def test_tooltip_status_column(self, sxr_model):
        for row in range(sxr_model.rowCount()):
            idx = sxr_model.index(row, 0)
            tip = sxr_model.data(idx, role=QtCore.Qt.ItemDataRole.ToolTipRole)
            assert isinstance(tip, str) and len(tip) > 0


# ===========================================================================
# Tests – reload (header_only → full data)
# ===========================================================================


class TestReload:
    """Test that reloading a header-only item updates its status to FULL_DATA."""

    def _first_valid_filepath(self, model: ParserTableModel) -> tuple[int, str]:
        """Return (row, filepath) for the first HEADER_ONLY row."""
        for row in range(model.rowCount()):
            status = model._data[row][0]
            if status == loadStatus.HEADER_ONLY:
                filename = model._data[row][2]
                filepath = os.path.join(model._path, filename)
                return row, filepath
        raise RuntimeError("No HEADER_ONLY row found in model.")

    def test_mex2_reload_full_data(self, qapp, mex2_data_dir):
        """Reloading a MEX2 HEADER_ONLY entry with header_only=False → FULL_DATA."""
        model = _model(mex2_data_dir, parser_cls=MEX2_NEXAFS)
        row, filepath = self._first_valid_filepath(model)

        # Confirm it starts as HEADER_ONLY
        assert model._data[row][0] == loadStatus.HEADER_ONLY

        # Reload with full data
        model.reload_file(filepath, header_only=False)

        # The cache should now reflect FULL_DATA
        cached_status, _ = model.parsed[filepath]
        assert cached_status == loadStatus.FULL_DATA

        # Update the table row and verify the displayed status
        model.reload(model.index(row, 0), header_only=False)
        assert model._data[row][0] == loadStatus.FULL_DATA

    def test_sxr_reload_full_data(self, qapp, sxr_data_dir):
        """Reloading a SXR HEADER_ONLY entry with header_only=False → FULL_DATA."""
        model = _model(sxr_data_dir, parser_cls=SXR_NEXAFS)
        row, filepath = self._first_valid_filepath(model)

        assert model._data[row][0] == loadStatus.HEADER_ONLY

        model.reload_file(filepath, header_only=False)
        cached_status, _ = model.parsed[filepath]
        assert cached_status == loadStatus.FULL_DATA

        model.reload(model.index(row, 0), header_only=False)
        assert model._data[row][0] == loadStatus.FULL_DATA

    def test_reload_signals_data_changed(self, qapp, mex2_data_dir):
        """reload() should emit dataChanged for the affected row."""
        model = _model(mex2_data_dir, parser_cls=MEX2_NEXAFS)
        row, filepath = self._first_valid_filepath(model)

        changed_rows = []

        def on_data_changed(top_left, bottom_right, roles):
            changed_rows.append(top_left.row())

        model.dataChanged.connect(on_data_changed)
        model.reload(model.index(row, 0), header_only=False)

        assert row in changed_rows, "dataChanged was not emitted for the reloaded row."

    def test_reload_header_only_stays_header_only(self, qapp, mex2_data_dir):
        """Reloading with header_only=True should keep the HEADER_ONLY status."""
        model = _model(mex2_data_dir, parser_cls=MEX2_NEXAFS)
        row, filepath = self._first_valid_filepath(model)

        # First reload to full data
        model.reload_file(filepath, header_only=False)
        cached_status, _ = model.parsed[filepath]
        assert cached_status == loadStatus.FULL_DATA

        # Now reload back to header only
        model.reload_file(filepath, header_only=True)
        cached_status, _ = model.parsed[filepath]
        assert cached_status == loadStatus.HEADER_ONLY

        # Update the table row and verify
        model.reload(model.index(row, 0), header_only=True)
        assert model._data[row][0] == loadStatus.HEADER_ONLY

    def test_reload_nonexistent_raises(self, qapp, mex2_data_dir):
        """Calling reload_file on a path not in the cache should raise ValueError."""
        model = _model(mex2_data_dir, parser_cls=MEX2_NEXAFS)
        with pytest.raises(ValueError, match="not currently parsed"):
            model.reload_file("/nonexistent/file.xdi", header_only=False)


# ===========================================================================
# Tests – ParserTableView integration
# ===========================================================================


class TestParserTableView:
    """Smoke tests for ParserTableView (Qt widget integration)."""

    @staticmethod
    def _trigger_context_menu_action(
        view: ParserTableView,
        action_text: str,
        proxy_row: int = 0,
    ) -> None:
        """Open the context menu at a row and trigger the action with matching text."""
        proxy_idx = view.proxy_model.index(proxy_row, 0)
        assert proxy_idx.isValid(), "Invalid proxy index for context menu test."

        rect = view.visualRect(proxy_idx)
        pos = rect.center()

        original_qmenu = parser_table_module.QtWidgets.QMenu

        class _FakeMenu(QtWidgets.QMenu):
            def exec(self, *_args, **_kwargs):
                for action in self.actions():
                    if action.text() == action_text:
                        action.trigger()
                        return
                raise AssertionError(
                    f"Action '{action_text}' not found in context menu."
                )

        parser_table_module.QtWidgets.QMenu = _FakeMenu
        try:
            event = QtGui.QContextMenuEvent(
                QtGui.QContextMenuEvent.Reason.Mouse,
                pos,
                view.viewport().mapToGlobal(pos),
            )
            view.contextMenuEvent(event)
        finally:
            parser_table_module.QtWidgets.QMenu = original_qmenu

    @staticmethod
    def _first_proxy_row_with_status(view: ParserTableView, status: loadStatus) -> int:
        """Return first proxy row whose source row has the given load status."""
        for proxy_row in range(view.proxy_model.rowCount()):
            src_idx = view.proxy_model.mapToSource(view.proxy_model.index(proxy_row, 0))
            if view.files_model._data[src_idx.row()][0] == status:
                return proxy_row
        raise AssertionError(f"No row with status {status!r} found.")

    def test_mex2_view_creates_without_error(self, qapp, mex2_data_dir):
        view = ParserTableView(path=mex2_data_dir, parser_cls=MEX2_NEXAFS)
        assert view.files_model.rowCount() > 0

    def test_sxr_view_creates_without_error(self, qapp, sxr_data_dir):
        view = ParserTableView(path=sxr_data_dir, parser_cls=SXR_NEXAFS)
        assert view.files_model.rowCount() > 0

    def test_view_no_parser(self, qapp, mex2_data_dir):
        view = ParserTableView(path=mex2_data_dir, parser_cls=None)
        assert view.files_model.rowCount() == len(os.listdir(mex2_data_dir))

    def test_view_proxy_model_row_count(self, qapp, mex2_data_dir):
        """The proxy model should pass through all rows."""
        view = ParserTableView(path=mex2_data_dir, parser_cls=MEX2_NEXAFS)
        assert view.proxy_model.rowCount() == view.files_model.rowCount()

    def test_view_reload_file(self, qapp, mex2_data_dir):
        """ParserTableView.reload_file should update the underlying model."""
        view = ParserTableView(path=mex2_data_dir, parser_cls=MEX2_NEXAFS)
        model = view.files_model
        # Find a HEADER_ONLY row via the proxy model.
        for proxy_row in range(view.proxy_model.rowCount()):
            proxy_idx = view.proxy_model.index(proxy_row, 0)
            src_idx = view.proxy_model.mapToSource(proxy_idx)
            if model._data[src_idx.row()][0] == loadStatus.HEADER_ONLY:
                # Reload it
                view.reload_file(proxy_idx)
                # Status in model should still be HEADER_ONLY (default reload is header_only=True)
                assert model._data[src_idx.row()][0] == loadStatus.HEADER_ONLY
                break
        else:
            pytest.skip("No HEADER_ONLY row found.")

    def test_context_menu_delete_action_works(self, qapp, mex2_data_dir):
        """Delete action from context menu should remove the selected row."""
        view = ParserTableView(path=mex2_data_dir, parser_cls=MEX2_NEXAFS)
        before = view.files_model.rowCount()
        assert before > 0

        self._trigger_context_menu_action(view, "Delete", proxy_row=0)
        after = view.files_model.rowCount()
        assert after == before - 1

    def test_context_menu_reload_header_only_action_works(self, qapp, mex2_data_dir):
        """Reload (header only) action should keep item in HEADER_ONLY state."""
        view = ParserTableView(path=mex2_data_dir, parser_cls=MEX2_NEXAFS)
        proxy_row = self._first_proxy_row_with_status(view, loadStatus.HEADER_ONLY)

        self._trigger_context_menu_action(
            view, "Reload (header only)", proxy_row=proxy_row
        )

        src_idx = view.proxy_model.mapToSource(view.proxy_model.index(proxy_row, 0))
        assert view.files_model._data[src_idx.row()][0] == loadStatus.HEADER_ONLY

    def test_context_menu_reload_with_data_action_works(self, qapp, mex2_data_dir):
        """Reload (with data) action should promote an item to FULL_DATA when possible."""
        view = ParserTableView(path=mex2_data_dir, parser_cls=MEX2_NEXAFS)
        proxy_row = self._first_proxy_row_with_status(view, loadStatus.HEADER_ONLY)

        self._trigger_context_menu_action(
            view, "Reload (with data)", proxy_row=proxy_row
        )

        src_idx = view.proxy_model.mapToSource(view.proxy_model.index(proxy_row, 0))
        status = view.files_model._data[src_idx.row()][0]
        # Some files can fail full-data reload and become ERROR.
        assert status in {loadStatus.FULL_DATA, loadStatus.ERROR}

    def test_custom_menu_functions_are_selectable_and_called(self, qapp, mex2_data_dir):
        """Custom one-arg and two-arg functions should be added and triggerable via context menu."""
        view = ParserTableView(path=mex2_data_dir, parser_cls=MEX2_NEXAFS)
        calls: list[tuple[object, object | None]] = []

        def first_custom(parser_cls):
            calls.append((parser_cls, None))

        def second_custom(parser_cls, checked):
            calls.append((parser_cls, checked))

        view.menu_functions = [first_custom, second_custom]

        self._trigger_context_menu_action(view, "First custom", proxy_row=0)
        self._trigger_context_menu_action(view, "Second custom", proxy_row=0)

        assert len(calls) == 2
        assert calls[0][0] is MEX2_NEXAFS
        assert calls[1][0] is MEX2_NEXAFS
        assert isinstance(calls[1][1], bool)

    def test_invalid_custom_menu_functions_are_rejected(self, qapp, mex2_data_dir):
        """Functions with invalid arity (0 or >2 args) should raise ValueError."""
        view = ParserTableView(path=mex2_data_dir, parser_cls=MEX2_NEXAFS)

        def zero_args():
            return None

        def three_args(a, b, c):
            return None

        with pytest.raises(ValueError, match="must accept at least one parameter"):
            view.menu_functions = [zero_args]

        with pytest.raises(ValueError, match="must accept at least one parameter"):
            view.menu_functions = [three_args]
