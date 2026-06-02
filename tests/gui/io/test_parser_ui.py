"""Tests for the parser table composite UI widget."""

import os
import pytest

from pyNexafs.parsers import parser_registry
from pyNexafs.parsers.au.aus_sync import SXR_NEXAFS
from pyNexafs.gui.widgets.io.parser_ui import ParserTableViewerWidget

pytest_plugins = ["tests.gui.fixtures"]


@pytest.fixture
def sxr_data_dir(sxr_dir):
    """Return SXR 2024-03 folder path for parser UI tests."""
    return os.path.join(sxr_dir, "2024-03")


class TestParserTableViewerWidget:
    """Integration tests for ParserTableViewerWidget controls and table wiring."""

    def _index_for_parser(
        self, widget: ParserTableViewerWidget, parser_cls: type
    ) -> int:
        for i in range(widget.parser_selector.count()):
            if widget.parser_selector.itemData(i) is parser_cls:
                return i
        raise AssertionError(f"Parser {parser_cls.__name__} not found in selector.")

    def _index_for_filetype(self, widget: ParserTableViewerWidget, ext: str) -> int:
        for i in range(widget.filetype_selector.count()):
            if widget.filetype_selector.itemData(i) == ext:
                return i
        raise AssertionError(f"Filetype {ext} not found in selector.")

    def test_parser_selector_lists_registered_parsers(self, qapp, sxr_data_dir):
        widget = ParserTableViewerWidget(init_path=sxr_data_dir)

        names = [
            widget.parser_selector.itemText(i)
            for i in range(widget.parser_selector.count())
        ]
        assert names[0] == "None"
        for parser_name in parser_registry.keys():
            assert parser_name in names

    def test_sxr_filters_update_table_on_edit_finish(self, qapp, sxr_data_dir):
        widget = ParserTableViewerWidget(init_path=sxr_data_dir)

        parser_index = self._index_for_parser(widget, SXR_NEXAFS)
        widget.parser_selector.setCurrentIndex(parser_index)

        mda_index = self._index_for_filetype(widget, ".mda")
        widget.filetype_selector.setCurrentIndex(mda_index)

        widget.text_filter_edit.setText("90")
        widget.text_filter_edit.editingFinished.emit()

        assert widget.table_view.proxy_model.rowCount() == 2

    def test_init_parser_sets_parser_selection(self, qapp, sxr_data_dir):
        widget = ParserTableViewerWidget(init_path=sxr_data_dir, init_parser=SXR_NEXAFS)

        assert widget.current_parser is SXR_NEXAFS

    def test_sxr_delete_then_refresh_restores_rows(self, qapp, sxr_data_dir):
        widget = ParserTableViewerWidget(init_path=sxr_data_dir, init_parser=SXR_NEXAFS)

        assert widget.table_view.proxy_model.rowCount() == 8

        proxy_idx = widget.table_view.proxy_model.index(0, 0)
        assert proxy_idx.isValid()
        source_idx = widget.table_view.proxy_model.mapToSource(proxy_idx)
        widget.table_view.files_model.delete(source_idx)

        assert widget.table_view.proxy_model.rowCount() == 7

        widget.path_selector.refresh_btn.click()

        assert widget.table_view.proxy_model.rowCount() == 8
