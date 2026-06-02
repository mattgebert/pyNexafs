"""Composable UI around ParserTableView.

Includes:
- directory path selection via directorySelector
- parser selection from parser_registry
- filetype selection based on parser ALLOWED_EXTENSIONS
- free-text filtering applied on editing finished
"""

from __future__ import annotations

import os

from pyNexafs.gui.config import QtWidgets
from pyNexafs.gui.widgets.io.selection import directorySelector
from pyNexafs.gui.widgets.io.parser_ui.parser_table import ParserTableView
from pyNexafs.parsers import parser_registry, parserBase


class ParserTableViewerWidget(QtWidgets.QWidget):
    """Container widget that controls a ParserTableView instance."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        init_path: str | None = None,
        init_parser: type[parserBase] | None = None,
    ):
        super().__init__(parent=parent)

        self.path_selector = directorySelector(parent=self, init_path=init_path)
        self.parser_selector = QtWidgets.QComboBox(self)
        self.filetype_selector = QtWidgets.QComboBox(self)
        self.text_filter_edit = QtWidgets.QLineEdit(self)
        self.text_filter_edit.setPlaceholderText("Filter rows (applied on edit finish)")
        self.show_cached_toggle = QtWidgets.QPushButton("Show cached", self)
        self.show_cached_toggle.setCheckable(True)
        self.show_cached_toggle.setChecked(False)

        self.table_view = ParserTableView(
            path=self.path_selector.path,
            parser_cls=None,
            parent=self,
        )

        self._build_layout()
        self._populate_parser_selector()

        # Default parser selection.
        if init_parser is not None:
            init_parser_name = init_parser.__name__
            for i in range(self.parser_selector.count()):
                parser_cls = self.parser_selector.itemData(i)
                if (
                    isinstance(parser_cls, type)
                    and parser_cls.__name__ == init_parser_name
                ):
                    self.parser_selector.setCurrentIndex(i)
                    break

        self._apply_parser_selection()
        self._connect_signals()
        self._update_show_cached_tooltip()

    def _build_layout(self) -> None:
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(self.path_selector)

        parser_row = QtWidgets.QHBoxLayout()
        parser_row.addWidget(QtWidgets.QLabel("Parser:", self))
        parser_row.addWidget(self.parser_selector, stretch=2)
        parser_row.addWidget(QtWidgets.QLabel("Filetype:", self))
        parser_row.addWidget(self.filetype_selector, stretch=1)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Text filter:", self))
        filter_row.addWidget(self.text_filter_edit, stretch=1)
        filter_row.addWidget(self.show_cached_toggle)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(path_row)
        layout.addLayout(parser_row)
        layout.addLayout(filter_row)
        layout.addWidget(self.table_view)

    def _connect_signals(self) -> None:
        self.path_selector.new_path.connect(self._on_path_changed)
        self.parser_selector.currentIndexChanged.connect(self._on_parser_changed)
        self.filetype_selector.currentIndexChanged.connect(self._on_filetype_changed)
        self.text_filter_edit.editingFinished.connect(
            self._on_text_filter_editing_finished
        )
        self.show_cached_toggle.toggled.connect(self._on_show_cached_toggled)
        self.table_view.files_model.modelReset.connect(self._update_show_cached_tooltip)

    def _populate_parser_selector(self) -> None:
        self.parser_selector.clear()
        self.parser_selector.addItem("None", None)
        for parser_name, parser_cls in parser_registry.items():
            self.parser_selector.addItem(parser_name, parser_cls)

    def _populate_filetype_selector(self, parser_cls: type[parserBase] | None) -> None:
        previous_ext = self.current_filetype_filter
        self.filetype_selector.blockSignals(True)
        self.filetype_selector.clear()
        self.filetype_selector.addItem("All", None)

        if parser_cls is not None:
            for ext in parser_cls.ALLOWED_EXTENSIONS:
                self.filetype_selector.addItem(ext, ext)

        if previous_ext is not None:
            idx = self.filetype_selector.findData(previous_ext)
            if idx >= 0:
                self.filetype_selector.setCurrentIndex(idx)

        self.filetype_selector.blockSignals(False)

    @property
    def current_parser(self) -> type[parserBase] | None:
        value = self.parser_selector.currentData()
        return value if isinstance(value, type) else None

    @property
    def current_filetype_filter(self) -> str | None:
        value = self.filetype_selector.currentData()
        return value if isinstance(value, str) else None

    @property
    def current_text_filter(self) -> str | None:
        text = self.text_filter_edit.text().strip()
        return text if text != "" else None

    def _apply_parser_selection(self) -> None:
        parser_cls = self.current_parser
        model = self.table_view.files_model
        model.parser = parser_cls
        model.load_path()
        self._populate_filetype_selector(parser_cls)
        self._apply_filters()
        self._update_show_cached_tooltip()

    def _apply_filters(self) -> None:
        filetype = self.current_filetype_filter
        filetype_filters = [filetype] if filetype is not None else None
        self.table_view.filters = (self.current_text_filter, filetype_filters)
        self._update_show_cached_tooltip()

    def _update_show_cached_tooltip(self) -> None:
        model = self.table_view.files_model
        current_path = os.path.normpath(os.path.abspath(model.path))
        cached_outside_path = sum(
            1
            for filepath in model.parsed.keys()
            if os.path.normpath(os.path.abspath(os.path.dirname(filepath)))
            != current_path
        )
        self.show_cached_toggle.setToolTip(
            "Include cached files in the table. "
            f"{cached_outside_path} cached file(s) are outside the current path."
        )

    def _on_path_changed(self, _changed: bool) -> None:
        self.table_view.files_model.path = self.path_selector.path
        self._apply_filters()
        self._update_show_cached_tooltip()

    def _on_parser_changed(self, _index: int) -> None:
        self._apply_parser_selection()

    def _on_filetype_changed(self, _index: int) -> None:
        self._apply_filters()

    def _on_text_filter_editing_finished(self) -> None:
        self._apply_filters()

    def _on_show_cached_toggled(self, checked: bool) -> None:
        self.table_view.files_model.show_cached = checked
        self._update_show_cached_tooltip()


if __name__ == "__main__":
    import sys
    from pyNexafs.parsers.au.aus_sync import SXR_NEXAFS

    app = QtWidgets.QApplication(sys.argv)
    # Get this filepath, and use reldir to get to the test data
    cwd = os.path.dirname(os.path.abspath(__file__))
    rel_path = os.path.join(
        cwd, "..", "..", "..", "..", "..", "tests", "test_data", "au", "SXR", "2024-03"
    )
    test_path = os.path.normpath(rel_path)
    print(f"Testing path: {test_path}, exists: {os.path.exists(test_path)}")
    if os.path.exists(test_path):
        init_path = test_path
        init_parser = SXR_NEXAFS
    else:
        init_path = None
        init_parser = None

    widget = ParserTableViewerWidget(
        init_path=init_path,
        init_parser=init_parser,
    )
    widget.setWindowTitle("Parser Table Viewer")
    widget.resize(1100, 650)
    widget.show()

    sys.exit(app.exec())
