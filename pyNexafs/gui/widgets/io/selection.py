"""
Folder selection for NEXAFS data processing.

This module contains the `directorySelector` class.
"""

import os

# Import settings from the parent package to determine which Qt wrapper to use for signals.
from pyNexafs.gui.config import QtCore, QtWidgets, QtGui

# Extras for __main__ testing
import sys


class directorySelector(QtWidgets.QWidget):
    """
    Module for selecting a directory path.

    Highlights the text

    Attributes
    ----------
    new_path : QtCore.pyqtSignal
        Generates a signal when a new path entry is updated.
    path : str
        The absolute filepath.

    Parameters
    ----------
    """

    # Signals
    # new_path = QtCore.pyqtSignal(bool) # QtPy, PyQt6
    new_path = QtCore.Signal(bool)  # PySide6

    # Constants
    _EDIT_DESC = "Path to load NEXAFS data from."
    """The edit description for the directory path."""
    _DIAG_CAP = "Select NEXAFS Directory"
    """The dialog caption."""
    _STYLESHEET_INVALID = "background-color: red;"
    """The stylesheet to apply to the path edit when an invalid path is entered."""

    def __init__(
        self, parent: QtWidgets.QWidget | None = None, init_path: str | None = None
    ):
        super().__init__(parent)

        # Layout
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)

        # Update margins to be compact if a parent is provided.
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            layout.setContentsMargins(0, 0, 0, 0)

        # Instance attributes
        if init_path is not None:
            if not os.path.exists(init_path):
                raise ValueError(f"Provided initial path `{init_path}` does not exist.")
            self._path = os.path.abspath(init_path)
        else:
            self._path = os.path.expanduser("~")  # default use home path.
        """The last valid directory selection."""

        # Update the path for formatting
        self._path = directorySelector._format_path(self._path)

        ### Instance widgets
        # Refresh button / toggle
        refresh_icon = QtGui.QIcon.fromTheme("view-refresh")
        self.refresh_btn = QtWidgets.QPushButton(refresh_icon, "")

        # Folder path
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setText(self._display_path(self._path, editing=False))
        self.path_edit.setAccessibleName("Directory")
        self.path_edit.setAccessibleDescription(self._EDIT_DESC)
        self.path_edit.installEventFilter(self)

        # Folder select button
        self.path_select_btn = QtWidgets.QPushButton("Browse")
        self.path_select_btn.clicked.connect(self.select_path)

        # Setup layout
        layout.addWidget(self.path_edit)
        layout.addWidget(self.path_select_btn)
        layout.addWidget(self.refresh_btn)

        # Collect Defaults
        self._path_edit_default_stylesheet = self.path_edit.styleSheet()

        # Connections
        self.path_edit.editingFinished.connect(self.validate_edit_path)
        self.refresh_btn.clicked.connect(self.refresh)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Show full path only while path edit is actively being edited."""
        if obj is self.path_edit:
            if event.type() == QtCore.QEvent.Type.FocusIn:
                self.path_edit.setText(self._display_path(self._path, editing=True))
                self.path_edit.setCursorPosition(len(self.path_edit.text()))
            elif event.type() == QtCore.QEvent.Type.FocusOut:
                self.path_edit.setText(self._display_path(self._path, editing=False))
                self.path_edit.setCursorPosition(0)
                # The displayed value is the last known-valid path in compact form,
                # so clear any transient invalid-entry styling on defocus.
                self.path_edit.setStyleSheet(self._path_edit_default_stylesheet)
        return super().eventFilter(obj, event)

    @property
    def path(self) -> str:
        """
        The absolute filepath.

        Parameters
        ----------
        filepath : str
            The filepath to set.

        Returns
        -------
        str
            The absolute filepath.
        """
        return os.path.abspath(self._path)

    @path.setter
    def path(self, filepath: str):
        if not os.path.exists(filepath):
            raise ValueError(f"Provided path `{filepath}` does not exist.")
        self._path = os.path.abspath(filepath)

    @path.deleter
    def path(self):
        self._path = os.path.expanduser("~")  # return to default home path.

    @staticmethod
    def validate_path(path: str) -> bool:
        """
        Validate a path string.

        Parameters
        ----------
        path : str
            The path string to validate.

        Returns
        -------
        bool
            True if the path is valid, False otherwise.
        """
        return os.path.isdir(path)

    def validate_edit_path(self):
        """
        Validate the path in the edit box.

        Updates if valid. Otherwise reset entry to the last valid path.
        """

        compact_display_path = self._display_path(self._path, editing=False)
        entered_text = self.path_edit.text()

        # If the line edit currently shows the compact display value,
        # do not re-validate it as a filesystem path.
        if entered_text == compact_display_path:
            self.path_edit.setText(compact_display_path)
            return

        path = self._format_path(entered_text)

        if path and directorySelector.validate_path(path):
            # Valid path, update the path and emit a change signal.
            self.path = path
            # self.new_path.emit(True)
            self.new_path.emit(True)  # Emit signal to trigger a data refresh.
            self.path_edit.setStyleSheet(self._path_edit_default_stylesheet)
            self.path_edit.setText(self._display_path(self._path, editing=False))
        else:
            # Invalid path, reset the text to the original path.
            self.path_edit.setText(self._display_path(self._path, editing=False))
            self.path_edit.setStyleSheet(self._STYLESHEET_INVALID)

    @staticmethod
    def _format_path(path: str) -> str:
        """
        Format a path string to be consistent.

        Parameters
        ----------
        path : str
            The incoming path string. Must be a valid path, tested by os.path.isdir().

        Returns
        -------
        str
            A formatted path string that always contains a trailing slash, with slashes matching OS type.
        """

        # Strip whitespace, ensure tailing slash, and convert mixed slashes to forward slashes.
        formatted_path = os.path.join("", path.strip(), "").replace("\\", "/")
        # Remove redundant slash duplicates
        while "//" in formatted_path:
            formatted_path = formatted_path.replace("//", "/")
        # Convert slashes to match OS
        slashes = os.sep
        formatted_path = formatted_path.replace("/", slashes)
        # Add final slash if not present.
        if not formatted_path.endswith(slashes):
            formatted_path += slashes
        return formatted_path

    @staticmethod
    def _display_path(path: str, editing: bool) -> str:
        """Return full path while editing, else a compact drive/.../tail form."""
        full_path = directorySelector._format_path(path)
        if editing:
            return full_path

        trimmed = full_path.rstrip(os.sep)
        if trimmed == "":
            return full_path

        drive, path_without_drive = os.path.splitdrive(trimmed)
        # Keep only the final 14 chars of the path tail (excluding drive),
        # preserving separators where possible.
        tail_source = path_without_drive.lstrip(os.sep)
        tail = tail_source if len(tail_source) <= 14 else tail_source[-14:]
        tail_display = f"{tail}{os.sep}"

        if drive:
            return f"{drive}{os.sep}...{os.sep}{tail_display}"
        if full_path.startswith(os.sep):
            return f"{os.sep}...{os.sep}{tail_display}"
        return f"...{os.sep}{tail_display}"

    def select_path(self) -> None:
        """
        UI prompt to select a directory and update the path.
        """
        path = QtWidgets.QFileDialog.getExistingDirectory(
            parent=None,
            caption=self._DIAG_CAP,
            dir=self._path,
            options=QtWidgets.QFileDialog.Option.ShowDirsOnly,
        )
        path = None if path == "" else path
        if path is None:
            # invalid path
            self.path_edit.setText(self._path)  # reset the text to the original path.
        else:
            # set editable path and validate.
            self.path_edit.setText(path)
            self.validate_edit_path()

    def refresh(self):
        """
        Refresh button callback to trigger a new_path signal without changing the path.

        Useful for triggering a data refresh when the directory contents have changed,
        but the path has not.
        """
        self.new_path.emit(True)


if __name__ == "__main__":
    QApplication = QtWidgets.QApplication
    app = QApplication(sys.argv)
    window = directorySelector()
    window.show()
    window.setWindowTitle("Directory Selector")
    sys.exit(app.exec())
