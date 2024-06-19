import os
from PyQt6 import QtWidgets, QtCore, QtGui


class directory_selector(QtWidgets.QHBoxLayout):
    """
    Module for selecting a directory path.

    Parameters
    ----------
    QWidget : _type_
        _description_
    """

    newPath = QtCore.pyqtSignal(bool)

    # String constants for widget elements
    edit_description = "Path to load NEXAFS data from."
    dialog_caption = "Select NEXAFS Directory"

    def __init__(self, parent=None):
        super().__init__(parent)

        # Instance attributes
        self.folder_path = os.path.expanduser("~")  # default use home path.
        self.folder_path = directory_selector.format_path(
            self.folder_path
        )  # adds final slash if not present.

        ### Instance widgets
        # Folder path
        self.folder_path_edit = QtWidgets.QLineEdit()
        self.folder_path_edit.setText(self.folder_path)
        self.folder_path_edit.accessibleName = "Directory"
        self.folder_path_edit.accessibleDescription = self.edit_description
        self.folder_path_edit.editingFinished.connect(self.validate_path)
        self.folder_path_edit_default_stylesheet = self.folder_path_edit.styleSheet()
        # Folder select button
        self.folder_select_button = QtWidgets.QPushButton("Browse")
        self.folder_select_button.clicked.connect(self.select_path)

        # Setup layout
        self.addWidget(self.folder_path_edit)
        self.addWidget(self.folder_select_button)

    def select_path(self):
        """
        Generates a file dialog to select a directory, then updates the path.
        """
        path = QtWidgets.QFileDialog.getExistingDirectory(
            parent=None,
            caption=self.dialog_caption,
            directory=self.folder_path,
            options=QtWidgets.QFileDialog.Option.ShowDirsOnly,
        )
        path = None if path == "" else path
        if path is None:
            # invalid path
            self.folder_path_edit.setText(
                self.folder_path
            )  # reset the text to the original path.
        else:
            # set editable path and validate.
            self.folder_path_edit.setText(path)
            self.validate_path()

    def validate_path(self):
        """
        Validate a manual entry path.

        Only updates internal path if the path is valid, otherwise sets background color to red.
        """
        # Get path text
        editable_path = directory_selector.format_path(self.folder_path_edit.text())

        # Check validity
        if editable_path is None:
            # invalid path
            self.folder_path_edit.setText(
                self.folder_path
            )  # reset the text to the original path.
            return

        if not os.path.isdir(editable_path):
            # invalid path
            self.folder_path_edit.setStyleSheet("background-color: red;")
        else:
            # Valid path

            # Check if path has changed
            new_path = False
            if editable_path != self.folder_path:
                # if path has changed, perform extra functions...
                new_path = True
            self.folder_path_edit.setStyleSheet(
                self.folder_path_edit_default_stylesheet
            )
            self.folder_path = editable_path
            self.folder_path_edit.setText(editable_path)

            if new_path:
                self.newPath.emit(True)

    @staticmethod
    def format_path(path: str) -> str:
        """
        Formats a path string to be consistent.

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
