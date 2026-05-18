"""Tests for the directorySelector widget.

This module tests the directorySelector class from pyNexafs.gui.widgets.io.selection,
including path validation, formatting, and signal emission.
"""

import os
import pytest

# Import Qt components - must be before importing the widget
from pyNexafs.gui.config import QtCore, QtWidgets

# Import the widget under test
from pyNexafs.gui.widgets.io.selection import directorySelector

# Load shared GUI fixtures for this module.
pytest_plugins = ["tests.gui.fixtures"]


class TestDirectorySelectorInitialization:
    """Test the initialization of directorySelector."""

    def test_init_default(self, qapp):
        """Test default initialization without initial path."""
        selector = directorySelector()
        assert os.path.isdir(selector.path)
        # Path should be the home directory
        assert selector.path == os.path.expanduser("~")

    def test_init_with_valid_path(self, qapp, mex1_dir):
        """Test initialization with a valid initial path."""
        selector = directorySelector(init_path=mex1_dir)
        # Normalize the expected path
        expected_path = os.path.abspath(mex1_dir)
        assert selector.path == expected_path

    def test_init_with_invalid_path(self, qapp):
        """Test initialization with an invalid path raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            directorySelector(init_path="/nonexistent/path/that/does/not/exist")

    def test_init_path_edit_display(self, qapp, mex2_dir):
        """Test that the path edit displays the initialized path."""
        selector = directorySelector(init_path=mex2_dir)
        # Normalize path and format it for display
        normalized_path = os.path.abspath(mex2_dir)
        expected_path = directorySelector._format_path(normalized_path)
        assert selector.path_edit.text() == expected_path

    def test_init_with_parent(self, qapp, mex1_dir):
        """Test initialization with a parent widget."""
        parent = QtWidgets.QWidget()
        selector = directorySelector(parent=parent, init_path=mex1_dir)
        assert selector.parent() == parent
        # Check that margins are zero
        assert selector.contentsMargins() == QtCore.QMargins(0, 0, 0, 0)


class TestPathValidation:
    """Test path validation functionality."""

    def test_validate_path_valid(self, mex1_dir):
        """Test validate_path returns True for valid directories."""
        assert directorySelector.validate_path(mex1_dir) is True

    def test_validate_path_invalid(self):
        """Test validate_path returns False for invalid paths."""
        assert directorySelector.validate_path("/nonexistent/path") is False

    def test_validate_path_file(self, temp_dir):
        """Test validate_path returns False for file paths."""
        # Create a temporary file
        test_file = os.path.join(temp_dir, "test_file.txt")
        with open(test_file, "w") as f:
            f.write("test")
        assert directorySelector.validate_path(test_file) is False

    def test_validate_edit_path_valid(self, qapp, mex1_dir, mex2_dir):
        """Test validate_edit_path with a valid path updates the selector."""
        selector = directorySelector(init_path=mex1_dir)
        selector.path_edit.setText(mex2_dir)
        selector.validate_edit_path()
        expected_path = os.path.abspath(mex2_dir)
        assert selector.path == expected_path
        # Stylesheet should be reset to default (not red)
        assert selector.path_edit.styleSheet() == selector._path_edit_default_stylesheet

    def test_validate_edit_path_invalid(self, qapp, mex1_dir):
        """Test validate_edit_path with invalid path reverts to previous path."""
        selector = directorySelector(init_path=mex1_dir)
        original_path = selector.path
        selector.path_edit.setText("/invalid/nonexistent/path")
        selector.validate_edit_path()
        assert selector.path == original_path
        # Stylesheet should be set to invalid (red background)
        assert selector.path_edit.styleSheet() == selector._STYLESHEET_INVALID

    def test_validate_edit_path_empty(self, qapp, mex1_dir):
        """Test validate_edit_path with empty string treats it as root directory.

        Note: _format_path("") returns the root directory (e.g., "\" on Windows),
        which is a valid directory, so the validation passes and sets the path to root.
        """
        selector = directorySelector(init_path=mex1_dir)
        selector.path_edit.setText("")
        selector.validate_edit_path()
        # After validation, empty string should result in root directory
        # (which is the result of _format_path(""))
        assert selector.path_edit.styleSheet() == selector._path_edit_default_stylesheet


class TestPathFormatting:
    """Test path formatting functionality."""

    def test_format_path_adds_trailing_slash(self, mex1_dir):
        """Test that _format_path adds a trailing slash."""
        formatted = directorySelector._format_path(mex1_dir)
        assert formatted.endswith(os.sep)

    def test_format_path_with_existing_trailing_slash(self, mex1_dir):
        """Test _format_path with path that already has trailing slash."""
        path_with_slash = mex1_dir + os.sep
        formatted = directorySelector._format_path(path_with_slash)
        # Should have exactly one trailing slash, not doubled
        assert formatted.endswith(os.sep)
        assert not formatted.endswith(os.sep + os.sep)

    def test_format_path_strips_whitespace(self, mex1_dir):
        """Test that _format_path strips leading/trailing whitespace."""
        path_with_spaces = f"  {mex1_dir}  "
        formatted = directorySelector._format_path(path_with_spaces)
        expected = directorySelector._format_path(mex1_dir)
        assert formatted == expected

    def test_format_path_removes_duplicate_slashes(self):
        """Test that _format_path removes duplicate slashes."""
        path = f"path{os.sep}{os.sep}to{os.sep}{os.sep}{os.sep}dir"
        formatted = directorySelector._format_path(path)
        # Should not have consecutive slashes
        assert f"{os.sep}{os.sep}" not in formatted

    def test_format_path_normalizes_slashes_to_os(self):
        """Test that _format_path converts slashes to OS-appropriate format."""
        # Create a path with forward slashes
        path = "path/to/some/dir"
        formatted = directorySelector._format_path(path)
        # Should use OS-appropriate separator
        assert os.sep in formatted
        # Should not have mixed separators
        if os.sep == "\\":
            # On Windows, should not have forward slashes
            assert "/" not in formatted


class TestPathProperty:
    """Test the path property getter/setter/deleter."""

    def test_path_getter(self, qapp, mex1_dir):
        """Test getting the path property."""
        selector = directorySelector(init_path=mex1_dir)
        path = selector.path
        assert os.path.isabs(path)
        assert os.path.isdir(path)

    def test_path_setter_valid(self, qapp, mex1_dir, mex2_dir):
        """Test setting the path property with a valid path."""
        selector = directorySelector(init_path=mex1_dir)
        selector.path = mex2_dir
        expected_path = os.path.abspath(mex2_dir)
        assert selector.path == expected_path

    def test_path_setter_invalid(self, qapp, mex1_dir):
        """Test setting the path property with invalid path raises ValueError."""
        selector = directorySelector(init_path=mex1_dir)
        with pytest.raises(ValueError, match="does not exist"):
            selector.path = "/nonexistent/path"

    def test_path_deleter_resets_to_home(self, qapp, mex1_dir):
        """Test deleting the path resets to home directory."""
        selector = directorySelector(init_path=mex1_dir)
        del selector.path
        home = os.path.expanduser("~")
        assert selector.path == home


class TestSignalEmission:
    """Test signal emission functionality."""

    def test_new_path_signal_on_valid_edit(self, qapp, mex1_dir, mex2_dir):
        """Test that new_path signal is emitted when validating a new path."""
        selector = directorySelector(init_path=mex1_dir)
        signal_received = []

        def on_new_path(value):
            signal_received.append(value)

        selector.new_path.connect(on_new_path)
        selector.path_edit.setText(mex2_dir)
        selector.validate_edit_path()

        assert len(signal_received) == 1
        assert signal_received[0] is True

    def test_refresh_emits_signal(self, qapp, mex1_dir):
        """Test that refresh method emits new_path signal."""
        selector = directorySelector(init_path=mex1_dir)
        signal_received = []

        def on_new_path(value):
            signal_received.append(value)

        selector.new_path.connect(on_new_path)
        selector.refresh()

        assert len(signal_received) == 1
        assert signal_received[0] is True

    def test_no_signal_on_invalid_edit(self, qapp, mex1_dir):
        """Test that signal is not emitted when path validation fails."""
        selector = directorySelector(init_path=mex1_dir)
        signal_received = []

        def on_new_path(value):
            signal_received.append(value)

        selector.new_path.connect(on_new_path)
        selector.path_edit.setText("/invalid/path")
        selector.validate_edit_path()

        # Signal should not be emitted for invalid path
        assert len(signal_received) == 0


class TestWidgetComponents:
    """Test the widget components."""

    def test_has_path_edit(self, qapp, mex1_dir):
        """Test that selector has a path_edit widget."""
        selector = directorySelector(init_path=mex1_dir)
        assert isinstance(selector.path_edit, QtWidgets.QLineEdit)

    def test_has_path_select_button(self, qapp, mex1_dir):
        """Test that selector has a path_select_btn button."""
        selector = directorySelector(init_path=mex1_dir)
        assert isinstance(selector.path_select_btn, QtWidgets.QPushButton)
        assert selector.path_select_btn.text() == "Browse"

    def test_has_refresh_button(self, qapp, mex1_dir):
        """Test that selector has a refresh_btn button."""
        selector = directorySelector(init_path=mex1_dir)
        assert isinstance(selector.refresh_btn, QtWidgets.QPushButton)

    def test_layout_is_horizontal(self, qapp, mex1_dir):
        """Test that the widget uses a horizontal layout."""
        selector = directorySelector(init_path=mex1_dir)
        layout = selector.layout()
        assert isinstance(layout, QtWidgets.QHBoxLayout)

    def test_widgets_in_layout(self, qapp, mex1_dir):
        """Test that all widgets are added to the layout."""
        selector = directorySelector(init_path=mex1_dir)
        layout = selector.layout()
        if layout is None:
            pytest.fail("Layout is not set on the widget")
        assert layout.count() == 3  # path_edit, path_select_btn, refresh_btn


class TestDirectorySelectorWithTestData:
    """Test directorySelector with actual test data directories."""

    def test_select_mex1_directory(self, qapp, mex1_dir):
        """Test selecting MEX1 directory from test data."""
        selector = directorySelector()
        selector.path = mex1_dir
        expected_path = os.path.abspath(mex1_dir)
        assert selector.path == expected_path

    def test_select_mex2_directory(self, qapp, mex2_dir):
        """Test selecting MEX2 directory from test data."""
        selector = directorySelector()
        selector.path = mex2_dir
        expected_path = os.path.abspath(mex2_dir)
        assert selector.path == expected_path

    def test_select_sxr_directory(self, qapp, sxr_dir):
        """Test selecting SXR directory from test data."""
        selector = directorySelector()
        selector.path = sxr_dir
        expected_path = os.path.abspath(sxr_dir)
        assert selector.path == expected_path

    def test_switch_between_test_data_directories(self, qapp, mex1_dir, mex2_dir):
        """Test switching between multiple test data directories."""
        selector = directorySelector(init_path=mex1_dir)
        expected_path_1 = os.path.abspath(mex1_dir)
        assert selector.path == expected_path_1

        selector.path = mex2_dir
        expected_path_2 = os.path.abspath(mex2_dir)
        assert selector.path == expected_path_2

        selector.path = mex1_dir
        assert selector.path == expected_path_1

    def test_accessible_names_and_descriptions(self, qapp, mex1_dir):
        """Test that widgets have proper accessible names and descriptions."""
        selector = directorySelector(init_path=mex1_dir)
        assert selector.path_edit.accessibleName() == "Directory"
        assert (
            selector.path_edit.accessibleDescription() == directorySelector._EDIT_DESC
        )
