"""Pytest configuration and fixtures for GUI tests."""

import pytest
import sys
from pyNexafs.gui.config import QtWidgets


@pytest.fixture(scope="session")
def qapp():
    """Create and return a QApplication instance for the test session.

    This fixture is session-scoped so the QApplication is created once
    and reused across all tests. This is important for Qt applications
    which can only have one QApplication instance per process.
    """
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    yield app
    # Note: We don't delete the app here as it should persist for the session
