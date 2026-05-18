"""Shared pytest fixtures for GUI tests."""

import os
import tempfile

import pytest


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), "..", "test_data")


@pytest.fixture
def test_data_au_dir(test_data_dir):
    """Return the path to the au subdirectory in test data."""
    return os.path.join(test_data_dir, "au")


@pytest.fixture
def mex1_dir(test_data_au_dir):
    """Return the path to the MEX1 test data directory."""
    return os.path.join(test_data_au_dir, "MEX1")


@pytest.fixture
def mex2_dir(test_data_au_dir):
    """Return the path to the MEX2 test data directory."""
    return os.path.join(test_data_au_dir, "MEX2")


@pytest.fixture
def sxr_dir(test_data_au_dir):
    """Return the path to the SXR test data directory."""
    return os.path.join(test_data_au_dir, "SXR")


@pytest.fixture
def temp_dir():
    """Return a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
