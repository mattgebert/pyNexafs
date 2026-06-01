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
    """Return the path to the AU test data directory."""
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
def mex2_dir_data_2024_03(mex2_dir):
    """Return the path to the MEX2 test data directory for 2024-03 parser."""
    return os.path.join(mex2_dir, "2024-03")


@pytest.fixture
def mex2_dir_data_2025_03(mex2_dir):
    """Return the path to the MEX2 test data directory for 2025-03 parser."""
    return os.path.join(mex2_dir, "2025-03")


# Fixture runs once per known MEX2 data folder.
@pytest.fixture(params=["2024-03", "2025-03"], ids=lambda name: f"mex2-{name}")
def mex2_data_dir(request, mex2_dir):
    """
    Return a folder with MEX2 data files in it.
    """
    return os.path.join(mex2_dir, request.param)


@pytest.fixture
def sxr_dir(test_data_au_dir):
    """Return the path to the SXR test data directory."""
    return os.path.join(test_data_au_dir, "SXR")


@pytest.fixture
def temp_dir():
    """Return a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
