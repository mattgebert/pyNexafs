"""
Configuration for Qt6 python wrapper in pyNexafs GUI.

This module provides an endpoint import to specify the Qt6 wrapper to use (PyQt6 or PySide6) for the pyNexafs GUI.
By default, uses QtPy to determine the wrapper, but can be set manually by changing the `wrapper` variable.
"""


# Provide a consistent interface for QtCore, QtWidgets, and QtGui across different Qt wrappers (PyQt6 and PySide6).
# These references will be used by submodules

# from qtpy import QtCore, QtWidgets, QtGui
# from PyQt6 import QtCore, QtWidgets, QtGui
from PySide6 import QtCore, QtWidgets, QtGui

__all__ = ["QtCore", "QtWidgets", "QtGui"]
