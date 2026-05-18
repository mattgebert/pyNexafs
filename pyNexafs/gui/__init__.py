"""
GUI components for pyNexafs, including widgets and configuration options.

Notably, `settings` is a configuration option to specify which Qt6 wrapper to use (PyQt6 or PySide6). By default, it is set to "PySide6", but can be changed to "PyQt6" if desired.

The `widgets` submodule contains the various GUI widgets used in the pyNexafs application, such as file loaders, viewers, and directory selectors.

"""

# Internal
from pyNexafs.gui import widgets


__all__ = ["widgets"]
