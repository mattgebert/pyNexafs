======================
Graphic User Interface
======================

.. warning::
    The pyNexafs GUI is currently in an early stage of development and is not yet fully functional.
    It is intended to provide a user-friendly interface for processing and visualising NEXAFS data, but it may still contain bugs and missing features.
    Users are encouraged to contribute to the development of the GUI by reporting issues and submitting pull requests on the GitHub repository.

The pyNexafs GUI is built from the inspiration of the `QANT <https://journals.iucr.org/s/issues/2016/01/00/rv5042/>`_ toolkit.
Simply, it provides fast and easy access to loading existing NEXAFS data files, processing (reduction and normalisation), and visualising the results in a timely fashion as to allow for on-the-fly decision making during beamtime experiments.
It is built using `PyQt6 <https://pypi.org/project/PyQt6/>`_.

PyPI Package
============
Once succesfully installed via PyPI, the pyNexafs GUI can be launched by running the following command in a terminal.

.. code-block:: console

    pyNexafs

Executable Builds
=================
Executable builds of the pyNexafs GUI are available for Windows and macOS.
These are built using `PyInstaller <https://pypi.org/project/pyinstaller/>`_ and can be downloaded from the `GitHub releases page <https://github.com/pyNexafs/pyNexafs/releases>`_ if they have been successfully built.
