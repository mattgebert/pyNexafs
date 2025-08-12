:html_theme.sidebar_secondary.remove: true

=====================================
pyNexafs
=====================================

Introduction
############

``pyNexafs`` is a comprehensive toolkit for NEXAFS\ [1]_ (or XANES\ [2]_) data, and is built to the feature-rich standards of `xraysoftmat <https://github.com/xraysoftmat>`_, so you can trust ongoing stability and developement.

``pyNexafs`` was designed to remove compatibility issues between different NEXAFS\ [1]_ data formats, and to provide a consistent interface for data reduction, fitting, and visualization. To learn more and get started, see

.. list-table:: pyNexafs Features
  :widths: 20 80
  :header-rows: 1

  * - **Feature**
    - **Description**
  * - Reduction
    - | Pre-edge removal (nearby edge, elastic peak, etc.)
      | Normalization
      | Background subtraction
  * - Fitting
    - Global dataset peak fitting
  * - Compatibility
    - | Support for :ref:`various file formats <support>`.
      | *Please raise an* `issue <https://github.com/xraysoftmat/pyNexafs/issues>`_ *with an example file / relevant information for unsupported formats*.
  * - Visualization
    - PyQt6 GUI for fast data processing
  * - Typing
    - | Enumerated NEXAFS detection sources.
      | Type hints for all functions and classes.

Links
#####

- Development: https://github.com/xraysoftmat/pyNexafs/
- Documentation: https://pyNexafs.readthedocs.io/
- PyPI: https://pypi.python.org/pypi/pyNexafs/


.. toctree::
    :hidden:

    install
    support
    CHANGELOG
    examples
    api

.. rubric:: Abbreviations
    :name: Abbreviations

.. [1] NEXAFS: Near-edge X-ray absorption fine structure
.. [2] XANES:  X-ray absorption near edge structure
