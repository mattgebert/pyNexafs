:html_theme.sidebar_secondary.remove: true

.. include:: ../README.rst

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

- Development (Github): https://github.com/xraysoftmat/pyNexafs/
- Releases:
    * Github Releases: https://github.com/xraysoftmat/pyNexafs/releases
    * PyPI: https://pypi.python.org/pypi/pyNexafs/
- Documentation: https://pyNexafs.readthedocs.io/


.. toctree::
    :hidden:

    source/install
    source/tutorials/index
    source/gui
    source/api

.. support
.. CHANGELOG
.. api
