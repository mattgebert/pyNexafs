=====================================
pyNexafs
=====================================

``pyNexafs`` is a comprehensive toolkit for NEXAFS\ [#a]_ (or XANES\ [#b]_) data, and is built to the feature-rich standards of `xraysoftmat <https://github.com/xraysoftmat>`_, so you can trust ongoing stability and developement.


|tool-semver| |tool-black| |tool-ruff| |tool-numpydoc|

|PyPI Version| |PyTest| |Coveralls| |Pre-commit|



.. |PyPI Version| image:: https://img.shields.io/pypi/v/pyNexafs?label=pyNexafs&logo=pypi
   :target: https://pypi.org/project/pyNexafs/
   :alt: pypi
.. |PyTest| image:: https://github.com/xraysoftmat/pyNexafs/actions/workflows/test.yml/badge.svg
    :alt: PyTest
    :target: https://github.com/xraysoftmat/pyNexafs/actions/workflows/test.yml
.. |Coveralls| image:: https://coveralls.io/repos/github/xraysoftmat/pyNexafs/badge.svg
    :alt: Coverage Status
    :target: https://coveralls.io/github/xraysoftmat/pyNexafs
.. |Pre-commit| image:: https://results.pre-commit.ci/badge/github/xraysoftmat/pyNexafs/main.svg
    :alt: pre-commit.ci status
    :target: https://results.pre-commit.ci/latest/github/xraysoftmat/pyNexafs/main

.. |tool-semver| image:: https://img.shields.io/badge/versioning-Python%20SemVer-blue.svg
    :alt: Python SemVer
    :target: https://python-semantic-release.readthedocs.io/en/stable/
.. |tool-black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Code style: black
    :target: https://github.com/psf/black
.. |tool-ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :alt: Ruff
    :target: https://github.com/astral-sh/ruff
.. |tool-numpydoc| image:: https://img.shields.io/badge/doc_style-numpydoc-blue.svg
    :alt: Code doc: numpydoc
    :target: https://github.com/numpy/numpydoc

Motivation
##########

``pyNexafs`` was designed to solve backwards-compatibility issues between (inevitably changing or upgraded) Synchrotron beamline configurations, and allow reproducable analysis (reduction, normalisation and fitting) of NEXAFS data. This is done by separating file parsing and NEXAFS (scan) handling.

Todo List
#########

This repository is in a Beta development state. The following list maps the required features to be implemented before a full released.

- ☑ Implement base classes for parsing and NEXAFS
- ☑ Implement core normalisation wrappers
- ☑ Implement support for Australian Synchrotron NEXAFS beamlines
- ☐ Fix parser classes to be more flexible (multi-dimensional data) but allow/encourage overriding of `to_scan` method.
- ☐ Ensure parser `load` method records the successful method signature.
- ☐ Comprehensive mapping structure for NEXAFS data types (including accessors for a single scan, i.e. `scan.drain`, `scan.flour` or `scan.PFY`...)
- ☐ Comprehensive readthedocs documentation.
- ☐ Comprehensive (>90%) unit testing for core API and modules.
- ☐ Functioning PyQt6 GUI
- ☐ Unit testing for PyQt6 GUI
- ☐ Generic NEXAFS file loader (two column)

.. ☑ ☐

This repository does not support EXAFS [#c]_, which has a very similar measurement philosophy yet very distinct physics.

.. rubric:: Footnotes

.. [#a] NEXAFS: Near Edge X-ray Absorption Fine Structure
.. [#b] XANES: X-ray Absorption Near Edge Spectroscopy
.. [#c] EXAFS: Extended X-ray Absorption Fine Structure
