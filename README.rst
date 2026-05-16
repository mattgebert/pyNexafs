=====================================
pyNexafs
=====================================

``pyNexafs`` is a comprehensive toolkit for NEXAFS\ [#a]_ (or XANES\ [#b]_) data, and is built to the feature-rich standards of `xraysoftmat <https://github.com/xraysoftmat>`_, so you can trust ongoing stability and developement.


|PyPI Version| |zenodo| |readthedocs| |Coveralls| |Pre-commit|

|PyTest| |Linting| |Documentation|

|tool-semver| |tool-black| |tool-ruff| |tool-numpydoc|

.. |PyPI Version| image:: https://img.shields.io/pypi/v/pyNexafs?label=pyNexafs&logo=pypi
   :target: https://pypi.org/project/pyNexafs/
   :alt: pypi
.. |PyTest| image:: https://github.com/xraysoftmat/pyNexafs/actions/workflows/tests.yml/badge.svg
    :alt: PyTest
    :target: https://github.com/xraysoftmat/pyNexafs/actions/workflows/tests.yml
.. |Linting| image:: https://github.com/xraysoftmat/pyNexafs/actions/workflows/linting.yml/badge.svg
    :alt: Linting
    :target: https://github.com/xraysoftmat/pyNexafs/actions/workflows/linting.yml
.. |Documentation| image:: https://github.com/xraysoftmat/pyNexafs/actions/workflows/docs.yml/badge.svg
    :alt: Documentation
    :target: https://github.com/xraysoftmat/pyNexafs/actions/workflows/docs.yml
.. |Coveralls| image:: https://coveralls.io/repos/github/xraysoftmat/pyNexafs/badge.svg
    :alt: Coverage Status
    :target: https://coveralls.io/github/xraysoftmat/pyNexafs
.. |Pre-commit| image:: https://results.pre-commit.ci/badge/github/xraysoftmat/pyNexafs/main.svg
    :alt: pre-commit.ci status
    :target: https://results.pre-commit.ci/latest/github/xraysoftmat/pyNexafs/main
.. |readthedocs| image:: https://img.shields.io/readthedocs/pyNexafs?version=latest&style=flat&label=ReadtheDocs
    :alt: Documentation
    :target: https://pynexafs.readthedocs.io/
.. |zenodo| image:: https://zenodo.org/badge/772544574.svg
  :target: https://doi.org/10.5281/zenodo.19102726

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

``pyNexafs`` was designed to solve backwards-compatibility issues between (inevitably changing or upgraded) Synchrotron beamline configurations, and allow reproducable analysis (reduction, normalisation and fitting) of NEXAFS data.
This is done by separating file parsing, reduction and NEXAFS (scan) handling.

This repository does not currently support EXAFS [#c]_, which has a very similar measurement philosophy yet very distinct physics.

.. rubric:: Footnotes

.. [#a] NEXAFS: Near Edge X-ray Absorption Fine Structure
.. [#b] XANES: X-ray Absorption Near Edge Spectroscopy
.. [#c] EXAFS: Extended X-ray Absorption Fine Structure
