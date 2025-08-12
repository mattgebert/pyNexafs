pyNexafs
########

A repository for reading, reducing and analyzing NEXAFS[^1] / XANES[^2] data.

.. image:: https://results.pre-commit.ci/badge/github/xraysoftmat/pyNexafs/main.svg
    :alt: pre-commit.ci status
    :target:
.. image:: https://img.shields.io/badge/versioning-Python%20SemVer-blue.svg
    :alt: Python SemVer
    :target: https://python-semantic-release.readthedocs.io/en/stable/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Code style: black
    :target: https://github.com/psf/black
.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :alt: Ruff
    :target: https://github.com/astral-sh/ruff
.. image:: https://img.shields.io/badge/doc_style-numpydoc-blue.svg
    :alt: Code doc: numpydoc
    :target: https://github.com/numpy/numpydoc

.. image:: https://github.com/xraysoftmat/pyNexafs/actions/workflows/test.yml/badge.svg
    :alt: PyTest
    :target: https://github.com/xraysoftmat/pyNexafs/actions/workflows/test.yml
.. image:: https://coveralls.io/repos/github/xraysoftmat/pyNexafs/badge.svg
    :alt: Coverage Status
    :target: https://coveralls.io/github/xraysoftmat/pyNexafs
.. image:: https://results.pre-commit.ci/badge/github/xraysoftmat/pyNexafs/main.svg
    :alt: pre-commit.ci status
    :target: https://results.pre-commit.ci/latest/github/xraysoftmat/pyNexafs/main

This repository does not support EXAFS[^3], which has a very similar measurement philosophy yet very distinct physics.

.. include:: ./todolist.md
    :parser: myst_parser.sphinx_

[^1]: NEXAFS: Near Edge X-ray Absorption Fine Structure
[^2]: XANES: X-ray Absorption Near Edge Spectroscopy
[^3]: EXAFS: Extended X-ray Absorption Fine Structure
