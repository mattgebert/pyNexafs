=========
Tutorials
=========

Introduction
############

PyNexafs was designed to be a general purpose tool for working with NEXAFS data, in particular to combat the typical issues:

- Backwards compatibility (with file format and naming changes) due to hardware and software updates at Synchrotrons and other facilities.
- Reducing multidimensional datasets (e.g. MDA files or Partial Fluorescence Yields).
- Data normalisation and background subtraction.
- Consistent naming conventions for NEXAFS channels / metadata.
- The need for fast data processing and visualization to keep up with the high data throughput at modern facilities.

With that in mind, pyNexafs is broken down into two main submodules:

- ``pyNexafs.parsers`` for data loading and reduction.
- ``pyNexafs.nexafs`` for data normalisation and quick analysis.

Parsers
#######

All ``pyNexafs`` parsers are required to obey the same design principles to be robust.

Parser Methods
==============
All parser classes are required to register **parsing** methods with the same naming signature.
An example is ``MEX2_NEXAFS.parse_``
 which are then registered and used to load data files.

Parser RELABELS
===============
Each parser has a ``RELABELS`` class attribute, which is a map of synonamous data labels to a single canonical label. This allows for consistent naming conventions across different file formats and beamlines, and also allows for backwards compatibility with older file formats that may have different naming conventions. A typical ``RELABELS`` map looks like this:


.. toctree::
   :maxdepth: 2
   :caption: Australian Synchrotron Parsers

   SXR
   MEX2

Utilities
#########

.. toctree::
   :maxdepth: 2
   :caption: Utilities

   MDA
