.. _MDA-FILES:
MDA Files
#########

MDA files are a generic data format used for storing multiple variable arrays of differing rank. The MDA format is efficient and flexible, designed according to the ``EPICS`` `saveData specification <https://epics-modules.github.io/sscan/sscanDoc.html>`_. Here's an example of how to use the ``pyNexafs.utils.mda``` module to read such raw files.

.. code-block:: python

    import pyNexafs as pnx

    # Create an instance of a MDAFileReader object for the filepath.
    mda_reader = pnx.utils.mda.MDAFileReader("path/to/MEX2_13385.mda")

    # Get the header information:
    header = mda_reader.read_header_as_dict()

.. code-block:: console

    {'mda_version': 1.399999976158142,
     'mda_scan_number': 13385,
     'mda_rank': 2,
     'mda_dimensions': [520, 4096],
     'mda_isRegular': 1,
     'mda_pExtra': 34398964,
     'mda_pmain_scan': 28}

We can see that the ``mda_rank`` is 2, indicating that there are two dimensions in the data. Let's get the parameter values (PV's) and the multi-dimensional data values.

.. code-block:: python

    # Read the parameter values
    pvs = mda_reader.read_parameters()

    # Read the multi-dimensional data values and headers
    data, scan_headers = mda_reader.read_scans()

The ``pvs`` now contains a dictionary of parameter values, for example:

.. code-block:: python

    pvs["MEX2SSCAN01:saveData_comment1"] # Returns (Description, Units, Value)

.. code-block:: console

    ('GUI comment 1 field', None, 'P3HT_1 55 degree')

The ``data`` is a tuple of different rank ``numpy`` arrays (i.e. 1D, 2D, etc.), while ``scan_headers`` is a tuple of ``MDAScan`` objects.
Let's have a look at the 1D data - it's shape is 520 rows of energies, with 66 different columns. The ``MDAScan.labels()`` corresponds to the same length.

.. code-block:: python

    print(data[0].shape, scan_headers[0].labels().__len__())


.. code-block:: console

    ((520, 66), 66)

Let's have a look at the 2D data - it's shape is 520 rows of energies like before, but now two more indexes with length 4096 (the number of energy bin channels) and 4 (the number of flourescence detector channels).

.. code-block:: python

    data[1].shape, scan_headers[1].labels().__len__()

.. code-block:: console

    ((520, 4096, 4), 4)
