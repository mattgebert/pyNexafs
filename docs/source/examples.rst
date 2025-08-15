=========
Examples
=========

MEX2
####

Medium Energy X-ray Absorption Spectroscopy at the Australian Synchrotron
-------------------------------------------------------------------------

A simple demonstration of using pyNexafs to load MEX2 data files, using the MEX2 test data located `on the github repository <https://github.com/xraysoftmat/pyNexafs/tree/main/tests/test_data/au/MEX2/2025-03>`_.

First, locate some data. The example data ``MEX2_13385.mda`` is the spectra of the P3HT sulfonated organic-semiconducting polmyer, measured at the 55 degree `magic` angle. The ``.mda`` format is an efficient binary format written according to the ``EPICS`` `saveData specification <https://epics-modules.github.io/sscan/sscanDoc.html>`_. See :ref:`MDA-Files` for more information on processing the MDA file format.

.. code-block:: python

    import os
    # Get the current working directory
    # Usually the folder you opened your IDE into.
    cwd = os.getcwd()

    # Construct the file path to the MEX2 data file
    filepath = os.path.join(cwd, "MEX2_13385.mda")

    # Check if the file exists.
    os.path.exists(filepath)

``pyNexafs`` hosts a variety of different beamline data formats, which can be accessed through the `parsers` module. The MEX2 parser is located in the `pyNexafs.parsers.au` submodule, and can be used to load MEX2 data files.

.. code-block:: python

    import pyNexafs as pnx
    # Load MEX2 data file
    parser = pnx.parsers.au.MEX2_NEXAFS(
        filepath = filepath, # The .xdi or .mda filepath
        header_only = False,
        relabel = True,
        energy_bin_domain = (2230, 2390), # Center at 2.31 keV
        use_recent_binning = False,
    )

Here we've used a heap of parameters:

- ``header_only``: If True, only the header information is loaded, not the scan data.
- ``relabel``: If True, the data channels are relabeled to more convenient names.
- ``energy_bin_domain``: This defines the integrating binning energy range (in eV) for the reduction. Historically two detector types have been used, a Dante and an Xpress3 MCA, which have unique names and unique energy bins. With unit-testing, ``pyNexafs`` is build to be backwards compatible between updated file specifications.
- ``use_recent_binning``: If True, the most recent binning settings (from a previous reduction) are used.

Because the MEX2 flourescence detector consists of a MCA (Multi-channel Analyzer), the flourescence is measured across many detection energy bins at every NEXAFS scan beam energy, and so needs to be reduced to a domain of interest to create a 1D NEXAFS flourescence profile. This reduced channel of data is referred to as `partial flourescence yield (PFY)`. In this case, we've chosen an energy range that spans underneath the Sulfur K edge (2474 eV), as we're measuring the relaxation flourescence of the anti-bonding orbital to groundstate transition. The range is 80 eV eitherside of 2310 eV, which corresponds to about 16 bins (160 eV) for the Xpress3 MCA detector.

In this ``parser`` form, we can inspect the raw data as processed by the parser functions. Most parsers classes ought to return data in the form of a ``tuple`` of (data, data labels, units, parameters).

.. code-block:: python

    display(parser.labels[0:12]) # Show the first 12 data labels

    ['Energy Setpoint',
    'Gate Time Setpoint',
    'Energy',
    'Bragg',
    'Current Monitor',
    'Beam Intensity Monitor',
    'I0',
    'SampleDrain',
    'MEX2ES01DAQ01:ch4:S:MeanValue_RBV',
    'MEX2ES01DPP02:DTC_Window1_AVG_CPS',
    'MEX2ES01DPP02:DTC_Window2_AVG_CPS',
    'MEX2ES01ZEB01:PC_GATE_WID:RBV']

Now that we have reduced the MCA channels, we can process the raw data.
Here's how ``pyNexafs`` assists; all data parsers are required to implement a method that converts to a standardized `Scan` object, with a typed interface.
This mapping is currently defined (for 1D scan data) in the ``COLUMN_ASSIGNMENTS`` class property.

.. code-block:: python

    display(parser.COLUMN_ASSIGNMENTS) # Show the column assignments

    {'x': ('Energy', 'Energy Setpoint', 'energy'),
    'y': [('I0', 'i0'),
    ('Sample Drain', 'SampleDrain'),
    ('Fluorescence', 'ifluor', 'Fluorescence Sum', 'Fluorescence Sum (Reduced)'),
    ('Count Time', 'count_time'),
    'ICR_AVG',
    'OCR_AVG',
    'Fluorescence Detector 1',
    'Fluorescence Detector 2',
    'Fluorescence Detector 3',
    'Fluorescence Detector 4',
    ('Bragg', 'bragg')],
    'y_errs': None,
    'x_errs': None}

    # This conversion mapping is automatically used in the parser.to_scan class method.
    scan = parser.to_scan()

To get a quick look at our data, we can use the `snapshot` method.

.. code-block:: python

    fig = scan.snapshot(columns=3) # Create a grid of plots, with 3 columns
    fig.suptitle("MEX2 Scan Snapshot")
    fig.tight_layout()
    fig.show()

.. plot::
    :include-source: False

    import matplotlib.pyplot as plt
    import pyNexafs as pnx
    import os
    import numpy as np

    cwd = os.getcwd() # conf.py
    cwd_to_root = "./../.."
    root_to_data = "tests/test_data/au/MEX2/2025-03/MEX2_13385.mda"
    filepath = os.path.normpath(os.path.join(cwd, cwd_to_root, root_to_data))

    parser = pnx.parsers.au.MEX2_NEXAFS(
        filepath=filepath,
        header_only=False,
        relabel=True,
        energy_bin_domain=(2230, 2390),
        use_recent_binning=False,
    )

    scan = parser.to_scan()
    fig = scan.snapshot()
    fig.suptitle("MEX2 Scan Snapshot")
    fig.show()

Then we can access the data:

.. code-block:: python

    # The scan object has a typed interface, with x and y data.
    drain_idx = scan.y_labels.index('SampleDrain') # Sample Drain signal
    x = scan.x # 1D
    y = scan.y[:, drain_idx] # First index is data rows, second is signal index.

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_title("MEX2 Sample Drain Signal")
    ax.set_xlabel(scan.x_label)
    ax.set_ylabel("Sample Drain")
    plt.show()

.. plot::
    :include-source: False
    :show-source-link: False

    import matplotlib.pyplot as plt
    import pyNexafs as pnx
    import os

    cwd = os.getcwd() # conf.py
    cwd_to_root = "./../.."
    root_to_data = "tests/test_data/au/MEX2/2025-03/MEX2_13385.mda"
    filepath = os.path.normpath(os.path.join(cwd, cwd_to_root, root_to_data))


    parser = pnx.parsers.au.MEX2_NEXAFS(
        filepath=filepath,
        header_only=False,
        relabel=True,
        energy_bin_domain=(2230, 2390),
        use_recent_binning=False,
    )

    scan = parser.to_scan()
    x = scan.x
    y = scan.y[:, scan.y_labels.index('SampleDrain')]

    plt.plot(x, y)
    plt.title("MEX2 Sample Drain Signal")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Sample Drain (a.u.)")
    plt.show()


We can now apply appropriate corrections and normalisations on the raw data.

.. code-block:: python

    scan_norm = pnx.nexafs.scanNorm(scan, "I0")
    scan_edge = pnx.nexafs.scanNormEdges(
        scan_norm,
        pre_edge_domain=(2460, 2465),
        post_edge_domain=(2540, 2560)
    )

    fig, ax = plt.subplots()
    for i, s in enumerate([scan, scan_norm, scan_edge]):
        x, y = s.x, s.y[:, s.y_labels.index('SampleDrain')]
        ax.plot(x, y, label=["raw", "norm", "edge"][i])
    ax.legend()

.. plot::
    :include-source: False

    import matplotlib.pyplot as plt
    import pyNexafs as pnx
    import os

    cwd = os.getcwd() # conf.py
    cwd_to_root = "./../.."
    root_to_data = "tests/test_data/au/MEX2/2025-03/MEX2_13385.mda"
    filepath = os.path.normpath(os.path.join(cwd, cwd_to_root, root_to_data))


    parser = pnx.parsers.au.MEX2_NEXAFS(
        filepath=filepath,
        header_only=False,
        relabel=True,
        energy_bin_domain=(2230, 2390),
        use_recent_binning=False,
    )

    scan = parser.to_scan()
    scan_norm = pnx.nexafs.scanNorm(scan, "I0")
    scan_edge = pnx.nexafs.scanNormEdges(scan_norm, pre_edge_domain=(2460, 2465), post_edge_domain=(2540, 2560))

    fig, ax = plt.subplots()
    for i, s in enumerate([scan, scan_norm, scan_edge]):
        x, y = s.x, s.y[:, s.y_labels.index('SampleDrain')]
        ax.plot(x, y, label=["raw", "norm", "edge"][i])
    ax.legend()

    plt.title("MEX2 Sample Drain Signal")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Sample Drain (a.u.)")
    plt.show()

.. _MDA-FILES:

MDA Files
#########

MDA files are a generic data format used for storing multiple variable arrays of differing rank. The MDA format is designed to be efficient and flexible. Here's an example of how to use the ``pyNexafs.utils.mda``` module to read such raw files.

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
