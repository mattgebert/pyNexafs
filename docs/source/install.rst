============
Installation
============

The ``pyNexafs`` package requires Python 3.11+, and is available from:

- PyPI: https://pypi.python.org/pypi/pyNexafs/
- GitHub: https://github.com/xraysoftmat/pyNexafs

Once you have installed Python and added it to the path, we recommend adding a virtual environment to manage dependencies and avoid conflicts with other Python packages, before installing ``pyNexafs``.
You can create a virtual environment using the following commands (replace ``myvenv`` with something useful):

.. code-block:: bash

    # Alternatively use uv: `uv venv myvenv`
    > python -m venv myvenv

    # On Windows use `myvenv\Scripts\activate`
    > source myvenv/bin/activate
    (myvenv) > ... # indication of the environment

Then install the package using pip:

.. code-block:: bash

    (myvenv) > pip install pyNexafs # install regular project
    # or
    (myvenv) > pip install pyNexafs --group dev # PEP735

Following `PEP735 <https://peps.python.org/pep-0735/>`_, pyNexafs also has dependency groups established. You may need to upgrade ``pip>=25.1`` to use dependency groups. The following groups are available:

- ``docs`` : Install Sphinx, numpydoc and other packages required for building the documentation.
- ``gui`` : Install graphic packages (PyQT, matplotlib, pandas).
- ``dev`` : Install all additional packages for developement, including graphics, documentation and testing.

You can check the package is installed:

.. code-block:: bash

    (myvenv) > python

.. code-block:: python

    >>> import pyNexafs
    >>> print(pyNexafs.__version__)

.. parsed-literal::
    \ |release|  # or whatever the latest version is

This virtual environment can then be set in your IDE workflows, such as VSCode (see `here <https://code.visualstudio.com/docs/python/environments>`_). Simply point the current session to the virtual environment folder.

You can deactivate the virtual environment anytime.

.. code-block:: bash

    (myvenv) > deactivate
    > ...
