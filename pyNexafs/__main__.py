"""
Main entry point for the pyNexafs package.

Runs the GUI for the package if the dependencies are met.
Can be run from the command line with: `python -m pyNexafs`

Arguments:
    directory: Optional path to a directory containing NEXAFS data files (defaults to home directory).
    --help: Show help message and exit.
    --version: Show the version of pyNexafs and exit.
    --traceback: Show detailed traceback if an error occurs.
"""

import os
from importlib.metadata import version
import traceback
import argparse


def main(dir_arg: str | None = None):
    """
    The main function to run the pyNexafs GUI.

    Parameters
    ----------
    dir_arg : str | None, optional
        A directory path to load NEXAFS files from.
        If None, defaults to the user home directory.
    """

    dir_exists = os.path.isdir(dir_arg) if dir_arg else False
    # Check if the dependencies are met
    try:
        import PyQt6

        assert PyQt6 is not None
        print(f"PyQt6 ({version('PyQt6')}) is installed. Starting the GUI...")
    except ImportError as e:
        print(
            "PyQt6 is not installed. Please install PyQt6 to run the GUI."
            + f"\nTraceback: {e}"
        )
        exit()

    from pyNexafs.gui.data_browser import gui

    # Create and run the app
    if dir_arg and dir_exists:
        gui(directory=os.path.normpath(dir_arg))
    else:
        gui()


def main_with_traceback(dir_arg: str | None = None):
    """
    A wrapper for the main function that shows a detailed traceback on errors.

    Parameters
    ----------
    dir_arg : str | None, optional
        A directory path to load NEXAFS files from.
        If None, defaults to the user home directory.
    """
    try:
        main(dir_arg)
    except Exception as e:
        from PyQt6 import QtWidgets

        # Create a QT window to display the error
        app = QtWidgets.QApplication([])
        app.setApplicationName("kkcalc: Kramers-Kronig Calculator (Error)")
        error_dialog = QtWidgets.QErrorMessage()
        # Prepare the message: the error and the traceback
        msg = f"An error occurred, causing kkcalc to crash.:\
               \n{str(e)}\
               \nPlease report this issue at https://github.com/xraysoftmat/kkcalc/issues"
        error_dialog.showMessage(msg)
        app.exec()

        # Add the detail
        tb = f"{traceback.format_exc()}"
        layout = error_dialog.layout()
        if layout is not None:
            layout.addWidget(QtWidgets.QLabel("Detailed traceback:"))
            layout.addWidget(QtWidgets.QLabel(tb))
        error_dialog.exec()


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(
        prog="pyNexafs",
        description="A Python package for analyzing NEXAFS data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    """The argument parser for the pyNexafs command line interface."""

    argument_parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('pyNexafs')}",
    )

    argument_parser.add_argument(
        "--traceback",
        action="store_true",
        help="Show detailed traceback on errors.",
    )

    argument_parser.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="Directory to load NEXAFS files from (defaults to home directory).",
    )

    args = argument_parser.parse_args()

    if args.traceback:
        main_with_traceback(args.directory)
    else:
        main(args.directory)
