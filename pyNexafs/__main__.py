"""
Main entry point for the pyNexafs package.

Runs the GUI for the package if the dependencies are met.
Can be run from the command line with the command `python -m pyNexafs`.
"""

if __name__ == "__main__":
    # Check if the dependencies are met
    try:
        import PyQt6
    except ImportError as e:
        print(
            "PyQt6 is not installed. Please install PyQt6 to run the GUI."
            + f"\nTraceback: {e}"
        )
        exit()

    from pyNexafs.gui.data_browser import gui

    # Create and run the app
    gui()
