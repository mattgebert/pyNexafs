import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTextEdit,
    QHBoxLayout,
    QWidget,
    QListView,
    QVBoxLayout,
    QListWidget,
)
import matplotlib.figure

from pyNexafs.gui.widgets.graphing.mpl_graphs import FigureCanvas
from pyNexafs.gui.widgets.fileloader import nexafs_fileloader
from pyNexafs.nexafs.scan import scan_base

import numpy as np


class viewerWidget(QVBoxLayout):
    def __init__(self):
        super().__init__()

        # Initialise elements
        self._selected_files = None
        self._dataseries_list = []  # List of common labels in current selection.
        self._dataseries_list_view = QListWidget()
        self._canvas = FigureCanvas(matplotlib.figure.Figure())  # use empty canvas.
        self._scan_objects = None
        self._selected_dataseries = None

        # Add to layout
        self.addWidget(QLabel("Data Series"))
        self.addWidget(self._dataseries_list_view)
        self.addWidget(self._canvas)

        # Attributes
        self._dataseries_list_view.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )

        # Connections
        self._dataseries_list_view.itemSelectionChanged.connect(
            self.on_label_selection_change
        )

    @property
    def canvas_figure(self):
        return self._canvas.figure

    @property
    def labels(self) -> list[str]:
        return self._dataseries_list

    @property
    def scans(self) -> dict[str, scan_base]:
        """
        Scan objects loaded into the viewer.

        Returns
        -------
        dict[str, scan_base]
            A shallow copy of the dictionary object containing scans.
        """
        return self._scan_objects.copy() if self._scan_objects is not None else None

    @scans.setter
    def scans(self, scans: dict[str, scan_base]):
        """
        Setter for the scan objects.

        Adds new scan objects and updates existing scan objects with the same filename.

        Parameters
        ----------
        scans : dict[str, scan_base]
            A dictionary of filename and corresponding scan objects.
        """
        if self._scan_objects is None:
            self._scan_objects = scans
        else:
            self._scan_objects.update(scans)

    @scans.deleter
    def scans(self):
        """
        Flushes the existing scan objects from the viewer.
        """
        self._scan_objects = None

    def set_file_selection(self, names: list[str] | None):
        """
        Sets the current selection of scans to be displayed.

        Parameters
        ----------
        names : list[str]
            List of filenames to display.
        """
        self._selected_files = names
        if names is not None:
            # Find common labels:
            scans = self.scans
            if len(names) > 1:
                labels = set(scans[names[0]]._y_labels)
                for name in names[1:]:
                    if name in scans:
                        labels = labels.intersection(set(scans[name]._y_labels))
            elif len(names) == 1:
                labels = scans[names[0]]._y_labels
            else:
                labels = []
            # Update attributes
            self._dataseries_list = list(labels)
            # Update graphics
            self._dataseries_list_view.clear()
            self._dataseries_list_view.addItems(self._dataseries_list)

    @property
    def selected_dataseries(self) -> list[str]:
        """
        The currently selected data series common to all scan objects.

        Returns
        -------
        list[str]
            List of strings of the data series selected.
        """
        return self._dataseries_list.copy()

    def on_label_selection_change(self) -> None:
        """
        Callback for when the selected labels change.
        """
        scans_subset = {
            name: scan
            for name, scan in self.scans.items()
            if name in self._selected_files
        }

    def graph_selection(self, names: list[str]):
        print(names)
        print(self.canvas_figure)
        ax = self.canvas_figure.add_subplot(111)

        x = np.linspace(0, 10, 100)
        rand_y = np.random.rand(100)

        ax.plot(x, rand_y)
        pass


class mainWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Initialise elements
        self.main_layout = QHBoxLayout()
        self.loader = nexafs_fileloader()
        self.viewer = viewerWidget()
        # self.converter = converterWidget()

        # Add to layout
        self.main_layout.addWidget(self.loader)
        self.main_layout.addLayout(self.viewer)
        # self.sub_layout.addWidget(self.converter)
        self.setLayout(self.main_layout)

        ### Connections
        # Load scans
        self.loader.selectionLoaded.connect(self._on_load_selection)
        # Remove scans with parser / directory change.
        self.loader.directory_selector.newPath.connect(self._on_dir_parser_change)
        self.loader.nexafs_parser_selector.currentIndexChanged.connect(
            self._on_dir_parser_change
        )

    def _on_dir_parser_change(self):
        del self.viewer.scans

    def _on_load_selection(self):
        # Load in the scan objects
        selection_parse_objs = self.loader.loaded_selection
        scans_copy = self.viewer.scans
        if scans_copy is not None:
            for name, parser in selection_parse_objs.items():
                if name not in scans_copy:
                    self.viewer.scans = {name: parser.to_scan(True)}
                else:
                    # If parser object is not the same, reload the scan object.
                    if scans_copy[name].parser is not parser:
                        self.viewer.scans = {name: parser.to_scan(True)}
        else:
            self.viewer.scans = {
                name: parser.to_scan(True)
                for name, parser in selection_parse_objs.items()
            }

        # Plot the scan objects
        self.viewer.set_file_selection(self.loader.selected_filenames)


def gui():
    app = QApplication(sys.argv)
    window = mainWidget()
    window.show()
    window.setWindowTitle("pyNexafs File Loader")
    app.exec()


if __name__ == "__main__":
    gui()
