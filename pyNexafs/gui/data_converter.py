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
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTB

import numpy as np


class viewerWidget(QVBoxLayout):
    def __init__(self):
        super().__init__()

        # Initialise elements
        self._selected_files = None
        self._dataseries_list = []  # List of common labels in current selection.
        self._dataseries_list_view = QListWidget()
        self._figure = matplotlib.figure.Figure()
        self._canvas = FigureCanvas(self._figure)  # use empty canvas.
        self._navtoolbar = NavTB(self._canvas)
        
        self._scan_objects = None
        self._selected_dataseries = None

        # Add to layout
        self.addWidget(QLabel("Data Series"))
        self.addWidget(self._dataseries_list_view)
        self.addWidget(self._navtoolbar)
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
            # Find common labels in scan objects:
            scans = self.scans
            if len(names) > 1:
                # Look through all scan objects
                scan = scans[names[0]]
                labels = (
                    set(scan._y_labels) if scan is not None else []
                )  # set to empty list if scan object is None.
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
        selected_items = self._dataseries_list_view.selectedItems()
        if len(selected_items) > 0:
            return [item.text() for item in selected_items]
        else:
            return None

    def on_label_selection_change(self) -> None:
        """
        Callback for when the selected labels change.
        """
        # Get the current scan object selection.
        if self._selected_files is not None and len(self._selected_files) > 0:
            scans_subset = {
                name: scan
                for name, scan in self.scans.items()
                if name in self._selected_files
            }
            # Get the selected fields.
            ds_list = self.selected_dataseries
            if ds_list is not None and len(ds_list) > 0: 
                # Plot onto a graph
                self.graph_selection(scans_subset, ds_list)

    def graph_selection(self, scans: dict[str, scan_base], dataseries_list: list[str]) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        # Iterate over dataseries first:
        for ds in dataseries_list:
            # Then iterate over each scan object.
            for name, scan in scans.items():
                try:
                    ind = scan._y_labels.index(ds)
                    x = scan.x
                    y = scan.y[:, ind]
                    ax.plot(x, y, label=name + ":" + ds)
                except ValueError:
                    # Catch Value errors if the label is not found.
                    continue
        self._figure.legend()
        self._canvas.draw()
        # self._canvas.
        

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
