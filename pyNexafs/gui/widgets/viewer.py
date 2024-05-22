from PyQt6.QtWidgets import (
    QWidget,
    QFileDialog,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QComboBox,
    QFrame,
    QGridLayout,
    QSizeGrip,
    QSplitter,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QTextEdit
from PyQt6.QtGui import QColor, QPalette
import os, sys
import matplotlib

# from pyNexafs.parsers import parser_loaders, parser_base
from pyNexafs.nexafs.scan import scan_abstract

# from pyNexafs.gui.widgets.graphing.plotly_graphs import PlotlyGraph
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import (
    FigureCanvas,
    NavTBQT,
    NEXAFS_NavQT,
)
from pyNexafs.gui.widgets.normaliser import NavTBQT_Norm
from pyNexafs.gui.widgets.fileloader import nexafsFileLoader

from typing import Type
import random


class nexafsViewer(QWidget):
    """
    General viewer for NEXAFS data.

    Includes various submodules, including directory browser, data viewer, and plot viewer.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialise elements
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)
        self._draggable = QSplitter(Qt.Orientation.Vertical)

        self._selected_files = None
        self._dataseries_list = []  # List of common labels in current selection.
        self._dataseries_list_view = QListWidget()
        self._figure = matplotlib.figure.Figure()
        self._canvas = FigureCanvas(self._figure)  # use empty canvas.
        self._navtoolbar = NavTBQT_Norm(self._canvas)
        # self._navtoolbar = NEXAFS_NavQT(self._canvas)
        # self._navtoolbar = NavTBQT(self._canvas)

        self._scan_objects = None
        self._dataseries_selected = None

        # Add to layout
        self._draggable.addWidget(QLabel("Data Series"))
        self._draggable.addWidget(self._dataseries_list_view)
        self._draggable.addWidget(self._navtoolbar)
        self._draggable.addWidget(self._canvas)
        self._layout.addWidget(self._draggable)

        # Attributes
        self._dataseries_list_view.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )

        # Connections
        self._dataseries_list_view.itemSelectionChanged.connect(
            self.on_label_selection_change
        )
        self._navtoolbar.normalisationUpdate.connect(self.on_normalisation_change)

    @property
    def canvas_figure(self):
        return self._canvas.figure

    @property
    def dataseries(self) -> list[str]:
        return self._dataseries_list

    @property
    def scans(self) -> dict[str, scan_abstract]:
        """
        Scan objects loaded into the viewer.

        Returns
        -------
        dict[str, scan_abstract]
            A shallow copy of the dictionary object containing scans.
        """
        return self._scan_objects.copy() if self._scan_objects is not None else None

    @scans.setter
    def scans(self, scans: dict[str, scan_abstract]):
        """
        Setter for the scan objects.

        Adds new scan objects and updates existing scan objects with the same filename.

        Parameters
        ----------
        scans : dict[str, scan_abstract]
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
                scan = scans[names[0]]
                labels = scan._y_labels
            else:
                labels = []
                scan = None
            # Update attributes in order, keeping original appearance order.
            ordered_intersection = []
            if scan is not None:
                for label in scan._y_labels:
                    if label in labels:
                        ordered_intersection.append(label)
            else:
                ordered_intersection = []
            self._dataseries_list = ordered_intersection
            # Update graphics
            self._dataseries_list_view.clear()
            self._dataseries_list_view.addItems(self._dataseries_list)

    @property
    def dataseries_selected(self) -> list[str] | None:
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

    @dataseries_selected.setter
    def dataseries_selected(self, labels: list[str]):
        """
        Setter for the selected data series.

        Parameters
        ----------
        labels : list[str]
            List of labels to select.
        """
        for i in range(self._dataseries_list_view.count()):
            item = self._dataseries_list_view.item(i)
            if item.text() in labels:
                item.setSelected(True)
            else:
                item.setSelected(False)

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
            ds_list = self.dataseries_selected
            if ds_list is not None and len(ds_list) > 0:
                # Plot onto a graph
                self.graph_selection(scans_subset, ds_list)

    def on_normalisation_change(self) -> None:
        """
        Callback for when the normalisation changes.
        """
        pass

    def graph_selection(
        self, scans: dict[str, scan_abstract], dataseries_list: list[str]
    ) -> None:
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = nexafsViewer()
    window.show()
    window.setWindowTitle("pyNexafs Viewer")
    sys.exit(app.exec())
