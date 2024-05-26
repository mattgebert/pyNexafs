from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QSplitter,
)
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel
import sys
import matplotlib, matplotlib.axes, matplotlib.figure

from pyNexafs.nexafs.scan import scan_abstract
from pyNexafs.parsers import parser_base
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas
from pyNexafs.gui.widgets.normaliser import NavTBQT_Norm, normaliserSettings

import numpy as np


class nexafsViewer(QWidget):
    """
    General viewer for NEXAFS parser objects.

    Takes
    Includes the use of a `normalisingGraph` to display and normalise NEXAFS data.


    """

    def __init__(self, norm_settings: normaliserSettings = None, parent=None):
        super().__init__(parent)
        self._norm_settings = norm_settings

        # Initialise elements
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)
        self._draggable = QSplitter(Qt.Orientation.Vertical)

        # Initialise properties
        self._selected_filenames = []
        self._scan_objects = {}
        self._dataseries_list = []  # List of common labels in current selection.

        # Initialise viewing elements
        self._dataseries_list_view = QListWidget()
        self._normGraph = normalisingGraph()

        # Add to layout
        self._draggable.addWidget(QLabel("Data Series"))
        self._draggable.addWidget(self._dataseries_list_view)
        self._draggable.addWidget(self._normGraph)
        self._layout.addWidget(self._draggable)

        # Attributes
        self._dataseries_list_view.setSelectionMode(
            QListWidget.SelectionMode.ExtendedSelection
        )

        # Connections
        self._dataseries_list_view.itemSelectionChanged.connect(
            self.on_label_selection_change
        )

    @property
    def dataseries(self) -> list[str]:
        """
        Returns the y_label intersection of currently selected scan objects.

        Returns
        -------
        list[str]
            Labels common to selected scan objects.
        """
        return self._dataseries_list

    def add_parsers_to_scans(
        self,
        parsers: dict[str, type[parser_base]] | list[type[parser_base]],
        load_all_columns: bool = True,
        override_scans: bool = False,
    ) -> None:
        """
        Add parsers to the scan list, through conversion to scan objects.

        If an scan object exists with the same filename, the `override_scans` flag will determine whether to replace the existing reference.

        Parameters
        ----------
        parsers : dict[str, scan_abstract] | list[type[parser_base]]
            Dictionary of filename and corresponding parser objects, or a list of parser objects upon which filenames will be interred from the filename attribute.
        load_all_columns : bool
            Load all columns unreferenced by `COLUMN_ASSIGNMENTS` parser object attribute, by default True.
        override_scans : bool
            Override existing scan objects with the same filename, by default False.
        """
        # Convert parsers list to dict.
        if isinstance(parsers, list):
            parsers = {parser.filename: parser for parser in parsers}
        # Update scans with new filename parsers unless override flag is set.
        scans = self.scans
        self.scans = {
            filename: parser.to_scan(load_all_columns)
            for filename, parser in parsers.items()
            if (filename not in scans or override_scans)
        }

    @property
    def scans(self) -> dict[str, type[scan_abstract]]:
        """
        Scan objects loaded into the viewer.

        Returns
        -------
        dict[str, scan_abstract]
            A shallow copy of the dictionary object containing scans.
        """
        return self._scan_objects.copy()

    @scans.setter
    def scans(self, scans: dict[str, scan_abstract]):
        """
        Setter for the scan objects.

        Adds new scan objects and updates existing scan objects with the same filename.

        Parameters
        ----------
        scans : dict[str, scan_abstract]
            A dictionary of filename keys and corresponding scan object values.
        """
        self._scan_objects.update(scans)

    @scans.deleter
    def scans(self):
        """
        Flushes the existing scan objects from the viewer.
        """
        self._scan_objects = {}

    @property
    def selected_filenames(self) -> list[str]:
        """
        The current selection of files. Empty list if None.

        Returns
        -------
        list[str]
            List of filenames selected.
        """
        return self._selected_filenames

    @selected_filenames.setter
    def selected_filenames(self, names: list[str]):
        """
        Setter for the selected files.

        Parameters
        ----------
        names : list[str]
            List of filenames to select.
        """
        # Update internal object
        self._selected_filenames = names if names is not None else []

        # Update the data series list
        if len(names) != 0:
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
        if self._selected_filenames is not None and len(self._selected_filenames) > 0:
            scans_subset = {
                name: scan
                for name, scan in self.scans.items()
                if name in self._selected_filenames
            }
            # Get the selected fields.
            ds_list = self.dataseries_selected
            # Propogate dataseries selection to normalisation widget.
            self._normGraph.graph_scans = scans_subset
            self._normGraph.dataseries_selection = ds_list
            self._normGraph.graph_selection()

    def on_normalisation_change(self) -> None:
        """
        Callback for when the normalisation changes.
        """
        self._norm_settings = self._normGraph.norm_settings


class normalisingGraph(QWidget):
    def __init__(
        self,
        graph_scans: dict[str, scan_abstract] = None,
        dataseries_selection: list[str] = [],
        background_fixed_scans: list[scan_abstract] = [],
        norm_settings: normaliserSettings = None,
        parent=None,
    ):
        super().__init__(parent)
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)
        self._figure = matplotlib.figure.Figure()
        self._canvas = FigureCanvas(self._figure)  # use empty canvas.
        self._toolbar = NavTBQT_Norm(self._canvas)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)

        # Update toolbar settings.
        self._toolbar.graph_scans = graph_scans
        self._toolbar.dataseries_selection = dataseries_selection
        self._toolbar.background_fixed_scans = background_fixed_scans
        self._toolbar.norm_settings = norm_settings

    @property
    def graph_scans(self) -> dict[str, scan_abstract]:
        return self._toolbar.graph_scans

    @graph_scans.setter
    def graph_scans(self, scans: dict[str, scan_abstract]):
        self.toolbar.graph_scans = scans

    @property
    def dataseries_selection(self) -> list[str]:
        return self._toolbar.dataseries_selection

    @dataseries_selection.setter
    def dataseries_selection(self, labels: list[str]):
        self.toolbar.dataseries_selection = labels

    @property
    def background_fixed_scans(self) -> list[scan_abstract]:
        return self._toolbar.background_fixed_scans

    @background_fixed_scans.setter
    def background_fixed_scans(self, scans: list[scan_abstract]):
        self.toolbar.background_fixed_scans = scans

    @property
    def norm_settings(self) -> normaliserSettings:
        return self._toolbar.norm_settings

    @norm_settings.setter
    def norm_settings(self, settings: normaliserSettings):
        self.toolbar.norm_settings = settings

    @property
    def figure(self) -> matplotlib.figure.Figure:
        return self._figure

    @figure.setter
    def figure(self, fig: matplotlib.figure.Figure):
        self._figure = fig
        self._canvas.figure = fig

    @property
    def canvas(self) -> FigureCanvas:
        return self._canvas

    @property
    def toolbar(self):
        return self._toolbar

    def graph_selection(self) -> None:
        scans = self.graph_scans
        dataseries_list = self.dataseries_selection
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        assert isinstance(ax, matplotlib.axes.Axes)
        # Iterate over dataseries first:
        if dataseries_list is not None:
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
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = nexafsViewer()
    window.setWindowTitle("pyNexafs nexafsViewer")
    # window = normalisingGraph()
    # window.setWindowTitle("pyNexafs Normalising Graph")
    window.show()
    sys.exit(app.exec())
