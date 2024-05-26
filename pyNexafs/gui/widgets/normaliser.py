import numpy as np
import matplotlib as mpl
import matplotlib.figure as mpl_figure
from enum import Enum
from typing import Type
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt, pyqtBoundSignal
from pyNexafs.nexafs import (
    scan_abstract,
    scan_normalised_edges,
    scan_normalised,
    scan_background_subtraction,
)
from pyNexafs.resources import ICONS
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import (
    FigureCanvas,
    NEXAFS_NavQT,
    NavTBQT,
)


class backgroundMethod(Enum):
    NONE = 0
    MOST_RECENT_KEYWORD = 1
    CLOSEST_TIME_KEYWORD = 2
    FIXED_SCAN = 3


class normaliserSettings:
    def __init__(
        self,
        background_method: backgroundMethod = backgroundMethod.NONE,
        pre_edge_norm_method: scan_normalised_edges.PREEDGE_NORM_TYPE = scan_normalised_edges.PREEDGE_NORM_TYPE.NONE,
        post_edge_norm_method: scan_normalised_edges.POSTEDGE_NORM_TYPE = scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE,
        background_scan: scan_abstract = None,
        background_filter_str: str = None,
        pre_edge_domain: tuple[float, float] = (None, None),
        post_edge_domain: tuple[float, float] = (None, None),
        pre_edge_level: float | None = None,
        post_edge_level: float | None = None,
    ):
        self._background_method = background_method
        self._pre_edge_norm_method = pre_edge_norm_method
        self._post_edge_norm_method = post_edge_norm_method
        self._background_scan = background_scan
        self._background_filter_str = background_filter_str
        self._pre_edge_domain = pre_edge_domain
        self._post_edge_domain = post_edge_domain
        self._pre_edge_level = pre_edge_level
        self._post_edge_level = post_edge_level

    @property
    def background_method(self) -> backgroundMethod:
        return self._background_method

    @background_method.setter
    def background_method(self, method: backgroundMethod):
        self._background_method = method

    @property
    def pre_edge_norm_method(self) -> scan_normalised_edges.PREEDGE_NORM_TYPE:
        return self._pre_edge_norm_method

    @pre_edge_norm_method.setter
    def pre_edge_norm_method(self, method: scan_normalised_edges.PREEDGE_NORM_TYPE):
        self._pre_edge_norm_method = method

    @property
    def post_edge_norm_method(self) -> scan_normalised_edges.POSTEDGE_NORM_TYPE:
        return self._post_edge_norm_method

    @post_edge_norm_method.setter
    def post_edge_norm_method(self, method: scan_normalised_edges.POSTEDGE_NORM_TYPE):
        self._post_edge_norm_method = method

    @property
    def background_scan(self) -> scan_abstract:
        return self._background_scan

    @background_scan.setter
    def background_scan(self, scan: Type[scan_abstract]):
        self._background_scan = scan

    @property
    def background_filter(self) -> str:
        return self._background_filter_str

    @background_filter.setter
    def background_filter(self, filter_str: str):
        self._background_filter_str = filter_str

    @property
    def pre_edge_domain(self) -> tuple[float, float]:
        return self._pre_edge_domain

    @pre_edge_domain.setter
    def pre_edge_domain(self, domain: tuple[float, float]):
        self._pre_edge_domain = domain

    @property
    def post_edge_domain(self) -> tuple[float, float]:
        return self._post_edge_domain

    @post_edge_domain.setter
    def post_edge_domain(self, domain: tuple[float, float]):
        self._post_edge_domain = domain

    @property
    def pre_edge_level(self) -> float | None:
        return self._pre_edge_level

    @pre_edge_level.setter
    def pre_edge_level(self, level: float | None):
        self._pre_edge_level = level

    @property
    def post_edge_level(self) -> float | None:
        return self._post_edge_level


class channelNormaliser(QtWidgets.QWidget):
    """
    Normaliser widget for choosing a scan object and corresponding channel to normalise by.

    Parameters
    ----------


    Attributes
    ----------

    """

    channelUpdated = pyqtBoundSignal(bool)

    def __init__(
        self,
        channel_method: backgroundMethod = backgroundMethod.NONE,
        channel_list: list[str] = [],
        scan_list: list[Type[scan_abstract]] = [],
        parent=None,
    ):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        ## Elements
        # Channel method selection
        self._channel_combobox = QtWidgets.QComboBox()
        self._channel_combobox.addItems(
            [
                name.capitalize().replace("_", " ")
                for name in backgroundMethod._member_names_
            ]
        )
        self._channel_combobox.setCurrentIndex(channel_method.value)
        # Scan object selection
        self._scan_combobox = QtWidgets.QComboBox()
        self._scan_combobox.addItems([scan.filename for scan in scan_list])
        # Channel selection
        self._channel_combobox = QtWidgets.QComboBox()
        self._channel_combobox.addItems(channel_list)
        # Selection Filter
        self._filter = QtWidgets.QLineEdit()
        self._filter.setPlaceholderText(
            "Enter keyword or regular expression to match scan."
        )

        # Initialise properties
        self.channel_method = channel_method
        self.scan_list = scan_list
        self.channel_list = channel_list

    @property
    def channel_method(self) -> backgroundMethod:
        return self._channel_method

    @channel_method.setter
    def channel_method(self, method: backgroundMethod):
        self._channel_method = method
        if self._channel_combobox.currentIndex() != method.value:
            self._channel_combobox.setCurrentIndex(method.value)

    @property
    def scan_list(self) -> list[Type[scan_abstract]]:
        return self._scan_list

    @scan_list.setter
    def scan_list(self, scans: list[Type[scan_abstract]]):
        self._scan_list = scans

    @property
    def channel_list(self) -> list[str]:
        return self._channel_list

    @channel_list.setter
    def channel_list(self, channels: list[str]):
        current_label = self._channel_combobox.currentText()
        self._channel_list = channels
        self._channel_combobox.clear()
        self._channel_combobox.addItems(channels)
        if current_label in channels:
            self._channel_combobox.setCurrentIndex(channels.index(current_label))


class scanNormaliser(QtWidgets.QWidget):
    def __init__(
        self,
        graph_scans: list[scan_abstract],
        dataseries_selection: list[str],
        background_fixed_scans: list[scan_abstract] = [],
        normalisation_settings: normaliserSettings = None,
        parent=None,
    ):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)

        # Initialise elements
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # Initalise scan properties, assign args later.
        self._graph_scans = []  # List of scans objects to be normalised and graphed.
        self._background_fixed_scans = (
            []
        )  # List of all scan|parser objects that may be used for background subtraction.
        self._dataseries_selected = []  # List of selected dataseries for plotting.

        # Graphical element
        self._figure = mpl_figure.Figure()
        self._canvas = FigureCanvas(self._figure)  # use empty canvas.
        # self._navtoolbar = NEXAFS_NavQT(self._canvas)
        self._navtoolbar = NavTBQT(self._canvas)

        # Background Subtraction elements
        bold_font = QtGui.QFont("Arial", 10, weight=QtGui.QFont.Weight.Bold)
        bkgd_layout_0 = QtWidgets.QGridLayout()
        self._background_method_combobox = QtWidgets.QComboBox()
        self._background_method_combobox.addItems(
            [
                name.capitalize().replace("_", " ")
                for name in backgroundMethod._member_names_
            ]
        )
        self._background_filter = QtWidgets.QLineEdit()
        self._background_filter.setPlaceholderText(
            "Enter keyword or regular expression to match background scan."
        )
        self._background_scan = None
        self._background_scan_label = QtWidgets.QLabel("Scan:")
        self._background_scan_combobox = QtWidgets.QComboBox()

        bkgd_layout_0.addWidget(
            QtWidgets.QLabel("Background Removal", font=bold_font), 0, 0, 1, 3
        )
        bkgd_layout_0.addWidget(QtWidgets.QLabel("Background Method:"), 1, 0, 1, 1)
        bkgd_layout_0.addWidget(self._background_method_combobox, 1, 1, 1, 2)
        bkgd_layout_0.addWidget(self._background_filter, 2, 0, 1, 3)
        bkgd_layout_0.addWidget(self._background_scan_label, 2, 0, 1, 1)
        bkgd_layout_0.addWidget(self._background_scan_combobox, 2, 1, 1, 2)

        # Channel normalisation elements:
        norm_layout_1 = QtWidgets.QGridLayout()
        # self._

        # Edge Normalisation elements
        norm_layout_0 = QtWidgets.QGridLayout()
        self._pre_edge_combobox = QtWidgets.QComboBox()
        self._post_edge_combobox = QtWidgets.QComboBox()
        self._pre_edge_combobox.addItems(
            [
                name.capitalize().replace("_", " ")
                for name in scan_normalised_edges.PREEDGE_NORM_TYPE._member_names_
            ]
        )
        self._post_edge_combobox.addItems(
            [
                name.capitalize().replace("_", " ")
                for name in scan_normalised_edges.POSTEDGE_NORM_TYPE._member_names_
            ]
        )
        self._pre_edge_domain_label = QtWidgets.QLabel("Domain: ")
        self._pre_edge_domain_x0 = QtWidgets.QLineEdit()
        self._pre_edge_domain_x1 = QtWidgets.QLineEdit()
        self._pre_edge_level_label = QtWidgets.QLabel("Baseline Level: ")
        self._pre_edge_level = QtWidgets.QLineEdit()

        self._post_edge_domain_label = QtWidgets.QLabel("Domain: ")
        self._post_edge_domain_x0 = QtWidgets.QLineEdit()
        self._post_edge_domain_x1 = QtWidgets.QLineEdit()
        self._post_edge_level_label = QtWidgets.QLabel("Baseline Level: ")
        self._post_edge_level = QtWidgets.QLineEdit()

        # Preedge
        self._pre_edge_level.setValidator(QtGui.QDoubleValidator())
        self._pre_edge_domain_x0.setPlaceholderText("x0")
        self._pre_edge_domain_x0.setValidator(QtGui.QDoubleValidator())
        self._pre_edge_domain_x1.setPlaceholderText("x1")
        self._pre_edge_domain_x1.setValidator(QtGui.QDoubleValidator())
        # Postedge
        self._post_edge_level.setValidator(QtGui.QDoubleValidator())
        self._post_edge_domain_x0.setPlaceholderText("x0")
        self._post_edge_domain_x0.setValidator(QtGui.QDoubleValidator())
        self._post_edge_domain_x1.setPlaceholderText("x1")
        self._post_edge_domain_x1.setValidator(QtGui.QDoubleValidator())

        # Layout
        norm_layout_0.addWidget(
            QtWidgets.QLabel("Normalisation Options", font=bold_font), 0, 0, 1, 6
        )
        norm_layout_0.addWidget(QtWidgets.QLabel("Pre-edge:"), 1, 0, 1, 3)
        norm_layout_0.addWidget(QtWidgets.QLabel("Post-edge:"), 1, 3, 1, 3)
        norm_layout_0.addWidget(self._pre_edge_combobox, 1, 1, 1, 2)
        norm_layout_0.addWidget(self._post_edge_combobox, 1, 4, 1, 2)
        norm_layout_0.addWidget(self._pre_edge_domain_label, 2, 0, 1, 1)
        norm_layout_0.addWidget(self._pre_edge_domain_x0, 2, 1, 1, 1)
        norm_layout_0.addWidget(self._pre_edge_domain_x1, 2, 2, 1, 1)
        norm_layout_0.addWidget(self._post_edge_domain_label, 2, 3, 1, 1)
        norm_layout_0.addWidget(self._post_edge_domain_x0, 2, 4, 1, 1)
        norm_layout_0.addWidget(self._post_edge_domain_x1, 2, 5, 1, 1)
        norm_layout_0.addWidget(self._pre_edge_level_label, 3, 0, 1, 1)
        norm_layout_0.addWidget(self._pre_edge_level, 3, 1, 1, 2)
        norm_layout_0.addWidget(self._post_edge_level_label, 3, 3, 1, 1)
        norm_layout_0.addWidget(self._post_edge_level, 3, 4, 1, 2)

        # Settings layout
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        line.setLineWidth(1)
        line.setMinimumWidth(1)
        line.setStyleSheet("background-color: black;")
        line.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding
        )
        settings_layout = QtWidgets.QHBoxLayout()
        settings_layout.addLayout(bkgd_layout_0)
        settings_layout.addWidget(line)

        settings_layout.addWidget(line)
        settings_layout.addLayout(norm_layout_0)
        settings_layout.setStretch(0, 1)
        settings_layout.setStretch(2, 2)

        # Connections
        self._background_method_combobox.currentIndexChanged.connect(
            self._on_background_method_change
        )
        self._background_scan_combobox.currentIndexChanged.connect(
            self._on_background_fixed_scans_change
        )
        self._pre_edge_combobox.currentIndexChanged.connect(
            self._on_pre_edge_method_change
        )
        self._post_edge_combobox.currentIndexChanged.connect(
            self._on_post_edge_method_change
        )

        # Add to layout
        self._layout.addLayout(settings_layout)
        self._layout.addWidget(self._navtoolbar)
        self._layout.addWidget(self._canvas)

        ### Intialise UI settings
        # Set default background method, also showing/hiding elements.
        self.background_method = backgroundMethod.NONE
        self.pre_edge_normalisation_method = (
            scan_normalised_edges.PREEDGE_NORM_TYPE.NONE
        )
        self.post_edge_normalisation_method = (
            scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE
        )

        # Assign args
        self.graph_scans = graph_scans
        self.background_fixed_scans = background_fixed_scans
        self.dataseries_selected = dataseries_selection
        if normalisation_settings is not None:
            self.settings = normalisation_settings

    @property
    def background_method(self) -> backgroundMethod:
        return self._background_method

    @background_method.setter
    def background_method(self, method: backgroundMethod):
        self._background_method = method
        # Update index if unmatched
        if self._background_method_combobox.currentIndex() != method.value:
            self._background_method_combobox.setCurrentIndex(method.value)
        # Enable / Disable filter
        if method == backgroundMethod.NONE or method == backgroundMethod.FIXED_SCAN:
            self._background_filter.setEnabled(False)
            self._background_filter.hide()
        else:
            self._background_filter.setEnabled(True)
            self._background_filter.show()

        # Enable / Disable scan selection
        if method == backgroundMethod.FIXED_SCAN:
            self._background_scan_combobox.setEnabled(True)
            self._background_scan_combobox.show()
            self._background_scan_label.setEnabled(True)
            self._background_scan_label.show()
        else:
            self._background_scan_combobox.setEnabled(False)
            self._background_scan_combobox.hide()
            self._background_scan_label.setEnabled(False)
            self._background_scan_label.hide()

    @property
    def background_scan(self):
        return self._background_scan

    @background_scan.setter
    def background_scan(self, scan: Type[scan_abstract]):
        self._background_scan = scan
        if self._background_method != backgroundMethod.FIXED_SCAN:
            self._background_filter.setEnabled(False)
            self._background_method_combobox.setCurrentIndex(
                backgroundMethod.FIXED_SCAN.value
            )

    @property
    def background_filter_str(self) -> str:
        return self._background_filter.text()

    @background_filter_str.setter
    def background_filter_str(self, filter: str):
        self._background_filter.setText(filter)

    def _on_background_method_change(self):
        # Get new method:
        self.background_method = backgroundMethod(
            self._background_method_combobox.currentIndex()
        )

    def _on_background_fixed_scans_change(self):
        # Set new scan reference
        if len(self._background_fixed_scans) > 0:
            self.background_scan = self._background_fixed_scans[
                self._background_scan_combobox.currentIndex()
            ]

    @property
    def pre_edge_normalisation_method(self) -> scan_normalised_edges.PREEDGE_NORM_TYPE:
        return self._pre_edge_normalisation_method

    @pre_edge_normalisation_method.setter
    def pre_edge_normalisation_method(
        self, method: scan_normalised_edges.PREEDGE_NORM_TYPE
    ):
        self._pre_edge_normalisation_method = method
        if self._pre_edge_combobox.currentIndex() != method.value:
            self._post_edge_combobox.setCurrentIndex(method.value)
        if method == scan_normalised_edges.PREEDGE_NORM_TYPE.NONE:
            self._pre_edge_domain_x0.setEnabled(False)
            self._pre_edge_domain_x1.setEnabled(False)
            self._pre_edge_domain_label.setEnabled(False)
            self._pre_edge_level.setEnabled(False)
            self._pre_edge_level_label.setEnabled(False)
            self._pre_edge_domain_x0.hide()
            self._pre_edge_domain_x1.hide()
            self._pre_edge_domain_label.hide()
            self._pre_edge_level.hide()
            self._pre_edge_level_label.hide()
        else:
            self._pre_edge_domain_x0.setEnabled(True)
            self._pre_edge_domain_x1.setEnabled(True)
            self._pre_edge_domain_label.setEnabled(True)
            self._pre_edge_level.setEnabled(True)
            self._pre_edge_level_label.setEnabled(True)
            self._pre_edge_domain_x0.show()
            self._pre_edge_domain_x1.show()
            self._pre_edge_domain_label.show()
            self._pre_edge_level.show()
            self._pre_edge_level_label.show()

    def _on_pre_edge_method_change(self):
        self.pre_edge_normalisation_method = scan_normalised_edges.PREEDGE_NORM_TYPE(
            self._pre_edge_combobox.currentIndex()
        )

    @property
    def post_edge_normalisation_method(
        self,
    ) -> scan_normalised_edges.POSTEDGE_NORM_TYPE:
        return self._post_edge_normalisation_method

    @post_edge_normalisation_method.setter
    def post_edge_normalisation_method(
        self, method: scan_normalised_edges.POSTEDGE_NORM_TYPE
    ):
        self._post_edge_normalisation_method = method
        if self._post_edge_combobox.currentIndex() != method.value:
            self._post_edge_combobox.setCurrentIndex(method.value)
        if method == scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE:
            self._post_edge_domain_x0.setEnabled(False)
            self._post_edge_domain_x1.setEnabled(False)
            self._post_edge_domain_label.setEnabled(False)
            self._post_edge_level.setEnabled(False)
            self._post_edge_level_label.setEnabled(False)
            self._post_edge_domain_x0.hide()
            self._post_edge_domain_x1.hide()
            self._post_edge_domain_label.hide()
            self._post_edge_level.hide()
            self._post_edge_level_label.hide()
        else:
            self._post_edge_domain_x0.setEnabled(True)
            self._post_edge_domain_x1.setEnabled(True)
            self._post_edge_domain_label.setEnabled(True)
            self._post_edge_level.setEnabled(True)
            self._post_edge_level_label.setEnabled(True)
            self._post_edge_domain_x0.show()
            self._post_edge_domain_x1.show()
            self._post_edge_domain_label.show()
            self._post_edge_level.show()
            self._post_edge_level_label.show()

    def _on_post_edge_method_change(self):
        self.post_edge_normalisation_method = scan_normalised_edges.POSTEDGE_NORM_TYPE(
            self._post_edge_combobox.currentIndex()
        )

    @property
    def pre_edge_domain(self) -> tuple[float, float]:
        x0 = self._pre_edge_domain_x0.text()
        x1 = self._pre_edge_domain_x1.text()
        x0 = float(x0) if x0 != "" else None
        x1 = float(x1) if x1 != "" else None
        return (x0, x1) if x0 is not None and x1 is not None else None

    @pre_edge_domain.setter
    def pre_edge_domain(self, domain: tuple[float | None, float | None] | None):
        if domain is None:
            self._pre_edge_domain_x0.setText("")
            self._pre_edge_domain_x1.setText("")
        else:
            self._pre_edge_domain_x0.setText(
                str(domain[0]) if domain[0] is not None else ""
            )
            self._pre_edge_domain_x1.setText(
                str(domain[1]) if domain[1] is not None else ""
            )

    @property
    def post_edge_domain(self) -> tuple[float | None, float | None]:
        x0 = self._post_edge_domain_x0.text()
        x1 = self._post_edge_domain_x1.text()
        x0 = float(x0) if x0 != "" else None
        x1 = float(x1) if x1 != "" else None
        return (x0, x1) if x0 is not None and x1 is not None else None

    @post_edge_domain.setter
    def post_edge_domain(self, domain: tuple[float | None, float | None] | None):
        if domain is None:
            self._post_edge_domain_x0.setText("")
            self._post_edge_domain_x1.setText("")
        else:
            self._post_edge_domain_x0.setText(
                str(domain[0]) if domain[0] is not None else ""
            )
            self._post_edge_domain_x1.setText(
                str(domain[1]) if domain[1] is not None else ""
            )

    @property
    def pre_edge_level(self) -> float | None:
        A = self._pre_edge_level.text()
        return float(A) if A != "" else None

    @pre_edge_level.setter
    def pre_edge_level(self, level: float | None):
        self._pre_edge_level.setText(str(level) if level is not None else "")

    @property
    def post_edge_level(self) -> float | None:
        A = self._post_edge_level.text()
        return float(A) if A != "" else None

    @post_edge_level.setter
    def post_edge_level(self, level: float | None):
        self._post_edge_level.setText(str(level) if level is not None else "")

    @property
    def graph_scans(self) -> list[scan_abstract]:
        return self._graph_scans

    @graph_scans.setter
    def graph_scans(self, scans: list[scan_abstract] | None):
        self._graph_scans = scans
        # Check validity of selected dataseries with new scans, if not delete entries.
        if isinstance(scans, list) and len(scans) > 0:
            if len(self._dataseries_selected) > 0:
                for label in self._dataseries_selected:
                    if not np.all([label in scan.y_labels for scan in scans]):
                        del self._dataseries_selected[
                            self._dataseries_selected.index(label)
                        ]

    @graph_scans.deleter
    def graph_scans(self):
        self._graph_scans = None

    @property
    def background_fixed_scans(self) -> list[scan_abstract]:
        return self._background_fixed_scans

    @background_fixed_scans.setter
    def background_fixed_scans(self, scans: list[scan_abstract] | None):
        self._background_fixed_scans = scans
        self._background_scan_combobox.clear()
        if isinstance(scans, list) and len(scans) > 0:
            self._background_scan_combobox.addItems([scan.filename for scan in scans])
        self._background_scan_combobox

    @background_fixed_scans.deleter
    def background_fixed_scans(self):
        self._background_fixed_scans = None

    @property
    def dataseries_selected(self) -> list[str]:
        return self._dataseries_selected

    @dataseries_selected.setter
    def dataseries_selected(self, labels: list[str]):
        self._dataseries_selected = labels
        # Check existing scan objects have the provided labels.
        if self.graph_scans is not None:
            for scan in self._graph_scans:
                containsLabel = [label in scan.y_labels for label in labels]
                if not np.all(containsLabel):
                    raise ValueError(
                        f"Provided labels {np.array(labels)[containsLabel]} do not exist in the scan objects."
                    )

    def generate_figure(self):
        # Requirements to be able to generate a figure:
        if (
            isinstance(self._graph_scans, list)
            and len(self._graph_scans) > 0
            and len(self._dataseries_selected) > 0
        ):
            if self._background_method == backgroundMethod.FIXED_SCAN:
                pass
        return

    @property
    def settings(self) -> normaliserSettings | None:
        # Check which settings are set to None
        nullBM = self.background_method == backgroundMethod.NONE
        nullPRE = (
            self.pre_edge_normalisation_method
            == scan_normalised_edges.PREEDGE_NORM_TYPE.NONE
        )
        nullPOST = (
            self.post_edge_normalisation_method
            == scan_normalised_edges.POSTEDGE_NORM_TYPE.NONE
        )
        # If all none, return None object.
        if nullBM and nullPRE and nullPOST:
            return None
        else:
            # Construct settings object
            settings = normaliserSettings(
                background_method=self.background_method,
                pre_edge_norm_method=self.pre_edge_normalisation_method,
                post_edge_norm_method=self.post_edge_normalisation_method,
                background_scan=self.background_scan,
                background_filter=self.background_filter,
                pre_edge_domain=self.pre_edge_domain,
                post_edge_domain=self.post_edge_domain,
                pre_edge_level=self.pre_edge_level,
                post_edge_level=self.post_edge_level,
            )
            return settings

    @settings.setter
    def settings(self, settings: normaliserSettings):
        self.background_method = settings.background_method
        self.pre_edge_normalisation_method = settings.pre_edge_norm_method
        self.post_edge_normalisation_method = settings.post_edge_norm_method
        self.background_scan = settings.background_scan
        self.background_filter_str = settings.background_filter_str
        self.pre_edge_domain = settings.pre_edge_domain
        self.post_edge_domain = settings.post_edge_domain
        self.pre_edge_level = settings.pre_edge_level
        self.post_edge_level = settings.post_edge_level


class scanNormaliserDialog(QtWidgets.QDialog):
    def __init__(
        self,
        graph_scans: list[scan_abstract],
        dataseries_selection: list[str],
        background_fixed_scans: list[scan_abstract] = [],
        normalisation_settings: normaliserSettings = None,
        parent=None,
    ):
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)

        # Add normaliser widget
        self._normaliser = scanNormaliser(
            graph_scans=graph_scans,
            dataseries_selection=dataseries_selection,
            background_fixed_scans=background_fixed_scans,
            normalisation_settings=normalisation_settings,
        )
        self._layout.addWidget(self._normaliser)

        # Add buttons
        QBtn = (
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self._buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self._buttonBox.accepted.connect(self.accept)
        self._buttonBox.rejected.connect(self.reject)
        self._layout.addWidget(self._buttonBox)

    def validate_settings(self) -> tuple[bool, str]:
        if self._normaliser.background_method == backgroundMethod.FIXED_SCAN:
            if self._normaliser.background_scan is None:
                return (False, "Background scan not selected.")
        # TODO implement futher validation checks.
        return (True, "")

    def accept(self):
        # Check if all settings are valid
        valid, message = self.validate_settings()
        if not valid:
            dlg = QtWidgets.QMessageBox(self)
            dlg.setWindowTitle("Normalisation Error")
            dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            dlg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dlg.setText(message)
            dlg.exec()
        else:
            super().accept()

    def reject(self):
        super().reject()

    @property
    def norm_settings(self) -> normaliserSettings:
        return self._normaliser.settings

    @norm_settings.setter
    def norm_settings(self, settings: normaliserSettings):
        self._normaliser.settings = settings

    @property
    def figure(self) -> mpl_figure.Figure:
        return self._normaliser._figure


class NavTBQT_Norm(NEXAFS_NavQT):
    normalisationUpdate = pyqtBoundSignal(bool)

    NavTB_divider = (None, None, None, None)
    NavTB_normalisation_option = (
        "Normalisation",
        "Allows use of normalisation and background subtraction options",
        (ICONS["normalisation_light"], ICONS["normalisation_dark"]),
        "norm_toolkit",
    )

    # Update toolitems to include normalisation option.
    toolitems = [*NEXAFS_NavQT.toolitems]
    toolitems.append(NavTB_divider)
    toolitems.append(NavTB_normalisation_option)

    def __init__(
        self,
        canvas,
        parent=None,
        coordinates=True,
        graph_scans: list[scan_abstract] = [],
        dataseries_selection: list[str] = [],
        background_fixed_scans: list[scan_abstract] = [],
        norm_settings: normaliserSettings = None,
    ):
        super().__init__(canvas, parent, coordinates)

        self._graph_scans = graph_scans
        self._dataseries_selection = dataseries_selection
        self._background_fixed_scans = background_fixed_scans
        self._norm_settings = norm_settings

    @property
    def graph_scans(self) -> list[scan_abstract]:
        return self._graph_scans

    @graph_scans.setter
    def graph_scans(self, scans: list[scan_abstract]):
        self._graph_scans = scans

    @property
    def dataseries_selection(self) -> list[str]:
        return self._dataseries_selection

    @dataseries_selection.setter
    def dataseries_selection(self, labels: list[str]):
        self._dataseries_selection = labels

    @property
    def background_fixed_scans(self) -> list[scan_abstract]:
        return self._background_fixed_scans

    @background_fixed_scans.setter
    def background_fixed_scans(self, scans: list[scan_abstract]):
        self._background_fixed_scans = scans

    @property
    def norm_settings(self) -> normaliserSettings:
        return self._norm_settings

    @norm_settings.setter
    def norm_settings(self, settings: normaliserSettings):
        self._norm_settings = settings

    @property
    def norm_figure(self) -> normaliserSettings:
        return self._norm_settings

    @norm_settings.setter
    def norm_figure(self, figure: mpl_figure.Figure):
        self._norm_figure = figure

    def norm_toolkit(self) -> None:
        normalisation_callback = scanNormaliserDialog(
            graph_scans=self._graph_scans,
            dataseries_selection=self._dataseries_selection,
            background_fixed_scans=self._background_fixed_scans,
            normalisation_settings=self._norm_settings,
        )
        if normalisation_callback.exec():
            # Store the normalisation options selected.
            self.norm_settings = normalisation_callback.norm_settings
            self.norm_figure = normalisation_callback.figure
            # Use signal to allow original UI to use properties.
            self.normalisationUpdate.emit(True)
        else:
            # Do nothing.
            pass


if __name__ == "__main__":

    app = QtWidgets.QApplication([])
    # window = scanNormaliser()
    window = scanNormaliserDialog([], [])
    window.setWindowTitle("Background subtraction & normalisation")
    window.show()

    if window.exec():
        print("OK")
        print(hasattr(window, "norm_settings"))
        print(window.norm_settings)
    else:
        print("Cancel")
