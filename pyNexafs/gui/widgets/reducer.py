from PyQt6 import QtGui, QtWidgets, QtCore
import numpy as np
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib as mpl
from matplotlib.backend_bases import MouseButton

# from pyNexafs.gui.widgets.graphing.matplotlib.widgets import NSpanSelector
from matplotlib.widgets import SpanSelector
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import NavTBQT
import numpy.typing as npt
from pyNexafs.utils.mda import MDAFileReader
import os
from typing import Iterator
from pyNexafs.utils.reduction import reducer


class EnergyBinReducer(QtWidgets.QWidget):
    """
    Reduces detector data with multiple energy bins (and channels) to a sum of counts.

    Parameters
    ----------
    energies : npt.NDArray
        A 1D array of energy values for the scanning beam energy.
    dataset : npt.NDArray
        The NEXAFS dataset with dimensions (beam_energy, detection_energy_bin, detector_index).
        The detector_index dimension is optional, used for multiple multi-channel analysers.
    bin_energies : npt.NDArray, optional
        A 1D array of energy values for each detection energy bin. If not provided, the bin numbers are used.
        If 2D, the second dimension must match the number of detectors.
    subsampling : int, optional
        The number of energy bins to skip when plotting the detector data.
        By default, every 5th bin is plotted.
    """

    def __init__(
        self,
        energies: npt.NDArray,
        dataset: npt.NDArray,
        bin_energies: npt.NDArray | None = None,
        subsampling: int = 5,
        parent=None,
    ):
        # Initialise the reduction class
        self._reduction: reducer = reducer(
            energies=energies, dataset=dataset, bin_energies=bin_energies
        )
        self._subsampling: int = subsampling

        # Initialize the parent class and layout
        super().__init__(parent)
        self._layout = QtWidgets.QVBoxLayout()
        self.setLayout(self._layout)
        if parent is not None:
            self.setContentsMargins(0, 0, 0, 0)
            self._layout.setContentsMargins(0, 0, 0, 0)

        # Generate a figure
        # For each detector, add a plot and colorbar
        root = int(np.ceil(np.sqrt(dataset.shape[2])))
        r, c = root, 2 * root
        subplots = plt.subplots(
            nrows=r,
            ncols=c,
            width_ratios=[1, 0.05] * r,
            layout="compressed",
        )
        self.detector_fig: plt.Figure = subplots[0]
        self.detector_axes: list[plt.Axes] = subplots[1]
        self.detector_canvas = FigureCanvas(self.detector_fig)
        self.detector_navtoolbar = NavTBQT(self.detector_canvas)
        self._layout.addWidget(self.detector_navtoolbar)
        self._layout.addWidget(self.detector_canvas)
        if len(self.detector_axes.shape) > 1:
            self.detector_axes = self.detector_axes.flatten()

        # Generate a second figure for the energy bin selection
        self._bin_layout = QtWidgets.QHBoxLayout()
        self._bin_edit_layout = QtWidgets.QVBoxLayout()
        self._bin_layout.addLayout(self._bin_edit_layout)
        self._bin_edit_layout.addWidget(QtWidgets.QLabel("Energy Binning:"))
        binning_label = (
            " #:" if not self._reduction.has_bin_energies else " Energy (eV):"
        )
        self._bin_edit_layout.addWidget(QtWidgets.QLabel("Lower" + binning_label))
        self.bin_lower = QtWidgets.QLineEdit()
        self._bin_edit_layout.addWidget(self.bin_lower)
        self._bin_edit_layout.addWidget(QtWidgets.QLabel("Upper" + binning_label))
        self.bin_upper = QtWidgets.QLineEdit()
        self._bin_edit_layout.addWidget(self.bin_upper)
        self.bin_lower.setValidator(
            QtGui.QDoubleValidator()
            if self._reduction.has_bin_energies
            else QtGui.QIntValidator()
        )
        self.bin_upper.setValidator(
            QtGui.QDoubleValidator()
            if self._reduction.has_bin_energies
            else QtGui.QIntValidator()
        )
        self._bin_edit_layout.addStretch(1)
        self._bin_edit_layout.addWidget(
            QtWidgets.QLabel(
                "Data shown has been modified\n to highlight features:\n"
                + "  1. Subtracted mean of each detector to\n    remove background noise levels.\n"
                + "  2. Translated from any negative values\n    to positive definite.\n"
                + "  3. Added 1e-2 to all values to allow \n    log plotting."
            )
        )

        self.bin_fig: plt.Figure
        self.bin_axes: plt.Axes
        self.bin_fig, self.bin_axes = plt.subplots(nrows=1, ncols=1)
        self.bin_canvas = FigureCanvas(self.bin_fig)
        self.bin_navtoolbar = NavTBQT(self.bin_canvas)
        self._bin_edit_layout.insertWidget(0, self.bin_navtoolbar)
        self._bin_layout.addWidget(self.bin_canvas)
        self._layout.addLayout(self._bin_layout)

        # Setup the energy bin selection
        self.plot_bin_features()

        # Setup the span selector
        self.span: SpanSelector = SpanSelector(
            # N=1,
            ax=self.bin_axes,
            onselect=self.onselect,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.2, facecolor="red"),
            interactive=True,
            drag_from_anywhere=True,
            button=MouseButton.LEFT,
        )
        # plot the full range
        self.plot()

        # Connect the bounds to the span selector
        self.bin_lower.editingFinished.connect(self.on_bounds_update)
        self.bin_upper.editingFinished.connect(self.on_bounds_update)

    def plot_bin_features(self) -> None:
        """
        Plots the bin features for each detector.

        The bin features are the sum of counts for each detector channel.
        """
        self.bin_axes.clear()
        bins, counts = self._reduction.reduce_to_bin_features()
        for i in range(self._reduction.detectors):
            self.bin_axes.plot(bins, counts[:, i], label=f"Detector {i}")

        self.bin_axes.set_xlabel(
            "Bin #" if not self._reduction.has_bin_energies else "Bin Energy (eV)"
        )
        self.bin_axes.set_ylabel("Log Channel Counts (A.U.)")
        self.bin_axes.set_yscale("log")
        self.bin_axes.legend()
        self.bin_fig.tight_layout()

    def on_bounds_update(self):
        """
        Callback function for when the bounds are updated.

        Modifies span selection to match the new bounds,
        which will also trigger onselect.
        """
        lb, ub = self.bin_lower.text(), self.bin_upper.text()
        if self._reduction.has_bin_energies:
            lb, ub = float(lb), float(ub)
        else:
            lb, ub = int(lb), int(ub)
        self.span.extents = (lb, ub)

    @property
    def dataset(self) -> npt.NDArray:
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: npt.NDArray):
        self._dataset = dataset

        self.plot()

    @staticmethod
    def _validify_inputs(
        energies: npt.NDArray,
        dataset: npt.NDArray,
        bin_energies: npt.NDArray | None = None,
    ) -> bool:
        """
        Validates the input dataset, energies and bin_energies.

        Raises a ValueError if the inputs are invalid, otherwise returns True.

        Parameters
        ----------
        dataset : npt.NDArray
            The NEXAFS dataset with dimensions (beam_energy, detection_energy_bin, detector_index).
            The detector_index dimension is optional, used for multiple multi-channel analysers.
        energies : npt.NDArray
            A 1D array of energy values for the scanning beam energy.
        bin_energies : npt.NDArray, optional
            A 1D array of energy values for each detection energy bin. If not provided, the bin numbers are used.
            If 2D, the second dimension must match the number of detectors.

        Returns
        -------
        tuple[npt.NDArray, npt.NDArray, npt.NDArray]
            The validated dataset, energies and bin_energies.
        """
        if not energies.ndim == 1:
            raise ValueError("`energies` must be a 1D numpy array.")
        if not (dataset.ndim == 2 or dataset.ndim == 3):
            raise ValueError(
                "`dataset` must be a 3D or 2D numpy array with indexes corresponding to"
                + " (beam_energy, detection_energy_bin, detector_index). The detector_index dimension is optional."
            )
        if not (energies.shape[0] == dataset.shape[0]):
            raise ValueError(
                f"`energies` must have the same length ({energies.shape[0]})"
                + f" as the first dimension of `dataset` ({dataset.shape[0]})."
            )
        if dataset.ndim == 2:
            dataset = dataset[:, :, np.newaxis]  # add final dimension.

        # Check bin_energies validity
        if bin_energies is not None:
            bin_energies = np.asarray(bin_energies)
            if not (bin_energies.ndim == 1 or bin_energies.ndim == 2):
                raise ValueError(
                    "`bin_energies` must be a 1D or 2D numpy array, with indexes corresponding to"
                    + " (detection_bin_energy, detector_index). The second dimension is optional."
                )
            if not (bin_energies.shape[0] == dataset.shape[1]):
                raise ValueError(
                    f"`bin_energies` must have the same length ({bin_energies.shape[0]}) "
                    + f"as the second dimension of `dataset` ({dataset.shape[1]})."
                )
        return True

    @property
    def domain(self) -> tuple[float, float] | tuple[int, int]:
        """
        The selection domain.

        Energies (`float`) are returned if `bin_energies` have been provided,
        otherwise bin indexes (`int`) are used.

        Returns
        -------
        lower : float | int
            The lower bound of the selected domain.
        upper : float | int
            The upper bound of the selected domain.
        """
        vals = (
            self.span.extents
            if self._reduction.has_bin_energies
            else tuple(np.round(self.span.extents).astype(int))
        )
        if vals[0] == vals[1]:
            # If the same value is provided, return the full range from the data.
            return tuple(self._reduction.bin_energies[[0, -1]])
        else:
            return vals

    @property
    def domain_incidies(self) -> list[tuple[int, int]]:
        """
        A list of the bin indices for each detector.

        Returns
        -------
        list[tuple[int, int]]
            List of the lower and upper selected bin indices for each detector.
        """
        return self._reduction.domain_to_indexes(self.domain)

    def plot(self, bin_domain: tuple[int, int] | None = None):
        """
        Plot the detector data for the given energy domain.

        Parameters
        ----------
        bin_domain : tuple[int, int], optional
            The energy domain to plot, by default None.
            If upper and lower bounds are not provided / identical, the full range is plotted.
        """
        if bin_domain is None or bin_domain[0] == bin_domain[1]:
            # Use full index range if no domain is provided or the same value is provided.
            bin_idxs = [
                (0, self._dataset.shape[1]) for i in range(self._dataset.shape[2])
            ]
        elif not self._reduction.has_bin_energies:
            # No energies values provided, so can't convert the domain to indexes.
            # Domain already in bin numbers. Round to nearest integer and use for all detectors.
            lb, ub = np.round(bin_domain).astype(int)
            bin_idxs = [(lb, ub)] * self._dataset.shape[2]
        else:
            bin_idxs = [
                self._bin_domain_to_indx(bin_domain, detector=i)
                for i in range(self._dataset.shape[2])
            ]

        cmap = plt.get_cmap("inferno")
        # Clear all existing axes
        for ax in self.detector_axes:
            ax.clear()
        # Plot the new data at the selected domain
        for i, ax in enumerate(self.detector_axes[::2]):  # for each detector
            bins = bin_idxs[i]
            # Reduce the data to subsampled values
            bl: int = int(bins[0])
            """Lower bin index"""
            bu: int = int(bins[1])
            """Upper bin index"""
            sampled_bin_energies = self._bin_energies[bl : bu : self._subsampling, i]
            sampled_data = self._dataset[:, bl : bu : self._subsampling, i]

            # Get the sub-sampled data and setup the color normalisation.
            norm = mpl_colors.Normalize(
                vmin=sampled_bin_energies.min(), vmax=sampled_bin_energies.max()
            )

            # Plot the data
            for j in range(sampled_data.shape[1]):  # for each subsampled bin
                ax.plot(
                    self._energies,
                    sampled_data[:, j],
                    c=cmap(norm(sampled_bin_energies[j])),
                )

            # Add the colorbar
            mpl.colorbar.ColorbarBase(
                self.detector_axes[2 * i + 1],
                cmap=cmap,
                norm=norm,
                label=(
                    "Bin Energy (eV)" if self._reduction.has_bin_energies else "Bin #"
                ),
            )

            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Counts")
            ax.set_title("Detector {}".format(i))
        # Update the canvas
        self.detector_canvas.draw()

    def onselect(self, xmin: float, xmax: float):
        """
        Callback function for the SpanSelector.

        Used to call `plot` with the selected energy domain.

        Parameters
        ----------
        xmin : float
            The lower bound of the selected energy domain.
        xmax : float
            The upper bound of the selected energy domain.
        """
        self.plot(bin_domain=(xmin, xmax))
        # Pause QEdit updates while changing values from the span selector
        self.bin_lower.blockSignals(True)
        self.bin_upper.blockSignals(True)
        if self._reduction.has_bin_energies:
            self.bin_lower.setText(str(xmin))
            self.bin_upper.setText(str(xmax))
        else:
            self.bin_lower.setText(str(int(round(xmin))))
            self.bin_upper.setText(str(int(round(xmax))))
        # Resume QEdit updates
        self.bin_lower.blockSignals(False)
        self.bin_upper.blockSignals(False)


class EnergyBinReducerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        energies: npt.NDArray,
        dataset: npt.NDArray,
        bin_energies: npt.NDArray | None = None,
        subsampling: int = 5,
        parent=None,
    ):
        super().__init__(parent)

        self.reducer = EnergyBinReducer(energies, dataset, bin_energies, subsampling)
        self._layout = QtWidgets.QVBoxLayout()
        self._layout.addWidget(self.reducer)
        self.setLayout(self._layout)
        self.setWindowTitle("Energy Bin Reducer")

        # self.dataset = dataset
        # self.layout = QtWidgets.QVBoxLayout()
        # self.setLayout(self.layout)
        # self.energy_bin_reducer = EnergyBinReducer(dataset=dataset)
        # self.layout.addWidget(self.energy_bin_reducer)

        # Add buttons
        QBtn = (
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Ok
        )
        self._buttonBox = QtWidgets.QDialogButtonBox(QBtn)
        self._buttonBox.accepted.connect(self.accept)
        self._buttonBox.rejected.connect(self.reject)
        self._layout.addWidget(self._buttonBox)

    @property
    def domain_incidies(self) -> list[tuple[int, int]]:
        """
        Returns the selected energy domain indices for each detector.

        Returns
        -------
        list[tuple[int, int]]
            List of the lower and upper selected bin indices for each detector.
        """
        return self.reducer.domain_incidies

    @property
    def domain(self) -> tuple[float, float] | tuple[int, int]:
        """
        Returns the selected energy domain.

        Returns
        -------
        tuple[float, float] | tuple[int, int]
            The lower and upper selected energies/indexes.
            If energies, a tuple of floats is returned, otherwise a tuple of integers for indices.
        """
        return self.reducer.domain

    def accept(self):
        # Check if all settings are valid
        # valid, message = self.validate_settings()
        valid = True  # No validation required for simple selector..
        if not valid:
            dlg = QtWidgets.QMessageBox(self)
            dlg.setWindowTitle("Normalisation Error")
            dlg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            dlg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            dlg.setText("Error in normalisation settings.")
            dlg.exec()
        else:
            super().accept()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.reject()
        elif (
            event.key() == QtCore.Qt.Key.Key_Return
            or event.key() == QtCore.Qt.Key.Key_Enter
        ):
            if self.reducer.bin_lower.hasFocus() or self.reducer.bin_upper.hasFocus():
                # Ignore the event when editing.
                return
            else:
                return event.accept()
        else:
            super().keyPressEvent(event)

    def reject(self):
        super().reject()


if __name__ == "__main__":
    path = os.path.dirname(__file__)
    package_path = os.path.normpath(os.path.join(path, "../../../"))
    mda_path = os.path.normpath(
        os.path.join(package_path, "tests/test_data/au/MEX2/MEX2_5643.mda")
    )
    mda_data, mda_scans = MDAFileReader(mda_path).read_scans()
    data1D, data2D = mda_data

    # Convert MEX2 Data
    energies = data1D[:, 0] * 1000  # Convert keV to eV
    span = [96, 500]
    binned_data = data2D[:, span[0] : span[1], :]
    TOTAL_BINS = span[1] - span[0]
    BIN_ENERGY = 11.935
    BIN_96_ENERGY = 1146.7
    bin_energies = np.linspace(
        BIN_96_ENERGY, BIN_96_ENERGY + TOTAL_BINS * BIN_ENERGY, TOTAL_BINS
    )

    app = QtWidgets.QApplication([])
    win = EnergyBinReducerDialog(
        energies=energies, dataset=binned_data, bin_energies=bin_energies
    )
    win.show()
    if win.exec():
        print("OK")
        print("Selected Domain:", win.domain)
        print("Selected Domain Indices for detectors:", win.domain_incidies)
    else:
        print("Cancel")
