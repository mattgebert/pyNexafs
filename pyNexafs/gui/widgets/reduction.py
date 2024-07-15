from PyQt6 import QtGui, QtWidgets, QtCore
import numpy as np
from pyNexafs.gui.widgets.graphing.matplotlib.graphs import FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib as mpl
from matplotlib.backend_bases import MouseButton
from pyNexafs.gui.widgets.graphing.matplotlib.widgets import NSpanSelector
import numpy.typing as npt
from pyNexafs.utils.mda import MDAFileReader
import os


class EnergyBinReducer(QtWidgets.QWidget):
    """
    Reduces detector energy bins and channels to some sum of counts.

    Parameters
    ----------
    QtWidgets : _type_
        _description_
    """

    def __init__(
        self,
        energies: npt.NDArray,
        dataset: npt.NDArray,
        bin_energies: npt.NDArray | None = None,
        subsampling: int = 3,
        parent=None,
    ):
        energies = np.asarray(energies)
        dataset = np.asarray(dataset)
        # Check dataset validity first
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

        self._bin_labels: bool = False
        """Whether the bin energies are bin numbers (False) or actual energies (True)."""

        if bin_energies is not None:
            self._bin_labels = True
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
            # Require bin energies to match the number of detectors
            if bin_energies.ndim == 1:
                if dataset.shape[2] > 1:
                    bin_energies = (
                        bin_energies[:, np.newaxis]
                        * np.ones(dataset.shape[2])[np.newaxis, :]
                    )
        else:
            # Create bin numbers:
            self._bin_labels = False
            bin_energies = np.linspace(
                0, self._dataset.shape[1], self._dataset.shape[1]
            )

        self._energies: np.ndarray = energies
        self._dataset: np.ndarray = dataset
        self._bin_energies: np.ndarray = bin_energies
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
        # self.fig = fig = plt.figure(figsize=(10, 10))
        subplots = plt.subplots(
            nrows=r,
            ncols=c,
            width_ratios=np.reshape([[1, 0.05] for i in range(r)], -1),  # flatten
            layout="compressed",
        )
        self.detector_fig: plt.Figure = subplots[0]
        self.detector_axes: list[plt.Axes] = subplots[1]
        self.detector_canvas = FigureCanvas(self.detector_fig)
        self._layout.addWidget(self.detector_canvas)
        if len(self.detector_axes.shape) > 1:
            self.detector_axes = self.detector_axes.flatten()

        # Generate a second figure for the energy bin selection
        self.bin_fig: plt.Figure
        self.bin_axes: plt.Axes
        self.bin_fig, self.bin_axes = plt.subplots(nrows=1, ncols=1)
        self.bin_canvas = FigureCanvas(self.bin_fig)
        self._layout.addWidget(self.bin_canvas)
        # Setup the energy bin selection
        for i in range(dataset.shape[2]):
            # Subtract average bin acquisition level to highlight data features
            # Axis are (beam_energy, detection_energy_bin, detector_index)
            # For each detector, subtract the mean of the energy bins from each bin
            ds_sub: np.ndarray = (
                dataset[:, :, i] - np.mean(dataset[:, :, i], axis=1)[:, np.newaxis]
            )
            # Sum over all energies to find the energy bins with signal.
            ds_sum = ds_sub.sum(axis=0)
            self.bin_axes.plot(self._bin_energies[:, i], ds_sum, label=f"Detector {i}")

        # self.bin_axes.set_xlabel("Energy (eV)")
        self.bin_axes.set_xlabel("Bin #" if not self._bin_labels else "Bin Energy (eV)")
        self.bin_axes.set_ylabel(
            "Channel Counts (A.U.)\n(Sum of mean deviations over all beam energies)"
        )
        self.bin_axes.legend()

        # Setup the span selector
        self.span = NSpanSelector(
            N=1,
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

    def plot(self, bin_domain: tuple[int, int] | None = None):
        # Setup the full range indexes if not provided.
        if not self._bin_labels:
            # If no labels, the range matches the indexes
            bin_idxs = [bin_domain for i in range(self._dataset.shape[2])]
        else:
            # If labels and a given domain, convert the range to indexes
            if bin_domain is not None:
                bin_idxs = []
                for i in range(self._dataset.shape[2]):
                    lb = np.argmin(np.abs(self._bin_energies[:, i] - bin_domain[0]))
                    ub = np.argmin(np.abs(self._bin_energies[:, i] - bin_domain[1]))
                    bin_idxs.append((lb, ub))
            else:
                # use the default full range
                bin_idxs = [
                    (0, self._dataset.shape[1]) for i in range(self._dataset.shape[2])
                ]

        cmap = plt.get_cmap(
            "inferno"
        )  # Use inferno, due to different temperature range.
        for i, ax in enumerate(self.detector_axes[::2]):  # for each detector
            # Clear the existing axis
            ax.clear()
            self.detector_axes[2 * i + 1].clear()
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
                label="Bin Energy (eV)" if self._bin_labels else "Bin #",
            )

            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel("Counts")
            ax.set_title("Detector {}".format(i))
        # Update the canvas
        self.detector_canvas.draw()

    def onselect(self, xmin, xmax):
        self.plot(bin_domain=(xmin, xmax))


class EnergyBinReducerDialog(QtWidgets.QDialog):
    def __init__(self, dataset: npt.NDArray, parent=None):
        super().__init__(parent)
        # self.dataset = dataset
        # self.layout = QtWidgets.QVBoxLayout()
        # self.setLayout(self.layout)
        # self.energy_bin_reducer = EnergyBinReducer(dataset=dataset)
        # self.layout.addWidget(self.energy_bin_reducer)


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
    win = EnergyBinReducer(
        energies=energies, dataset=binned_data, bin_energies=bin_energies
    )
    win.show()
    app.exec()
