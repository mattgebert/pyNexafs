"""
Parser classes for the Medium Energy X-ray 2 (MEX1) beamline at the Australian Synchrotron.
"""

import PyQt6
from PyQt6 import QtWidgets
from pyNexafs.parsers import parser_base, parser_meta
from pyNexafs.nexafs.scan import scan_base
from pyNexafs.utils.mda import MDAFileReader
from pyNexafs.gui.widgets.reducer import EnergyBinReducerDialog
from io import TextIOWrapper
from typing import Any, Self
from numpy.typing import NDArray
import numpy as np
import ast
import warnings
import datetime as dt
import os
from pyNexafs.utils.reduction import reducer
import traceback

# TODO: What are the values for MEX1? What are the bins?
# Additional data provided by the MEX1 beamline for the data reduction
# BIN_ENERGY_DELTA = 11.935
# BIN_96_ENERGY = 1146.7
TOTAL_BINS = 4096
# TOTAL_BIN_ENERGIES = np.linspace(
#     start=BIN_96_ENERGY - 95 * BIN_ENERGY_DELTA,
#     stop=BIN_96_ENERGY + (TOTAL_BINS - 96) * BIN_ENERGY_DELTA,
#     num=TOTAL_BINS,
# )
INTERESTING_BINS_IDX = [80, 2000]  # 80 to 900 for MEX2.
# INTERESTING_BINS_ENERGIES = TOTAL_BIN_ENERGIES[
# INTERESTING_BINS_IDX[0] : INTERESTING_BINS_IDX[1]
# ]


class MEX1_NEXAFS_META(parser_meta):
    def __init__(
        cls: type,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwds: Any,
    ) -> "MEX1_NEXAFS":

        # Create an extra class property for MEX1 mda data, to track binning settings
        cls.reduction_bin_domain: list[tuple[int, int]] | None = None
        """Tracker for the binning settings used in the most recent data reduction."""

        # Perform the normal class creation via parser_meta.
        return super().__init__(name=name, bases=bases, namespace=namespace, **kwds)


class MEX1_NEXAFS(parser_base, metaclass=MEX1_NEXAFS_META):
    """
    Australian Synchrotron Soft X-ray (SXR) NEXAFS parser.

    Parses data formats including '.asc' and '.mda' formats from the SXR
    Near Edge X-ray Absorption Fine Structure (NEXAFS) tool.

    Attributes
    ----------
    ALLOWED_EXTENSIONS
    SUMMARY_PARAM_RAW_NAMES
    COLUMN_ASSIGNMENTS
    RELABELS

    Parameters
    ----------
    filepath : str | None
        The file path to the data file.
    header_only : bool, optional
        If True, only the header of the file is read, by default False
    relabel : bool | None, optional
        If True, then the parser will relabel the data columns, by default None
    use_recent_binning : bool, optional
        If True, then the '.mda' parsers will use the most recent class
        reduction binning settings. By default True, as UIs will use this,
        and assume the parser will use the same binning settings.
        TODO: Perhaps make this a property of the base class, that way UIs can
        reset such settings if needed upon directory change etc.

    Notes
    -----
    Implemented for data as of 2024-Mar.
    """

    ALLOWED_EXTENSIONS = [".xdi", ".mda"]
    SUMMARY_PARAM_RAW_NAMES = [
        "Sample|Comment 1",
        "Angle" "Element.symbol",
        "Element.edge",
        "E1",
        "E2",
        "E3",
        "E4",
    ]
    COLUMN_ASSIGNMENTS = {
        "x": "Energy|Energy Setpoint|energy",
        "y": [
            "Bragg|bragg",
            "Fluorescence|iflour|Fluorescence Sum",
            "Count Time|count_time",
            "I0|i0",
            "Sample Drain|SampleDrain",
            "ICR_AVG",
            "OCR_AVG",
        ],
        "y_errs": None,
        "x_errs": None,
    }

    RELABELS = {
        # -------------------MDA File-------------------
        # 'MEX1SSCAN01:saveData_comment1':,
        # 'MEX1SSCAN01:saveData_comment2':,
        # 'MEX1SSCAN01:saveData_realTime1D':,
        # 'MEX1SSCAN01:saveData_fileSystem':,
        # 'MEX1SSCAN01:saveData_subDir':,
        # 'MEX1SSCAN01:saveData_fileName':,
        # 'MEX1SSCAN01:scan1.P1SM':,
        # 'MEX1SSCAN01:scan1.P2SM':,
        # 'MEX1SSCAN01:scan1.P3SM':,
        # 'MEX1SSCAN01:scan1.P4SM':,
        # 'MEX1SSCAN01:scanTypeSpec':,
        # 'MEX1SSCAN01:scan1.BSPV':,
        # 'MEX1SSCAN01:scan1.BSCD':,
        # 'MEX1SSCAN01:scan1.BSWAIT':,
        # 'MEX1SSCAN01:scan1.ASPV':,
        # 'MEX1SSCAN01:scan1.ASCD':,
        # 'MEX1SSCAN01:scan1.ASWAIT':,
        # 'MEX1SSCAN01:scan1.PDLY':,
        # 'MEX1SSCAN01:scan1.DDLY':,
        # 'TS01:SECONDS_MONITOR':,
        # 'MEX1ES01GLU01:MEX_TIME':,
        # 'MEX1SLT01:VSIZE.RBV':,
        # 'MEX1SLT01:VCENTRE.RBV':,
        # 'MEX1SLT01:HSIZE.RBV':,
        # 'MEX1SLT01:HCENTRE.RBV':,
        # 'MEX1SLT01:VSIZE.OFF':,
        # 'MEX1SLT01:VCENTRE.OFF':,
        # 'MEX1SLT01:HSIZE.OFF':,
        # 'MEX1SLT01:HCENTRE.OFF':,
        # 'MEX1SLT01MOT01.RBV':,
        # 'MEX1SLT01MOT03.RBV':,
        # 'MEX1SLT01MOT02.RBV':,
        # 'MEX1SLT01MOT04.RBV':,
        # 'MEX1SCRN01LGHT01:BRIGHTNESS_MONITOR':,
        # 'MEX1SCRN01ACTP01:INSERT_WITHDRAW_STATUS':,
        # 'MEX1MIR01:POSITIONER:select.RVAL':,
        # 'MEX1MIR01:TRANS.RBV':,
        # 'MEX1MIR01:YAW.RBV':,
        # 'MEX1MIR01MOT06.RBV':,
        # 'MEX1MIR01:HEIGHT.RBV':,
        # 'MEX1MIR01:ROLL.RBV':,
        # 'MEX1MIR01:PITCH.RBV':,
        # 'MEX1MIR01MOT01.RBV':,
        # 'MEX1MIR01MOT02.RBV':,
        # 'MEX1MIR01MOT03.RBV':,
        # 'MEX1MIR01MOT04.RBV':,
        # 'MEX1MIR01MOT05.RBV':,
        # 'MEX1DCM01:ENERGY_RBV':,
        # 'MEX1DCM01:ENERGY_EV_RBV':,
        # 'MEX1DCM01:OFFSET_RBV':,
        # 'MEX1DCM01:XTAL_INBEAM.RVAL':,
        # 'MEX1DCM01:FINE_PITCH_MRAD_RBV':,
        # 'MEX1DCM01:FINE_ROLL_MRAD_RBV':,
        # 'MEX1DCM01MOT01.RBV':,
        # 'MEX1DCM01MOT02.RBV':,
        # 'MEX1DCM01MOT05.RBV':,
        # 'MEX1DCM01MOT03.RBV':,
        # 'MEX1DCM01MOT04.RBV':,
        # 'MEX1DCM01MOT01.OFF':,
        # 'MEX1DCM01MOT02.OFF':,
        # 'MEX1DCM01:y2_track':,
        # 'MEX1DCM01:y2_mvmin':,
        # 'MEX1DCM01:th_mvmin':,
        # 'MEX1DCM01:Dspace':,
        # 'MEX1DCM01:Mono1110DSpace':,
        # 'MEX1DCM01:Mono1110ThetaOffset':,
        # 'MEX1DCM01:Mono1110HeightOffset':,
        # 'MEX1DCM01:Mono1110Pitch':,
        # 'MEX1DCM01:Mono1110Roll':,
        # 'MEX1DCM01:Mono1110Centre':,
        # 'MEX1DCM01:Mono11130DSpace':,
        # 'MEX1DCM01:Mono11130ThetaOffset':,
        # 'MEX1DCM01:Mono11130HeightOffset':,
        # 'MEX1DCM01:Mono11130Pitch':,
        # 'MEX1DCM01:Mono11130Roll':,
        # 'MEX1DCM01:Mono11130Centre':,
        "MEX1ES01DET01:MCA1:ArrayData": "MCA Ch1",
        "MEX1ES01DET01:MCA2:ArrayData": "MCA Ch2",
        "MEX1ES01DET01:MCA3:ArrayData": "MCA Ch3",
        "MEX1ES01DET01:MCA4:ArrayData": "MCA Ch4",
        # -------------------XDI File-------------------
        # 'Facility.name':,
        # 'Beamline.name':,
        # 'Mono.d_spacing':,
        # 'Element.symbol':,
        # 'Element.edge':,
        # 'Processing.version':,
        # 'Reference.Element':,
        # 'ROI.start_bin':,
        # 'ROI.end_bin':,
        # 'File.Name':,
        "MEX1ES01ZEB01:CALC_ENERGY_EV": "energy",
        "MEX1ES01ZEB01:BRAGG_WITH_OFFSET": "bragg",
        "SR11BCM01:CURRENT_MONITOR": "ring_current",
        "MEX1ES01ZEB01:GATE_TIME_SET": "count_time",
        "MEX1ES01DAQ01:ch1:S:MeanValue_RBV": "i0",
        "MEX1ES01DAQ01:ch2:S:MeanValue_RBV": "i1",
        "MEX1ES01DAQ01:ch3:S:MeanValue_RBV": "i2",
        "MEX1ES01DET01:C1SCA:0:Value_RBV": "S1_clock_ticks",
        "MEX1ES01DET01:C2SCA:0:Value_RBV": "S2_clock_ticks",
        "MEX1ES01DET01:C3SCA:0:Value_RBV": "S3_clock_ticks",
        "MEX1ES01DET01:C4SCA:0:Value_RBV": "S4_clock_ticks",
        "MEX1ES01DET01:C1SCA:5:Value_RBV": "S1_window1",
        "MEX1ES01DET01:C2SCA:5:Value_RBV": "S2_window1",
        "MEX1ES01DET01:C3SCA:5:Value_RBV": "S3_window1",
        "MEX1ES01DET01:C4SCA:5:Value_RBV": "S4_window1",
        "MEX1ES01DET01:C1SCA:8:Value_RBV": "S1_DTFactor",
        "MEX1ES01DET01:C2SCA:8:Value_RBV": "S2_DTFactor",
        "MEX1ES01DET01:C3SCA:8:Value_RBV": "S3_DTFactor",
        "MEX1ES01DET01:C4SCA:8:Value_RBV": "S4_DTFactor",
        "S1_real_time - S1_clock_ticks / 80MHz (clock rate xspress3 mini)": "S1_real_time (clock_ticks / 80MHz)",
        "S2_real_time - S2_clock_ticks / 80MHz (clock rate xspress3 mini)": "S2_real_time",
        "S3_real_time - S3_clock_ticks / 80MHz (clock rate xspress3 mini)": "S3_real_time",
        "S4_real_time - S4_clock_ticks / 80MHz (clock rate xspress3 mini)": "S4_real_time",
        "S1_ifluor - S1_window1 * S1_DTFactor / S1_real_time": "S1_ifluor (window * dead_time / real_time)",
        "S2_ifluor - S2_window1 * S2_DTFactor / S2_real_time": "S2_ifluor",
        "S3_ifluor - S3_window1 * S3_DTFactor / S3_real_time": "S3_ifluor",
        "S4_ifluor - S4_window1 * S4_DTFactor / S4_real_time": "S4_ifluor",
        "avg_ifluor - (ifluor_S1 + ifluor_S2 + ifluor_S3 + ifluor_S4)/4 -- deadtime correction and time normalisation per sensor": "ifluor_avg_corr",
        "S1_mufluor - S1_ifluor / i0": "S1_mufluor (normalised by i0)",
        "S2_mufluor - S2_ifluor / i0": "S2_mufluor",
        "S3_mufluor - S3_ifluor / i0": "S3_mufluor",
        "S4_mufluor - S4_ifluor / i0": "S4_mufluor",
        "avg_mufluor - ifluor_avg_corr / i0 -- deadtime correction and time normalisation per sensor, normalised by i0": "mufluor_avg_corr",
    }

    def __init__(
        self,
        filepath: str | None,
        header_only: bool = False,
        relabel: bool | None = None,
        use_recent_binning: bool = True,
    ) -> None:
        # Manually add kwargs
        kwargs = {}
        if use_recent_binning is not None:
            kwargs.update(use_recent_binning=use_recent_binning)
        super().__init__(filepath, header_only, relabel, **kwargs)

    @classmethod
    def parse_xdi_2024_08(
        cls,
        file: TextIOWrapper,
        header_only: bool = False,
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """Reads Australian Synchrotron MEX1 .xdi files, as of 2024-Aug.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.xdi', 'r'))
        header_only : bool, optional
            If True, then only the header of the file is read and NDArray is returned as None, by default False

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array,
            labels as a list of strings,
            units as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .xdi file.
        """
        # Initialise structures
        params = {}
        labels = []
        units = None

        # Check valid format.
        if not file.name.endswith(".xdi"):
            raise ValueError(f"File {file.name} is not a valid .xdi file.")

        ### Read file
        # Check header is correct
        assert file.readline() == "# XDI/1.0\n"

        ## 1 Initial Parameters
        column_descriptions = {}
        column_type_assignments = {}
        # Read first param line.
        line = file.readline()
        i = 0
        while "# " == line[0:2] and line != "# ///\n":
            line = line[2:].strip().split(": ", 1)  # split first colon
            param, value = tuple(line)

            # Categorise information
            if param.startswith("Column."):
                # Label name
                labels.append(value)
                params[param] = value
            elif param.startswith("Describe.Column."):
                # Label description. No dedicated structure for this, add to params.
                col_name = params[param.replace("Describe.", "")]
                if col_name + " - " in value:
                    value = value.replace(col_name + " - ", "")
                if " -- " in value:
                    value, descr = value.split(" -- ", 1)
                else:
                    descr = None
                params[param] = value
                column_descriptions[i] = (
                    value,
                    col_name + " -- " + descr if descr is not None else col_name,
                    "",  # No units
                )
                column_type_assignments[i] = f"1-D Detector{i:4}"
            else:
                if param == "Samples":
                    # Parse the datapoint length list
                    samples = ast.literal_eval(value)
                    samples = [x.strip() if isinstance(x, str) else x for x in samples]
                    # Add to params
                    for i in range(len(samples)):
                        params[f"Datapoints.Column.{i}"] = samples[i]
                else:
                    # General parameter, add to params.
                    params[param] = value
            # Load new param line
            line = file.readline()
            i += 1

        assert line == "# ///\n"  # Check end of initial parameters

        # Get samplename
        line = file.readline()
        sample_name = line[2:].strip()
        params["Sample"] = sample_name

        # Skip header lines before data
        assert file.readline() == "# \n"
        line = file.readline()
        # Some xdi conversion have a "default mda2xdi" line.
        try:
            assert line == "# xdi from default mda2xdi preset for mex1.\n"
            line = file.readline()
        except AssertionError:
            pass
        # Some xdi files have error messages
        while (
            line.strip()
            == "# All values in column i1 replaced with 1 because consecutive negatives were detected."
        ):
            line = file.readline()
        # Always ends with a line of "#--------\n" before data.
        assert line == "#--------\n"

        # Read data columns
        header_line = file.readline()
        assert header_line[0:2] == "# "
        labels = (
            header_line[2:].strip().split()
        )  # split on whitespace, even though formatting seems to use "   " (i.e. three spaces).
        labels = [label.strip() if type(label) == str else label for label in labels]

        if header_only:
            # Do not process remaining lines
            return None, labels, units, params

        # Read data
        lines = file.readlines()  # read remaining lines efficiently

        # Convert data to numpy array.
        data = np.loadtxt(lines)
        data = np.array(data)

        return data, labels, units, params

    @classmethod
    def parse_mda_2024_08(
        cls,
        file: TextIOWrapper,
        header_only: bool = False,
        use_recent_binning: bool = False,
        energy_bin_range: tuple[float, float] | None = None,
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """
        Reads Australian Synchrotron .mda files for MEX1 Data

        Created for data as of 2024-Aug.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.mda', 'r'))
        header_only : bool, optional
            If True, then only the header of the file is read and
            NDArray is returned as None, by default False
        energy_bin_range : tuple[float, float] | None, optional
            The energy range to bin the data, by default None
        use_recent_binning : bool, optional
            If True, then the most recent binning settings are used
            Ignored if `energy_bin_range` is specified.
            for data reduction, by default False

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array,
            labels as a list of strings,
            units as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .mda file.
        """
        # Initialise parameter list
        params = {}
        labels = []
        units = []

        # Check valid format.
        if not file.name.endswith(".mda"):
            raise ValueError(f"File {file.name} is not a valid .mda file.")

        # Need to reopen the file in byte mode.
        file.close()
        mda = MDAFileReader(file.name)
        mda_header = mda.read_header_as_dict()
        ## Previously threw error for higher dimensional data, now just a warning.
        mda_params = mda.read_parameters()
        mda_arrays, mda_scans = mda.read_scans(header_only=header_only)

        # Add values to params dict
        params.update(mda_header)
        params.update(mda_params)

        # Initialise to None for header only reading.
        mda_1d = None
        # Add column types and descriptions to params.
        if not header_only:
            # 1D array essential
            mda_1d = mda_arrays[0]
            # 2D array optional
            if len(mda_arrays) > 1:
                mda_2d = mda_arrays[1]
                mda_2d_scan = mda_scans[1]
                # Check 'multi-channel-analyser-spectra of fluorescence-detector' names are as expected
                florescence_labels = [
                    "MEX1ES01DET01:MCA1:ArrayData",
                    "MEX1ES01DET01:MCA2:ArrayData",
                    "MEX1ES01DET01:MCA3:ArrayData",
                    "MEX1ES01DET01:MCA4:ArrayData",
                ]
                assert mda_2d_scan.labels() == florescence_labels

                # Take properties from 1D and 2D arrays:
                energies = mda_1d[:, 0] * 1000  # Convert keV to eV for MEX beamline
                dataset = mda_2d[
                    :, INTERESTING_BINS_IDX[0] : INTERESTING_BINS_IDX[1], :
                ]
                bin_e = None  # INTERESTING_BINS_ENERGIES  # pre-calibrated.

                ## Perform binning on 2D array:
                # Is an existing binning range available?
                if energy_bin_range is not None:
                    # Update the class variable with the new binning settings.
                    cls.reduction_bin_domain = energy_bin_range
                    red = reducer(energies, dataset, bin_e)
                elif use_recent_binning and cls.reduction_bin_domain is not None:
                    # Uses the most recent binning settings.
                    red = reducer(energies, dataset, bin_e)
                else:
                    # Create a QT application to run the dialog.
                    if QtWidgets.QApplication.instance() is None:
                        app = QtWidgets.QApplication([])
                    # Run the Bin Selector dialog
                    window = EnergyBinReducerDialog(
                        energies=energies, dataset=dataset, bin_energies=bin_e
                    )
                    # window.reducerUI.bin_axes.set_xlabel("")
                    window.show()
                    if window.exec():
                        # If successful, store the binning settings and data reducer
                        cls.reduction_bin_domain = window.domain
                        red = window.reducer
                    else:
                        raise ValueError("No binning settings selected.")

                # Use the binning settings to reduce the data.
                _, reduced_summed_data = red.reduce_by_sum(
                    bin_domain=cls.reduction_bin_domain, axis=None  # all axes
                )
                _, reduced_single_detector_data = red.reduce_by_sum(
                    bin_domain=cls.reduction_bin_domain,
                    axis="bin_energies",  # just the bin energies
                )

            # 3D array unhandled.
            if len(mda_arrays) > 2:
                warnings.warn(
                    "MDA file(s) contain more than two dimension, handling of higher dimensions is not yet implemented."
                )
        # Collect units and labels:
        scan_1d = mda_scans[0]
        positioners = scan_1d.positioners
        detectors = scan_1d.detectors
        if len(mda_scans) > 1:
            scan_2d = mda_scans[1]
            positioners += scan_2d.positioners
            detectors += scan_2d.detectors

        column_types = {
            "Positioner": (
                "name",
                "descr",
                "step mode",
                "unit",
                "rdbk name",
                "rdbk descr",
                "rdbk unit",
            ),
            "Detector": (
                "name",
                "descr",
                "unit",
            ),
        }
        column_descriptions = {
            i: [
                p.name,
                p.desc,
                p.step_mode,
                p.unit,
                p.readback_name,
                p.readback_desc,
                p.readback_unit,
            ]
            for i, p in enumerate(positioners)
        }
        column_descriptions.update(
            {
                i + len(positioners): [d.name, d.desc, d.unit]
                for i, d in enumerate(detectors)
            }
        )
        params["column_types"] = column_types
        params["column_descriptions"] = column_descriptions

        # Collect units and labels:
        for i, p in enumerate(positioners):
            labels.append(p.name)
            units.append(p.unit)
        for i, d in enumerate(detectors):
            labels.append(d.name)
            units.append(d.unit)
        # If 2D data is present, add reduced data to 1D data.
        if not header_only and len(mda_arrays) > 1:
            # Check rows (energies) match length
            assert reduced_summed_data.shape[0] == mda_1d.shape[0]
            assert reduced_single_detector_data.shape[0] == mda_1d.shape[0]
            # Add reduced data to 1D data as extra columns
            mda_1d = np.c_[mda_1d, reduced_single_detector_data, reduced_summed_data]
            # Add labels and units for reduced data
            # (Detector data labels/units already added via positioners and detectors above.)
            labels += ["MCA Sum (Reduced)"]
            units += ["a.u."]
        # Use scan time if available, otherwise let system time be used.
        if "MEX1ES01GLU01:MEX_TIME" in params:
            params["created"] = params["MEX1ES01GLU01:MEX_TIME"]
        return mda_1d, labels, units, params


def MEX1_to_QANT_AUMainAsc(
    parser: parser_base,
    extrainfo_mapping={
        "SR14ID01MCS02FAM:X.RBV": None,  # "Mono.d_spacing",
        "SR14ID01MCS02FAM:Y.RBV": None,
        "SR14ID01MCS02FAM:Z.RBV": None,
        "SR14ID01MCS02FAM:R1.RBV": None,
        "SR14ID01MCS02FAM:R2.RBV": None,
        "SR14ID01NEXSCAN:saveData_comment1": "Sample",
        "SR14ID01NEXSCAN:saveData_comment2": None,  # "Reference.Element",
    },
) -> list[str]:
    """
    Converts a parser mapping to to QANT format.

    Parameters
    ----------
    parser : parser_base
        The parser object (with data, labels, units, and params loaded) to convert.
    extrainfo_mapping : dict[str:str|None], optional
        Optional mapping for known read-values for the QANT AUMainAsc format to
        parser parameter names. By default the dictionary key values are:
            {"SR14ID01MCS02FAM:X.RBV": None,
            "SR14ID01MCS02FAM:Y.RBV": None,
            "SR14ID01MCS02FAM:Z.RBV": None,
            "SR14ID01MCS02FAM:R1.RBV": None,
            "SR14ID01MCS02FAM:R2.RBV": None,
            "SR14ID01NEXSCAN:saveData_comment1": None,
            "SR14ID01NEXSCAN:saveData_comment2": None,}

    Returns
    -------
    list[str]
        A list of lines for the QANT AUMainAsc format, with newline terminations included.
    """
    possible_read_values = [
        "SR14ID01MCS02FAM:X.RBV",
        "SR14ID01MCS02FAM:Y.RBV",
        "SR14ID01MCS02FAM:Z.RBV",
        "SR14ID01MCS02FAM:R1.RBV",
        "SR14ID01MCS02FAM:R2.RBV",
        "SR14ID01NEXSCAN:saveData_comment1",
        "SR14ID01NEXSCAN:saveData_comment2",
    ]
    # Check validity of the extrainfo_mapping
    for key, value in extrainfo_mapping.items():
        if value is not None:
            if value not in parser.params:
                raise ValueError(f"Parameter {value} not found in parser params.")
            elif key not in possible_read_values:
                raise ValueError(f"Parameter {key} not found in possible read values.")
    # Create reverse dict
    extrainfo_remapping = {v: k for k, v in extrainfo_mapping.items()}

    # Check vailidty of parser object:
    if parser.data is None:
        raise ValueError("Parser object does not have data loaded.")

    # Check validity of the dimensionality of the object:
    if len(parser.data.shape) != 2:
        raise ValueError("Parser object data is not 2D.")

    # Define a container for the output strings, line by line.
    ostrs = []

    # Define fake asc version number:
    ostrs.append("## mda2ascii 0.3.2 generated output\n")
    ostrs.append("\n")
    ostrs.append("\n")

    # Rename for consistency between mda and asc formats.
    mda_param_names = ["mda_version", "mda_scan_number", "mda_rank", "mda_dimensions"]
    asc_param_names = [
        "MDA File Version",
        "Scan Number",
        "Overall scan dimension",
        "Total requested scan size",
    ]
    for mda, asc in zip(mda_param_names, asc_param_names):
        if mda in parser.params:
            parser.params[asc] = parser.params[mda]
            del parser.params[mda]

    # Define MDA versioning from parameters, or create fictitious version.
    ostrs.append(
        "# MDA File Version = 1.3\n"
        if "MDA File Version" not in parser.params
        else f"# MDA File Version = {parser.params['MDA File Version']}\n"
    )
    ostrs.append(
        "# Scan Number = 1\n"
        if "Scan Number" not in parser.params
        else f"# Scan Number = {parser.params['Scan Number']}\n"
    )
    ostrs.append(
        "# Overall scan dimension = 1-D\n"
        if "Overall scan dimension" not in parser.params
        else f"# Overall scan dimension = {parser.params['Overall scan dimension']}-D\n"
    )
    ostrs.append(
        f"# Total requested scan size = {len(parser.data)}\n"
        if "Total requested scan size" not in parser.params
        else f"# Total requested scan size = {parser.params['Total requested scan size']}\n"
    )
    ostrs.append("\n")
    ostrs.append("\n")

    # Define the extra PVs
    ostrs.append("#  Extra PV: name, descr, values (, unit)\n")
    ostrs.append("\n")
    param_idx = 1
    for param in parser.params:
        if param not in asc_param_names:
            wparam = (
                param
                if param not in extrainfo_remapping
                or extrainfo_remapping[param] is None
                else extrainfo_remapping[param]
            )
            line = f"# Extra PV {param_idx}: {wparam}"
            if (
                hasattr(parser.params[param], "__len__")
                and not type(parser.params[param]) is str
            ):
                for val in parser.params[param]:
                    wval = (
                        val
                        if val not in extrainfo_remapping
                        or extrainfo_remapping[val] is None
                        else extrainfo_remapping[val]
                    )
                    line += f", {wval}"
                line += "\n"
            else:
                line += f", {parser.params[param]}, \n"
            ostrs.append(line)
            param_idx += 1
    ostrs.append("\n")
    ostrs.append("\n")

    # Define the scan header:
    ostrs.append("# 1-D Scan\n")
    ostrs.append(
        f"# Points completed = {parser.data.shape[0]} of {parser.data.shape[0]}\n"
        if "Points completed" not in parser.params
        else f"# Points completed = {parser.params['Points completed']}\n"
    )
    ostrs.append(
        f"# Scanner = SR14ID01NEXSCAN:scan1\n"
        if "Scanner" not in parser.params
        else f"# Scanner = {parser.params['Scanner']}\n"
    )
    if "Scan time" in parser.params:
        ostrs.append(f"# Scan time = {parser.params['Scan time']}\n")
    elif "created" in parser.params and isinstance(
        parser.params["created"], dt.datetime
    ):
        ostrs.append(
            f"# Scan time = {parser.params['created'].strftime(r"%b %d, %Y %H:%M:%S.%f")}\n"
        )
    elif "modified" in parser.params and isinstance(
        parser.params["modified"], dt.datetime
    ):
        ostrs.append(
            f"# Scan time = {parser.params['modified'].strftime(r"%b %d, %Y %H:%M:%S.%f")}\n"
        )
    else:
        # Use current time
        ostrs.append(
            f"# Scan time = {dt.datetime.now().strftime(r"%b %d, %Y %H:%M:%S.%f")}\n"
        )
    ostrs.append("\n")

    if "column_types" in parser.params and isinstance(
        parser.params["column_types"], dict
    ):

        for coltype in parser.params["column_types"].keys():
            line = f"#  {coltype}:"
            for val in parser.params["column_types"][coltype]:
                line += f" {val}"
                if val != parser.params["column_types"][coltype][-1]:
                    line += ","
                else:
                    line += "\n"
            ostrs.append(line)
    else:
        # Use default
        ostrs.append(
            "#  Positioner: name, descr, step mode, unit, rdbk name, rdbk descr, rdbk unit\n"
        )
        ostrs.append("#  Detector: name, descr, unit\n")
        ostrs.append("\n")

    # Define the Column Descriptions
    ostrs.append("# Column Descriptions:\n")
    if "column_descriptions" in parser.params:
        # Column descriptions have been saved. Use these.
        init_idx = 1  # default, add 1 to column description indexes
        if parser.params["column_descriptions"][0][0] != "Index":
            # Create index column if not present.
            ostrs.append(f"#{1:5}  [     Index      ]\n")
            init_idx = 2  # Because of added extra index.
        for i, col in enumerate(parser.params["column_descriptions"]):
            if "column_type_assignments" in parser.params:
                col_type = parser.params["column_type_assignments"][i]
            else:
                col_type = f"1-D Detector{i+1:4}"  # Nth detector
            line = f"#{i+init_idx+1:5}  [" + col_type + "]  "
            for val in col:
                line += f"{val}" if val is not None else ""
                if val != col[-1]:
                    line += ", "
                else:
                    line += "\n"
            ostrs.append(line)
    else:
        # Create column descriptions from units and labels.
        ostrs.append(f"#{1:5}  [     Index      ]\n")
        for i, label in enumerate(parser.labels):
            line = f"#{i+2:5}  "
            if i == 0:
                # Assume energy labels...
                line += f"[1-D Positioner 1]  {label}, "
                if "units" in parser.params and parser.params["units"][i] is not None:
                    unit = parser.params["units"][i]
                    line += f"Mono setpoint, TABLE, {unit}, {label}, Mono setpoint, {unit}\n"
                else:
                    line += f"Mono setpoint, TABLE, eV, {label}, Mono setpoint, eV\n"
            else:
                # Assume detector labels, no description.
                line += f"[1-D Detector{i:4}]  {label}, "
                if "units" in parser.params and parser.params["units"][i] is not None:
                    unit = parser.params["units"][i]
                    line += f"{label}, , {unit}\n"
                else:  # no unit.
                    line += f"{label}, , \n"
            ostrs.append(line)
    ostrs.append("\n")

    # Define the Scan Values
    ostrs.append("# 1-D Scan Values\n")
    for i, row in enumerate(parser.data):
        line = f"{i+1}"
        for val in row:
            line += f"\t{val}"
        line = line[:-1] + "\n"
        ostrs.append(line)

    # End of file
    return ostrs


if __name__ == "__main__":
    # Example usage
    path = os.path.dirname(__file__)
    package_path = os.path.normpath(os.path.join(path, "../../../../"))

    # Open an mdi file
    mda_path = os.path.normpath(
        os.path.join(package_path, "tests/test_data/au/MEX1/MEX1_40747.mda")
    )

    # HEADER
    test1 = MEX1_NEXAFS(mda_path, header_only=True)
    # BODY
    test2 = MEX1_NEXAFS(mda_path, header_only=False)
    # Check if previous binning is applied to new data.
    test3 = MEX1_NEXAFS(mda_path, header_only=False)

    import matplotlib.pyplot as plt

    plt.close("all")
    subplts = plt.subplots(1, 1)
    fig: plt.Figure = subplts[0]
    ax: plt.Axes = subplts[1]
    ave_mca_idx = -1
    energy_idx = test2.search_label_index("ring_current")
    ax.plot(
        test2.data[:, energy_idx],
        test2.data[:, ave_mca_idx],
        label="Test2" + test2.labels[ave_mca_idx],
    )
    ax.set_xlabel(test2.labels[energy_idx])
    # plt.ioff()

    # Also open an xdi file
    xdi_path = os.path.normpath(
        os.path.join(package_path, "tests/test_data/au/MEX1/MEX1_40747_processed.xdi")
    )
    test4 = MEX1_NEXAFS(xdi_path, header_only=False)
    energy_idx2 = test4.search_label_index("ring_current")
    ax.plot(
        test4.data[:, energy_idx2],
        test4.data[:, ave_mca_idx],
        label="Test4" + test4.labels[ave_mca_idx],
    )
    print(test4.labels[energy_idx2])

    ax.legend()
    plt.ion()
    # plt.show(block=False)
    plt.show(block=True)
