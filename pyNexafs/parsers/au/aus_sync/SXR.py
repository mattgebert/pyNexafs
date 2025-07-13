"""
Parser classes for the Soft X-ray (SXR) beamline at the Australian Synchrotron.
"""

from pyNexafs.parsers import parser_base
from pyNexafs.utils.mda import MDAFileReader
from io import TextIOWrapper
from typing import Any
from numpy.typing import NDArray
import numpy as np
import overrides


class SXR_NEXAFS(parser_base):
    """
    Australian Synchrotron Soft X-ray (SXR) NEXAFS parser.

    Parses data formats including '.asc' and '.mda' formats from the SXR
    Near Edge X-ray Absorption Fine Structure (NEXAFS) tool.

    Attributes
    ----------
    ALLOWED_EXTENSIONS
    COLUMN_ASSIGNMENTS
    SUMMARY_PARAM_RAW_NAMES
    RELABELS

    Notes
    -----
    Implemented for data as of 2024-Mar.
    """

    ALLOWED_EXTENSIONS = [
        ".asc",
        ".mda",
    ]  # ascii files exported from the binary .mda files generated.

    COLUMN_ASSIGNMENTS = {
        "x": "SR14ID01PGM_CALC_ENERGY_MONITOR.P",
        "y": [
            # "SR14ID01PGM:LOCAL_SP",
            "SR14ID01IOC68:scaler1.S20",  # Drain Current VF
            "SR14ID01IOC68:scaler1.S17",  # Direct PHD VF
            "SR14ID01IOC68:scaler1.S18",  # I0 VF
            "SR14ID01IOC68:scaler1.S19",  # Ref Foil VF
            "SR14ID01IOC68:scaler1.S20",  # Drain Current VF
            "SR14ID01IOC68:scaler1.S21",  # Channeltron Front (PEY)
            "SR14ID01IOC68:scaler1.S22",  # MCP (TFY)
            "SR14ID01IOC68:scaler1.S23",  # Hemispherical Analyser (AEY)
            # No longer included in files...?
            # "SR14ID01IOC68:scaler1.S3", # I0 VF
            # "SR14ID01IOC68:scaler1.S4", # Ref Foil VF
            # "SR14ID01IOC68:scaler1.S8", # Direct PHD VF
        ],
        "y_errs": None,
        "x_errs": None,
    }

    SUMMARY_PARAM_RAW_NAMES = [
        "SR14ID01NEXSCAN:saveData_comment1",
        "SR14ID01NEXSCAN:saveData_comment2",
        "SR14ID01NEX01:R_MTR.RBV",
        "SR14ID01NEX01:RULER_ID",
        "SR14ID01:BRANCH_MODE",
        "SR14ID01NEX01:C_MTR.RBV",
        "SR14ID01NEX01:Z_MTR.RBV",
        "SR14ID01NEX01:X_MTR.RBV",
        "SR14ID01NEX01:Y_MTR.RBV",
    ]

    RELABELS = {
        "SR14ID01PGM:REMOTE_SP": "Photon Energy",
        "SR14ID01PGM:LOCAL_SP": "Local Energy Setpoint",
        "SR14ID01PGM_ENERGY_SP": "Energy Setpoint",
        "SR14ID01PGM_CALC_ENERGY_MONITOR.P": "Encoder Photon Energy",
        "SR14ID01IOC68:scaler1.TP": "Exp Time",
        "SR14ID01IOC68:scaler1.S2": "Drain Current VF",
        "SR14ID01IOC68:scaler1.S3": "I0 VF",
        "SR14ID01IOC68:scaler1.S4": "Ref Foil VF",
        "SR14ID01IOC68:scaler1.S6": "MCP",
        "SR14ID01IOC68:scaler1.S8": "Direct PHD VF",
        "SR14ID01IOC68:scaler1.S9": "BL PHD VF",
        "SR14ID01IOC68:scaler1.S10": "Channeltron",
        "SR14ID01IOC68:scaler1.S11": "TFY PHD VF",
        "SR14ID01IOC68:scaler1.S18": "I0 VF #2",
        "SR14ID01IOC68:scaler1.S19": "Ref Foil VF #2",
        "SR14ID01IOC68:scaler1.S20": "Drain Current VF #2",
        "SR14ID01IOC68:scaler1.S21": "Channeltron Front (PEY)",
        "SR14ID01IOC68:scaler1.S22": "MCP (TFY)",
        "SR14ID01IOC68:scaler1.S23": "Hemispherical Analyser (AEY)",
        "SR14ID01IOC68:scaler1.S17": "Direct PHD VF #2",
        "SR14ID01AMP01:CURR_MONITOR": "Drain Current - Keithley1",
        "SR14ID01AMP02:CURR_MONITOR": "BL PHD - Keithley2",
        "SR14ID01AMP03:CURR_MONITOR": "I0 - Keithley3",
        "SR14ID01AMP04:CURR_MONITOR": "Ref Foil -Keithley4",
        "SR14ID01AMP05:CURR_MONITOR": "Direct PHD - Keithley5",
        "SR14ID01AMP06:CURR_MONITOR": "Keithley6",
        "SR14ID01AMP07:CURR_MONITOR": "Ref Foil - Keithley7",
        "SR14ID01AMP08:CURR_MONITOR": "Drain Current - Keithley8",
        "SR14ID01AMP09:CURR_MONITOR": "I0 - Keithley9",
        "SR14ID01:BL_GAP_REQUEST": "Undulator Gap Request",
        "SR14ID01:GAP_MONITOR": "Undulator Gap Readback",
        "SR11BCM01:CURRENT_MONITOR": "Ring Current",
        "SR14ID01NEXSCAN:saveData_comment1": "Note 1",
        "SR14ID01NEXSCAN:saveData_comment2": "Note 2",
        "SR14ID01:BRANCH_MODE": "Branch",
        "SR14ID01NEX01:RULER_ID": "Ruler ID",
        "SR14ID01NEX01:R_MTR.RBV": "R",
        "SR14ID01NEX01:C_MTR.RBV": "C",
        "SR14ID01NEX01:Z_MTR.RBV": "Z",
        "SR14ID01NEX01:X_MTR.RBV": "X",
        "SR14ID01NEX01:Y_MTR.RBV": "Y",
    }

    @classmethod
    def parse_asc_202403(
        cls, file: TextIOWrapper, header_only: bool = False
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """
        Read Australian Synchrotron SXR NEXAFS `.asc` files.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.asc', 'r')).
        header_only : bool, optional
            If True, then only the header of the file is read and NDArray is returned as None, by default False.

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
            If the file is not a valid .asc file.
        """

        # Initialise structures
        params = {"filename": file.name}
        labels = []
        units = []

        # Check valid format.
        if not file.name.endswith(".asc"):
            raise ValueError(f"File {file.name} is not a valid .asc file.")

        ### Read file
        # Check header is correct
        assert file.readline() == "## mda2ascii 0.3.2 generated output\n"

        # skip 2 lines
        [file.readline() for i in range(2)]

        ## 1 Initial Parameters including MDA File v, Scan Num, Ovrl scn dim, scn size
        for i in range(4):
            line = file.readline()[2:].strip().split(" = ", 1)
            if len(line) == 2:
                params[line[0]] = line[1]

        # skip 2 lines
        [file.readline() for i in range(2)]
        # Check PV header is correct
        assert file.readline() == "#  Extra PV: name, descr, values (, unit)\n"

        # skip 1 line
        file.readline()

        ## 2 Extra Parameter/Values
        line = file.readline()
        while line != "\n":
            # Discard first 11 characters and linenumber '# Extra PV 1:'.
            line = line.strip().split(":", 1)[1]
            # Separate the name, description, values and units.
            line = line.split(",")
            # Trim values and append to params in a tuple.
            vals = [line[i].strip().replace('"', "") for i in range(1, len(line))]
            # Convert string values to float / int if possible:
            if len(vals) == 4:  # has units column
                try:
                    vals[3] = float(vals[3]) if "." in vals[3] else int(vals[3])
                except ValueError:
                    pass
            # Add vals to params
            params[line[0].strip()] = vals

            # read the newline
            line = file.readline()

        # skip 1 extra line
        file.readline()

        ## 3 Scan Header:
        assert file.readline() == "# 1-D Scan\n"
        for i in range(3):
            line = file.readline()[2:].strip().split(" = ", 1)
            if len(line) == 2:
                params[line[0]] = line[1]

        # skip 1 line
        file.readline()

        ## 4 Scan Column properties:
        # Labels for columns describing data columns.
        # i.e.
        # i.e. #  Positioner: name, descr, step mode, unit, rdbk name, rdbk descr, rdbk unit
        column_types = {}
        line = file.readline()
        while line != "\n":
            line = line.strip().split(": ", 1)
            col_name = line[0][3:]  # ignore "#  " before name
            col_attrs = line[1].split(", ")
            column_types[col_name] = col_attrs
            # read the newline
            line = file.readline()

        # skip 1 line
        file.readline()

        # Column descriptions
        # i.e. #    2  [1-D Positioner 1]  SR14ID01PGM:LOCAL_SP, Mono setpoint, TABLE, eV, SR14ID01PGM:LOCAL_SP, Mono setpoint, eV
        column_descriptions = {}
        column_type_assignments = {}
        line = file.readline()
        while line != "\n":
            # Take index info
            index = int(line[1:6].strip())
            index_line = (
                index == 1
            )  # Boolean to determine if on index description line.
            # Take coltype info
            desc_type = line[8:26]
            assert (
                desc_type[0] == "[" and desc_type[-1] == "]"
            )  # Check parenthesis of the line parameters
            desc_type = desc_type[1:-1].strip()
            # Check if valid col type
            valid = False if not index_line else True
            for coltype in column_types:
                if coltype in desc_type:
                    valid = True
                    break
            if not valid:
                raise ValueError(f"Invalid column type {desc_type} in line {line}")

            # Take info
            desc_info = line[28:].split(", ")
            column_descriptions[index] = desc_info
            column_type_assignments[index] = desc_type
            # Check that the initial parameter begins with the Instrument descriptor.
            if not index_line:
                assert desc_info[0].startswith("SR14ID01")  # code for initial
            # Add to labels and units to lists
            labels += [desc_info[0].strip()] if not index_line else ["Index"]
            if "Positioner" in desc_type:
                pUnit = desc_info[3].strip()  # hardcoded 'unit' position
                units += [pUnit]
            elif "Detector" in desc_type:
                dUnit = desc_info[2].strip()  # hardcoded 'unit' position
                units += [dUnit if dUnit != "" else None]
            elif index_line:
                units += [None]

            # read next line
            line = file.readline()

        # add column data to params
        params["column_types"] = column_types
        params["column_descriptions"] = column_descriptions
        params["column_type_assignments"] = column_type_assignments

        if header_only:
            # Do not process remaining lines
            return None, labels, units, params

        ## 5 Scan Values
        assert file.readline() == "# 1-D Scan Values\n"
        lines = file.readlines()  # read remaining lines efficiently

        # Convert data to numpy array.
        data = np.loadtxt(lines, delimiter="\t")
        # [np.loadtxt(line, delimiter="\t")
        #     for line in lines]
        data = np.array(data)

        return data, labels, units, params

    @classmethod
    def parse_mda(
        cls, file: TextIOWrapper, header_only: bool = False
    ) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """
        Read Australian Synchrotron SXR NEXAFS .mda files.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.mda', 'r')).
        header_only : bool, optional
            If True, only the header of the file is read and NDArray is returned as None, by default False.

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

        # Check valid format.
        if not file.name.endswith(".mda"):
            raise ValueError(f"File {file.name} is not a valid .mda file.")

        # Need to reopen the file in byte mode.
        file.close()
        mda = MDAFileReader(file.name)

        mda_header = mda.read_header_as_dict()
        if mda_header["mda_rank"] != 1:
            raise ValueError("MDA file is not 1D, incompatible for regular NEXAFS.")
        mda_params = mda.read_parameters()
        mda_arrays, mda_scans = mda.read_scans(header_only=header_only)

        # Add values to params dict
        params.update(mda_header)
        params.update(mda_params)
        # Add column types and descriptions to params.
        mda_1d = mda_arrays[0]
        scan_1d = mda_scans[0]
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
            for i, p in enumerate(scan_1d.positioners)
        }
        column_descriptions.update(
            {
                i + len(scan_1d.positioners): [d.name, d.desc, d.unit]
                for i, d in enumerate(scan_1d.detectors)
            }
        )
        params["column_types"] = column_types
        params["column_descriptions"] = column_descriptions
        # Collect units and labels:
        labels = []
        units = []
        for i, p in enumerate(scan_1d.positioners):
            labels.append(p.name)
            units.append(p.unit)
        for i, d in enumerate(scan_1d.detectors):
            labels.append(d.name)
            units.append(d.unit)

        if header_only:
            return None, labels, units, params
        return mda_1d, labels, units, params

    @property
    @overrides.overrides
    def summary_param_values(self) -> list[Any]:
        """
        Return a list of important parameter values of the data file.

        Uses the list element corresponding to 'value' for each file.
        Overrides base method to use the 'value' element of the SXR parameter list.

        Returns
        -------
        list
            List of important parameter values.
        """
        # Get second element which is the parameter number.
        return [self.params[key][1] for key in self.SUMMARY_PARAM_RAW_NAMES]

    @property
    @overrides.overrides
    def summary_param_names_with_units(self) -> list[str]:
        """
        Return a list of important parameter names with units.

        Requires a loaded dataset to return the units of the parameters.
        Not a pre-defined class method.

        Returns
        -------
        list
            List of important parameter names with units.
        """
        pNames = self.SUMMARY_PARAM_RAW_NAMES
        pUnits = [
            (
                self.params[pName][2]
                if (
                    self.params is not None  # Params loaded
                    and pName in self.params  # Parameter listed
                    and hasattr(
                        self.params[pName], "__len__"
                    )  # Parameter has a list of values
                    and len(self.params[pName])
                    == 3  # 3 params for value, description, unit
                    and self.params[pName][2] != ""  # Unit value is not empty.
                )
                else None
            )
            for pName in pNames
        ]
        if not self.relabel:
            return [
                f"{pName} ({unit})" if unit is not None else pName
                for pName, unit in zip(pNames, pUnits)
            ]
        else:
            relabel_names = []
            for pName, unit in zip(pNames, pUnits):
                if unit is not None:
                    relabel_names.append(
                        f"{self.RELABELS[pName]} ({unit})"
                        if pName in self.RELABELS
                        else f"{pName} ({unit})"
                    )
                else:
                    relabel_names.append(
                        self.RELABELS[pName] if pName in self.RELABELS else pName
                    )
            return relabel_names
