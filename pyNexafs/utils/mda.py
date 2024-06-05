import os, io
from xdrlib3 import Unpacker, Packer
import pandas as pd
from enum import Enum
from typing import Any


class DBR(Enum):
    """Enumerate for the type of the parameter in the MDA file.

    Seems to be based off EPICS data storage.
    https://epics.anl.gov/docs/CAproto.html"""

    STRING = 0
    SHORT = 1
    INT = 2
    FLOAT = 3
    ENUM = 4
    LONG = 5
    DOUBLE = 6
    STS_STRING = 7
    STS_SHORT = 8
    STS_FLOAT = 9
    STS_ENUM = 10
    STS_CHAR = 11
    STS_LONG = 12
    STS_DOUBLE = 13
    TIME_STRING = 14
    TIME_SHORT = 15
    TIME_FLOAT = 16
    TIME_ENUM = 17
    TIME_CHAR = 18
    TIME_DOUBLE = 19
    GR_STRING = 20
    GR_SHORT = 21
    GR_INT = 22
    GR_FLOAT = 23
    GR_ENUM = 24
    GR_CHAR = 25
    GR_LONG = 26
    GR_DOUBLE = 27
    CTRL_STRING = 28
    CTRL_SHORT = 29  # INT16
    CTRL_FLOAT = 30
    CTRL_ENUM = 31
    CTRL_CHAR = 32  # UINT8
    CTRL_LONG = 33  # INT32
    CTRL_DOUBLE = 34


class MDAFileReader:
    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        self.path = path
        self._buffered_reader = None

        self.pos_header = 0
        self.pos_pExtra = None
        self.pos_main_scan = None

    @property
    def buffered_reader(self) -> io.BufferedReader:
        if self._buffered_reader is None:
            self._buffered_reader = open(self.path, "rb")
        return self._buffered_reader

    @buffered_reader.deleter
    def buffered_reader(self):
        if self._buffered_reader is not None:
            self._buffered_reader.close()
            self._buffered_reader = None

    def read_header(
        self, byte_default=100
    ) -> tuple[float, int, int, list[int], int, int, int]:
        """Reads the header information of an MDA file.

        Records the following MDA file variables:
            version,
            scan_number,
            rank,
            dimensions,
            isRegular,
            pExtra,
            pmain_scan

        Parameters
        ----------
        byte_default : int, optional
            The number of bytes to read from the file. Default is 100.

        Returns
        -------
        tuple[float, int, int, list[int], int, int, int]
            A tuple of the header information containing:
                version : float
                    The version of the MDA file.
                scan_number : int
                    The scan number of the MDA file.
                rank : int
                    The rank (number of dimensions) of the MDA file.
                dimensions,
                    The dimensions (size of each dimension) of the MDA file.
                isRegular,
                    The regularity of the MDA file. TODO: Find out what this means.
                pExtra,
                    The byte location of the extra parameters in the MDA file.
                pmain_scan
                    The byte location of the main scan in the MDA file.
        """
        br = self.buffered_reader
        br.seek(0)
        data = br.read(byte_default)
        # Unpack the header
        u = Unpacker(data)
        header = MDAFileReader._unpack_header(u)
        # Record the position of the scan in the file.
        self.pos_main_scan = br.tell() - (byte_default - u.get_position())
        del br
        # Record the position of the pExtra in the file.
        self.pos_pExtra = header[-1]
        # Add the main scan position to the header
        return (*header, self.pos_main_scan)

    @staticmethod
    def _unpack_header(
        u: Unpacker, min_rank: int = 1
    ) -> tuple[float, int, int, list[int], int, int]:
        """
        Unpacking script for header information of the MDA file.

        Parameters
        ----------
        u : Unpacker
            Unpacking object of the initial bytes of the MDA file.
        min_rank : int, optional
            The minimum buffer size of the MDA file. Default is 1.

        Returns
        -------
        tuple[float, int, int, list[int], int, int]
            A tuple of the header information containing:
                version : float
                    The version of the MDA file.
                scan_number : int
                    The scan number of the MDA file.
                rank : int
                    The rank (number of dimensions) of the MDA file.
                dimensions,
                    The dimensions (size of each dimension) of the MDA file.
                isRegular,
                    The regularity of the MDA file.
                pExtra,
                    The byte location of the extra parameters in the MDA file.
        """
        # Check if the unpacker is sufficiently long enough to contain the header information.
        if len(u.get_buffer()) < 4 * 5 + 4 * min_rank:
            raise ValueError(
                "Unpacker obj insufficiently long to decode all header information."
            )
        # Decode
        version = u.unpack_float()  # 4 bytes
        scan_number = u.unpack_int()  # 4 bytes
        rank = u.unpack_int()  # 4 bytes
        dimensions = u.unpack_farray(rank, u.unpack_int)  # 4 bytes * rank
        isRegular = u.unpack_int()  # 4 bytes
        pExtra = u.unpack_int()  # 4 bytes
        return version, scan_number, rank, dimensions, isRegular, pExtra

    @staticmethod
    def header_to_dataFrame(
        header: tuple[float, int, int, list[int], int, int, int]
    ) -> pd.DataFrame:
        (
            version,
            scan_number,
            rank,
            dimensions,
            is_regular,
            p_extra,
            p_main_scan,
        ) = header
        return pd.DataFrame(
            {
                "version": [version],
                "scan_number": [scan_number],
                "rank": [rank],
                "dimensions": [dimensions],
                "isRegular": [is_regular],
                "pExtra": [p_extra],
                "pmain_scan": [p_main_scan],
            }
        )

    def read_parameters(self) -> dict[str, tuple[str, str, str]]:
        """
        Reads the pExtra values in the MDA file.

        Returns
        -------
        dict[str, tuple[str, str, str]]
            A dictionary of pExtra data, ordered by name : tuple.
            The tuple contains the following data:
                description : str
                    The description of the parameter.
                unit : str
                    The unit of the parameter.
                value : str
                    The value of the parameter.
        """
        if self.pos_pExtra is None:
            self.read_header()
            if self.pos_pExtra is None:
                # Check if the pExtra position was found in the MDA file.
                raise ValueError("pExtra position not found in MDA file.")
        br = self.buffered_reader
        br.seek(self.pos_pExtra)
        buff = br.read()
        u = Unpacker(buff)
        params = MDAFileReader._read_pExtra(u)
        del self.buffered_reader
        return params

    @staticmethod
    def _read_pExtra(
        u: Unpacker,
    ) -> dict[
        str | None, tuple[str | None, str | None, Any] | tuple[str | None, str | None]
    ]:
        """
        Reads the parameter information from a pExtra buffer of the MDA file.

        Parameters
        ----------
        u : Unpacker
            Unpacker object of the pExtra section of the MDA file.

        Returns
        -------
        dict[str | None, tuple[str | None, str | None , Any] | tuple[str | None, str | None]]
            A dictionary of pExtra data, ordered by name : tuple.
            The tuple contains 3 items, with the following data:
                description : str | None
                    The description of the parameter.
                unit : str | None
                    The unit of the parameter.
                value : Any
                    The value of the parameter.

        Raises
        ------
        ImportError
            _description_
        """
        # Read the number of extra parameters
        numExtra = u.unpack_int()
        params = {}
        for i in range(numExtra):
            ## Read the name of the parameter
            len_name = u.unpack_int()
            if len_name:
                # Repeated length check
                assert len_name == u.unpack_int()
                name = u.unpack_fstring(len_name).decode("utf-8")
            else:
                name = None
            ## Read the description of the parameter:
            len_desc = u.unpack_int()
            if len_desc:
                # Repeated length check
                assert len_desc == u.unpack_int()
                desc = u.unpack_fstring(len_desc).decode("utf-8")

            else:
                desc = None

            ## Does the parameter have a unit?
            count = None
            param_type = u.unpack_int()
            if param_type != 0:  # Then is not simple string value, and has unit.
                count = u.unpack_int()  # Number of values in the parameter
                len_unit = u.unpack_int()  # Length of the unit string
                if len_unit:
                    assert len_unit == u.unpack_int()
                    unit = u.unpack_fstring(len_unit).decode("utf-8")
                else:
                    unit = None
            else:
                unit = None

            ## Read the value of the parameter:
            param_type = DBR(param_type)
            match param_type:
                case DBR.STRING:
                    val_len = u.unpack_int()
                    if val_len:
                        assert val_len == u.unpack_int()
                        val = u.unpack_fstring(val_len).decode("utf-8")
                    else:
                        val = None
                case DBR.CTRL_CHAR:
                    arr = u.unpack_farray(count, u.unpack_int)
                    val = ""
                    # Need to treat byte array as null-terminated string
                    for i in range(count):
                        if arr[i] == 0:
                            break
                        val += chr(arr[i])

                case DBR.CTRL_SHORT:
                    val = u.unpack_farray(count, u.unpack_int)
                case DBR.CTRL_LONG:
                    val = u.unpack_farray(count, u.unpack_int)
                case DBR.CTRL_FLOAT:
                    val = u.unpack_farray(count, u.unpack_float)
                case DBR.CTRL_DOUBLE:
                    val = u.unpack_farray(count, u.unpack_double)
                case _:
                    raise ImportError(
                        f"Unspecified unpacking for DBR type: {param_type}"
                    )
            if count is not None:
                if len(val) == 1:
                    val = val[0]
            params[name] = (desc, unit, val)
            print(name, param_type, params[name])
        return params
