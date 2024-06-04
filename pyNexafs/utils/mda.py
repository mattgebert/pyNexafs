import os, io
from xdrlib3 import Unpacker, Packer
import pandas as pd


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

    # @staticmethod
    # def decode_string(u: Unpacker) -> str:
    #     length = u.unpack_int()
    #     if length:
    #         #TODO: Unpacker doesn't use length parameter (already depacks internally)
    #         return u.unpack_string(length)
    #     else:
    #         return ""

    @staticmethod
    def _read_pExtra(u: Unpacker) -> dict[str, tuple[str, str, str]]:
        # Read the number of extra parameters
        numExtra = u.unpack_int()
        print(numExtra)
        params = {}
        for i in range(numExtra):
            name = u.unpack_string()
            desc = u.unpack_string()
            dtype = u.unpack_int()

            print(name, desc, dtype)
            break
            # name, desc, unit, val = MDAFileReader.(u)
            # params[name] = (desc, unit, val)
        return None
