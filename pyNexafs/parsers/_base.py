import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abc
import sys, io, os
from io import TextIOWrapper
from typing import Any, TypeVar
from numpy.typing import NDArray
from ..nexafs.scan import base_scan

PARSER = TypeVar("PARSER", bound="parser_base") #typing hint for parser_base class inheritance.

class parser_base(metaclass=abc.ABCMeta):
    """ General class that parses raw datafiles to acquire useful data information. 
        Requires implementation of file_parser method.
    """
    
    ALLOWED_EXTENSIONS = []
    
    def __init__(self, filepath: str | None = None) -> None:
        
        # ABC super.
        super().__init__()
        # Initialise variables
        self.filepath = filepath
        self.data = None
        self.labels = []
        self.units = []
        self.params = {}
        
        if filepath is None:
            return
        elif type(filepath) is str:
            self.load() # Load data
        else:
            raise TypeError(f'Filepath is {type(filepath)}, not str.') 
        
    @classmethod
    @abc.abstractmethod
    def file_parser(cls, file: TextIOWrapper) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """Generates data, labels, and params from a given file.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWapper of the datafile.

        Returns
        -------
        tuple[NDArray, list[str], list[str], dict]
            A tuple of the data (NDArray), labels (list), units (list) and parameters (dict) of the datafile.
        """
        valid_filename = False
        for ext in cls.ALLOWED_EXTENSIONS:
            if file.name.endswith(ext):
                valid_filename = True
                break
        if not valid_filename:
            raise ValueError(f"File {file.name} is not a valid file type for {cls.__name__}.")
        
        data = None
        labels = []
        units = []
        params = {}
        return data, labels, units, params

    @abc.abstractmethod
    def to_scan(self) -> type[base_scan]:
        """Converts the parser object to a base_scan_1D object.

        Returns
        -------
        type[base_scan_1D]
            
        """
        return None

    def load(self, file: str | TextIOWrapper | None = None) -> None:
        """Loads data from the specified file, and attaches it to the object.
        Additionally rewrites filename attribute if a new file is loaded.

        Parameters
        ----------
        file : str | TextIOWrapper | None, optional
            File information can be passed as a string or a TextIOWrapper object.
            If None, then the object filename attribute is used to load the data.
            If filename is also None, then a ValueError is raised.

        Raises
        ------
        ValueError
            Raised if no file is provided and the object filename attribute is None.
        """
        # Load object filename
        load_filename = self.filename
        if type(file) is TextIOWrapper:
            # File already loaded
            self.data, self.labels, self.params = self.file_parser(file)
            # If a file is provided override filename.
            self.filename = file.name
            return
        elif type(file) is str:
            # Update filename
            load_filename = file
            
        # Try to load filename
        if load_filename is None:
            raise ValueError('No file/filename provided to load data.')
        else:
            with open(load_filename, 'r') as load_file:
                self.data, self.labels, self.units, self.params = self.file_parser(load_file)
        # Update filename after data load
        self.filename = load_filename
        return
    
    def copy(self) -> type[PARSER]:
        """ Generates a copy of the parser object."""
        newobj = type(self)(None)
        # Perform deep copy of data, labels, and params.
        newobj.filename = self.filename #str copy
        newobj.data = self.data.copy() #numpy copy
        newobj.labels = self.labels.copy() #str list copy
        newobj.units = self.units.copy() #str list copy
        for key in self.params: #dict key str - value Any copy
            value = self.params[key]
            newobj.params[key] = value if isinstance(value, (int, str, float, bool)) else value.copy()
        return newobj
            
    def filesize(self) -> int:
        """Returns the size of the file in bytes."""
        
        if self.filename is None:
            raise ValueError('No file loaded.')
        return os.path.getsize(self.filename)
    
    def memorysize(self) -> int:
        """Returns the size of the object (and it's attributes) in memory bytes."""
        return sys.getsizeof(self) + sys.getsizeof(self.data) + sys.getsizeof(self.labels) + sys.getsizeof(self.params)