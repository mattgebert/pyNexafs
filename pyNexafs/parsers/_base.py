import numpy as np
import pandas as pd
import abc
import matplotlib.pyplot as plt
import sys, io, os
from io import TextIOWrapper
from typing import Any, TypeVar
from numpy.typing import NDArray
from overrides import overrides


PARSER = TypeVar("PARSER", bound="parser_base") #typing hint for parser_base class inheritance.

class parser_base(metaclass=abc.ABCMeta):
    """ General class that parses raw datafiles to acquire useful data information. 
        Requires implementation of file_parser method.
    """
    
    ALLOWED_EXTENSIONS = []
    
    def __init__(self, filename: str | None) -> None:
        
        # ABC super.
        super().__init__()
        # Initialise variables
        self.filename = filename
        self.data = None
        self.labels = []
        self.params = {}
        
        if filename is None:
            return
        elif type(filename) is str:
            self.load() # Load data
        
 
        
    @staticmethod
    @abc.abstractmethod
    def file_parser(file: TextIOWrapper) -> tuple[NDArray, list[str], dict[str, Any]]:
        """Generates data, labels, and params from a given file.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWapper of the datafile.

        Returns
        -------
        tuple[NDArray, list[str], dict]
            A tuple of the data (NDArray), labels (list) and parameters (dict) of the datafile.
        """
        data = None
        labels = []
        params = {}
        return data, labels, params            

    def load(self, file: str | TextIOWrapper = None) -> None:
        load_filename = self.filename
        if type(file) is TextIOWrapper:
            # File already loaded
            self.data, self.labels, self.params = self.file_parser(file)
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
                self.data, self.labels, self.params = self.file_parser(load_file)
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
        """Returns the size of the object in memory bytes."""
        return sys.getsizeof(self)
    
    