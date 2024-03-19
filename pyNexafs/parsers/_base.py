import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import abc
import sys, io, os
from io import TextIOWrapper
from typing import Any, TypeVar, Type
from numpy.typing import NDArray
from ..nexafs.scan import base_scan

PARSER = TypeVar("PARSER", bound="parser_base") #typing hint for parser_base class inheritance.

class parser_base(metaclass=abc.ABCMeta):
    """ 
    General class that parses raw datafiles to acquire useful data information. 
    Requires implementation of file_parser method,
    

    """
    
    @property
    @abc.abstractmethod
    @staticmethod
    def ALLOWED_EXTENSIONS() -> list[str]:
        """
        Allowable extensions for the parser.
        'parser_base.file_parser' will check validity of file extensions.
        """
        return []
    
    @property
    @abc.abstractmethod
    @staticmethod
    def COLUMN_ASSIGNMENTS() -> dict[str, str | list[str] | None]:
        """
        Assignments of scan input variables to column names.
        'parser_base.to_scan' will use construct the scan parameters.
        
        Assignments can be a single column name, or a list of column names.
        y_errs and x_errs can be None if not present in the data.
        """
        
        return {
            "x" :            "Data_Column_1_Label",
            "y" :           ["Data_Column_2_Label",
                             "Data_Column_3_Label",
                             "Data_Column_4_Label"], # or "Data_Column_2_Label"
            "y_errs" :      ["Data_Column_5_Label",
                             "Data_Column_6_Label",
                             "Data_Column_7_Label"], # or "Data_Column_5_Label" or None
            "x_errs" :       None # or "Data_Column_8_Label"
        }
    
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

    def _validated_assignments(self) -> dict[str, str | list[str] | None]:
        """
        Returns the validated column assignments for the parser.
        See 'parser_base.COLUMN_ASSIGNMENTS' for more information.

        Returns
        -------
        dict[str, str | list[str] | None]
            See 'parser_base.COLUMN_ASSIGNMENTS' for more information.

        Raises
        ------
        ValueError
            'x' assignment is not a string.
        ValueError
            'y' assignment is not a list or string.
        ValueError
            'y' assignment is a list, but subelement is not a string.
        ValueError
            'y_errs' assignment is not a list, string, or None.
        ValueError
            'y_errs' assignment is a list, but subelement is not a string.
        ValueError
            'y_errs' assignment does not match length of 'y' assignment.
        ValueError
            'x_errs' assignment is not a string or None.
        """
        
        assignments = self.COLUMN_ASSIGNMENTS
        if not isinstance(assignments["x"], str):
            raise ValueError(f"X assignment {assignments['x']} is not a string.")
        if not isinstance(assignments["y"], (list, str)):
            raise ValueError(f"Y assignment {assignments['y']} is not a list or string.")
        if isinstance(assignments["y"], list):
            for y in assignments["y"]:
                if not isinstance(y, str):
                    raise ValueError(f"Y list assignment {y} is not a string.")
        if not isinstance(assignments["y_errs"], (list, str, type(None))):
            raise ValueError(f"Y errors assignment {assignments['y_errs']} is not a list, string, or None.")
        if isinstance(assignments["y_errs"], list):
            if len(assignments["y_errs"]) != len(assignments["y"]):
                raise ValueError(f"Y errors list assignment {assignments['y_errs']}\n does not match length of Y list assignment {assignments['y']}.")
            for y in assignments["y_errs"]:
                if not isinstance(y, str):
                    raise ValueError(f"Y errors list assignment {y} is not a string.")
        if not isinstance(assignments["x_errs"], (str, type(None))):
            raise ValueError(f"X errors assignment {assignments['x_errs']} is not a string or None.")
        return assignments

    def to_scan(self) -> Type[base_scan] | base_scan:
        """
        Converts the parser object to a base_scan object.

        Returns
        -------
        type[base_scan]
            Returns a base_scan object.
        """
        scan_obj = base_scan(parser=self)
        return scan_obj
        
    @classmethod
    @abc.abstractmethod
    def file_parser(cls, file: TextIOWrapper) -> tuple[NDArray, list[str], list[str], dict[str, Any]]:
        """
        Abstract method that generates data, labels, and params from a given file.
        Overridden method should be initially called to check for valid filetype (super().file_parser(file)).

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWapper of the datafile (i.e. file=open('file.csv', 'r'))

        Returns
        -------
        tuple[NDArray, 
            list[str] | None,
            list[str] | None,
            dict]
            A tuple of the data (NDArray), labels (list), units (list) and parameters (dict) of the datafile.
        """
        valid_filepath = False
        for ext in cls.ALLOWED_EXTENSIONS:
            if file.name.endswith(ext):
                valid_filepath = True
                break
        if not valid_filepath:
            raise ValueError(f"File {file.name} is not a valid file type for {cls.__name__}.")
        
        data = None
        labels = []
        units = []
        params = {}
        return data, labels, units, params

    def load(self, file: str | TextIOWrapper | None = None) -> None:
        """
        Loads data from the specified file, and attaches it to the object.
        Additionally rewrites filepath attribute if a new file is loaded.

        Parameters
        ----------
        file : str | TextIOWrapper | None, optional
            File information can be passed as a string or a TextIOWrapper object.
            If None, then the object filepath attribute is used to load the data.
            If filepath is also None, then a ValueError is raised.

        Raises
        ------
        ValueError
            Raised if no file is provided and the object filepath attribute is None.
        """
        # Load object filepath
        load_filepath = self.filepath # Might be None
        
        # Check if file parameter is provided:
        if type(file) is TextIOWrapper:
            # File already loaded
            self.data, self.labels, self.params = self.file_parser(file)
            # If a file is provided override filepath.
            self.filepath = file.name
            return
        elif type(file) is str:
            # Update filepath
            load_filepath = file
            
        # Try to load filepath
        if load_filepath is None:
            raise ValueError('No file/filepath provided to load data.')
        else:
            with open(load_filepath, 'r') as load_file:
                data, labels, units, params = self.file_parser(load_file)
        
        # Pull column length of data to compare to units and labels.
        col_len = data.shape[1]
        if labels is not None and len(labels) != col_len:
            raise ValueError(f'Labels length {len(labels)} does not match data columns {col_len}.')
        if units is not None and len(units) != col_len:
            raise ValueError(f'Units length {len(units)} does not match data columns {col_len}.')
        
        # Assign data, labels, units, and params to object.
        self.data, self.labels, self.units, self.params = data, labels, units, params
        
        # Update filepath after data load
        self.filepath = load_filepath
        return
    
    def copy(self) -> type[PARSER]:
        """
        Generates a copy of the parser object.

        Returns
        -------
        type[PARSER]
            A copy of the parser object.
        """    
        
        newobj = type(self)(None)
        # Perform deep copy of data, labels, and params.
        newobj.filepath = self.filepath #str copy
        newobj.data = self.data.copy() #numpy copy
        newobj.labels = self.labels.copy() #str list copy
        newobj.units = self.units.copy() #str list copy
        for key in self.params: #dict key str - value Any copy
            value = self.params[key]
            newobj.params[key] = value if isinstance(value, (int, str, float, bool)) else value.copy()
        return newobj
            
    def filesize(self) -> int:
        """Returns the size of the file in bytes."""
        
        if self.filepath is None:
            raise ValueError('No file loaded.')
        return os.path.getsize(self.filepath)
    
    def memorysize(self) -> int:
        """Returns the size of the object (and it's attributes) in memory bytes."""
        return sys.getsizeof(self) + sys.getsizeof(self.data) + sys.getsizeof(self.labels) + sys.getsizeof(self.params)
    
    def to_DataFrame(self) -> pd.DataFrame:
        """
        Converts the data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame of the data and labels.
        """
        return pd.DataFrame(self.data, columns=self.labels)

    def search_label_index(self, search: str) -> int:
        """
        Returns the index of the label in the labels list.
        
        Parameters
        ----------
        search : str
            String label to search for in the labels list.

        Returns
        -------
        int
            Index of the label in the labels list.

        Raises
        ------
        AttributeError
            Raised if the label is not found in the labels list.
        """
        for i in range(len(self.labels)):
            if self.labels[i] == search:
                return i
        
        raise AttributeError(f"Label {search} not found in labels.")
        
