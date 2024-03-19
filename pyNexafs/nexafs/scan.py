import numpy.typing as npt
from typing import Type, TypeVar
from ..parsers import parser_base

SCAN = TypeVar("SCAN", bound="base_scan") #typing hint for parser_base class inheritance.

class base_scan():
    """Base class for synchrotron scans, using specific photon beam energies (eV).
    Allows for multiple Y channels, reflecting various collectors that can be used in the beamline.
    Scan links to parser object, which contains the raw data and metadata.
    """
    
    def __init__(self, parser: Type[parser_base] | parser_base) -> None:
        """
        Load the parser data into the scan object for manipulation.

        Parameters
        ----------
        parser : Type[parser_base] | parser_base
            A parser object that contains the raw data and metadata.
            The COLUMN_ASSIGNMENTS property must be set to the correct column assignments.
        """
        # Create link to parser object
        self.parser = parser
        
        # Load data from parser
        self._load_from_parser()
        
        return
    
    def _load_from_parser(self) -> None:
        """Load data from parser object."""
        
        ### Check that assignments are valid for base scan.
        assignments = self.parser._validated_assignments()
        
        ### Load data from parser
        # X data - locate column
        x_index = self.parser.search_label_index(assignments["x"])
        # Y data - locate columns
        y_labels = assignments["y"]
        if isinstance(y_labels, list):
            y_indcies = [self.parser.search_label_index(label) for label in assignments["y"]]
        else: #Singular index.
            y_indcies = [self.parser.search_label_index(assignments["y"])]
        # Y errors - locate columns    
        y_errs_labels = assignments["y_errs"]
        if isinstance(y_errs_labels, list):
            y_errs_indcies = [self.parser.search_label_index(label) 
                              for label in assignments["y_errs"]
                              if label is not None else None]
        elif isinstance(y_errs_labels, str):
            y_errs_indcies = [self.parser.search_label_index(y_errs_labels)]
        else: # y_errs_labels is None:
            y_errs_indcies = None
        # X errors - locate column
        x_errs_index = self.parser.search_label_index(assignments["x_errs"]
                        ) if assignments is not None else None
        
        ### Generate data clones.
        # Data-points
        self._x = self.parser.data[:,x_index].copy()
        self._y = self.parser.data[:,y_indcies].copy()
        self._y_errs = self.parser.data[:,y_errs_indcies].copy() if y_errs_indcies is not None else None
        self._x_errs = self.parser.data[:,x_errs_index].copy() if x_errs_index is not None else None
        # Labels and Units
        self._x_label = self.parser.labels[x_index] if self.parser.labels is not None else None
        self._x_unit = self.parser.units[x_index] if self.parser.units is not None else None
        self._y_labels = [self.parser.labels[i] for i in y_indcies] if self.parser.labels is not None else None
        self._y_units = [self.parser.units[i] for i in y_indcies] if self.parser.units is not None else None
        return
    
    def copy(self) -> SCAN:
        """_summary_

        Returns
        -------
        SCAN
            _description_
        """
        newobj = type(self)(self.x, 
                            self.y, 
                            self.y_errs, 
                            self.x_errs, 
                            self.y_labels,
                            self.y_units)
        return newobj

