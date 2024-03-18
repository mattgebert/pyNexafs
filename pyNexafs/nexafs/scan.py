import numpy.typing as npt
from typing import Type, TypeVar


SCAN = TypeVar("SCAN", bound="base_scan") #typing hint for parser_base class inheritance.

class base_scan():
    """Base class for synchrotron scans, using specific photon beam energies (eV).
    Allows for multiple Y channels, reflecting various collectors that can be used in the beamline.
    """
    
    def __init__(self, x: npt.NDArray,
                 y: npt.NDArray,
                 y_errs: npt.NDArray | None = None,
                 x_errs: npt.NDArray | None = None,
                 y_labels: list[str] | None = None,
                 y_units: list[str] | None = None
                 ) -> None:
        
        self.x = x.copy()
        self.y = y.copy()
        self.x_errs = x_errs.copy() if x_errs is not None else None
        self.y_errs = y_errs.copy() if y_errs is not None else None
        self.y_labels = y_labels.copy() if y_labels is not None else None
        self.y_units = y_units.copy() if y_units is not None else None
        
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
    
    