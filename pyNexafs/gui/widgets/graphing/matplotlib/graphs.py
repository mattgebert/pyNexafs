import sys
import numpy as np
import matplotlib as mpl
mpl.use("QtAgg")
import matplotlib.figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backend_bases import NavigationToolbar2 as NavTB
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavTBQT
from pyNexafs.gui.widgets.graphing.matplotlib.styling import StyleSelection

class FigureCanvas(FigureCanvasQTAgg):
    def __init__(self, mpl_fig: matplotlib.figure.Figure):
        super().__init__(mpl_fig)

class NEXAFS_Nav(NavTB):
    toolitems = tuple(list(NavTB.toolitems) + [
        (None, None, None, None),
        ('Normalisation', 
         'Allows use of normalisation and background subtraction options',
         'home',
         'norm_toolkit'),
    ])
    
    def __init__(self, canvas):
        NavTB.toolitems = self.toolitems
        super().__init__(canvas)
        
    def norm_toolkit():
        
        
class NEXAFS_NavQT(NavTBQT):
    def __init__(self, canvas):
        super().__init__(canvas)
    