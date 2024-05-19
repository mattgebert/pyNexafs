import sys
import typing
import numpy as np
import matplotlib as mpl
mpl.use("QtAgg")
import matplotlib.figure
import matplotlib.colors
from PyQt6 import QtGui, QtWidgets, QtCore
import PIL.ImageQt as PILImageQt
import PIL.Image as PILImage
import overrides

class QColormapPushButton(QtWidgets.QPushButton):
    def __init__(self, cmap, min_h = 5, min_w=128, parent=None):
        super().__init__(parent=parent)
        self._cmap = None
        self.cmap = cmap
        self._min_h = min_h
        self._min_w = min_w
    
    @overrides.override
    def resizeEvent(self, event):
        if callable(self._resizeFn):
            self._resizeFn()
    
    def _resizeFn(self):
        if self.cmap is not None:
            # Generate new icon width.
            w = self.width()
            qpix = self.cmap_to_qpix(
                cmap=self.cmap,
                pix_height=self._min_h,
                pix_width=self._min_w if w < self._min_w else w
            )
            icon = QtGui.QIcon(qpix)
            self.setIcon(icon)
            # Sizing requirements
            button_min_height = qpix.height() + 10
            button_min_width = qpix.width() + 10
            self.setMinimumHeight(button_min_height)
            self.setMinimumWidth(button_min_width)
        
    @property
    def cmap(self) -> matplotlib.colors.Colormap:
        return self._cmap
    
    @cmap.setter
    def cmap(self, val: matplotlib.colors.Colormap | str) -> None:
        if isinstance(val, str):
            if val not in matplotlib.colormaps:
                raise 
            else:
                self._cmap = val
        elif isinstance(val, matplotlib.colors.Colormap):
            self._cmap = val
        else:
            raise TypeError(f"{val} is not a registered colormap string or object.")
        
    @staticmethod
    def cmap_to_qpix(
        cmap: matplotlib.colors.Colormap | str,
        pix_height:int = 20,
        pix_width:int = 128) -> QtGui.QPixmap:
        """
        Converts a matplotlib.colormap to a QtGui.QPixmap, for easy use in graphic elements.
        
        When the colormap has less elements than pix_width, the colormap is repeated to extend 
        unique colours. When the colormap has more elements than pix_width (including from 
        the result of repeating) uses interpolation to generate intermediate colours.

        Parameters
        ----------
        cmap : matplotlib.colors.Colormap
            A mapltolib colormap object, such as matplotlib.colormaps['magma']    
        pix_height : int, optional
            The desired height of the QPixmap, by default 20 (pixels)
        pix_width : int, optional
            The desired width of the QPixmap, by default 128 (pixels)

        Returns
        -------
        QtGui.QPixmap
            A Qt pixelmap object, usable for graphic displays in GUIs.
        """
        print(cmap.name)
        # Get colours.
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        elif isinstance(cmap, matplotlib.colors.Colormap):
            pass
        else:
            raise TypeError(f"{cmap} is not a matplotlib Colormap or registered colormap name.")
            
        clrs = cmap(np.linspace(0, 1, pix_width))
        # Generate Qt Pixmap
        shape = (pix_height, *clrs.shape)
        data = np.zeros(shape)
        data[:] = clrs[np.newaxis] * 255
        pil_img = PILImage.fromarray(np.uint8(data), mode="RGBA")
        pil_imgqt = PILImageQt.ImageQt(pil_img)
        qpix = QtGui.QPixmap.fromImage(pil_imgqt)
        return qpix

class StyleSelector(QtWidgets.QWidget):
    
    def __init__(self, parent=None, cm_width=128, cm_height=20):
        super().__init__(parent)
        
        # Adjustable scroll
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scrollContent = QtWidgets.QWidget(scroll)
        scrollLayout = QtWidgets.QVBoxLayout()
        scrollLayout.addWidget(scroll)
        self.setLayout(scrollLayout)
        
        layout = QtWidgets.QGridLayout()
        scrollContent.setLayout(layout)
        scroll.setWidget(scrollContent)
        headers = ["Density", "Colourmap", "Bar"]
        for col, header in enumerate(headers):
            layout.addWidget(QtWidgets.QLabel(header + ":"), 0, col)
        layout.setColumnStretch(2,1)
        
        # self.style_selector = QtWidgets.QComboBox()
        cmaps = list(matplotlib.colormaps)
        for n, c in enumerate(cmaps):
            cmap = mpl.colormaps[c]
            # Setup button
            button = QColormapPushButton(cmap, cm_width, cm_height)
            
            # Setup elements
            label_name = QtWidgets.QLabel()
            label_name.setText(c)
            label_density = QtWidgets.QLabel()
            label_density.setText(str(cmap.N))
            
            # Set scaling and expanding policy on elements:
            for elem in [button]:
                elem.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Preferred,
                    QtWidgets.QSizePolicy.Policy.Expanding
                )
                
            layout.addWidget(label_density, n+1, 0)
            layout.addWidget(label_name, n+1, 1)
            layout.addWidget(button,n+1,2)

# Demo of QT app
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = StyleSelector()
    window.setWindowTitle("pyNexafs Style Selector")
    window.show()
    sys.exit(app.exec())