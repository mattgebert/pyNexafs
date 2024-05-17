import sys
import numpy as np
import matplotlib as mpl
mpl.use("QtAgg")
import matplotlib.figure
from PyQt6 import QtGui, QtWidgets
import PIL.ImageQt as PILImageQt
import PIL.Image as PILImage

class StyleSelection(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)
        # self.style_selector = QtWidgets.QComboBox()
        cmaps = list(matplotlib.colormaps)
        a = 0
        b = 0
        m = 0
        for n, c in enumerate(cmaps):
            cmap = mpl.colormaps[c]
            # # Draw a colour maps for each item
            if hasattr(cmap, "colors"):
                np_cmap = np.array(cmap.colors) # 0 to 1
                # Reshape colours to fit 126 pixels regardless of length.
                threshold = 128
                if np_cmap.shape[0] < threshold:
                    # Duplicate to extend
                    np_cmap = np.repeat(np_cmap, np.ceil(threshold/np_cmap.shape[0]), axis=0)
                
                # Perform a second resizing, as repeating may make array larger than desired size.
                if np_cmap.shape[0] > threshold:
                    # Interpolate to shrink
                    np_cmap_temp = np.zeros((128,3))
                    for i in range(3):
                        np_cmap_temp[:,i] = np.interp(
                            np.linspace(0,127,128),
                            np.linspace(0,np_cmap.shape[0]-1,np_cmap.shape[0]),
                            np_cmap[:,i]
                        )
                    # Reassign cmap
                    np_cmap = np_cmap_temp
                
                # Generate Qt Pixmap
                shape = (20, np_cmap.shape[0], 4)
                data = np.ones(shape) * 255
                data[:,:,0:3] = np_cmap[np.newaxis,:,:] * 255
                pil_img = PILImage.fromarray(np.uint8(data), mode="RGBA")
                pil_imgqt = PILImageQt.ImageQt(pil_img)
                qpix = QtGui.QPixmap.fromImage(pil_imgqt)
                
                ## Add pixmap and text to grid.
                # Get grid position
                a,b = int((n-m) / 4), int((n-m) % 4)
                
                # Setup button and vbox
                button = QtWidgets.QPushButton()
                vbox = QtWidgets.QVBoxLayout()
                button.setLayout(vbox)
                # Setup elements
                lbl = QtWidgets.QLabel()
                lbl.setText(c)
                pic = QtWidgets.QLabel()
                pic.resize(qpix.width(), qpix.height())
                pic.setPixmap(qpix)
                # Add elements to vbox
                vbox.addWidget(lbl)
                vbox.addWidget(pic)
                # Set scaling and expanding policy on elements:
                for elem in [pic, button]:
                    elem.setSizePolicy(
                        QtWidgets.QSizePolicy.Policy.Expanding,
                        QtWidgets.QSizePolicy.Policy.Expanding
                    )
                vbox.setStretch(0,0)
                vbox.setStretch(1,1)
                vbox_margins = vbox.getContentsMargins()
                button.setMinimumWidth(128 + vbox_margins[0] + vbox_margins[2])
                button.setMinimumHeight(qpix.height() + 20 + vbox_margins[1] + vbox_margins[3])
                # Add button to grid.
                layout.addWidget(button,a,b)
            else:
                m+=1
        

# Demo of QT app
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = StyleSelection()
    window.setWindowTitle("pyNexafs Style Selector")
    window.show()
    sys.exit(app.exec())