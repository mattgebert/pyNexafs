import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QTextEdit

from pyNexafs.gui.widgets import nexafsViewer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("pyNexafs")
        self.viewer = nexafsViewer()
        self.setCentralWidget(self.viewer)


def gui():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()


if __name__ == "__main__":
    gui()
