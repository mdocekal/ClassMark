"""
Created on 19. 12. 2018
ClassMark is a classifier benchmark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import sys

from PySide2.QtWidgets import QApplication
from PySide2.QtCore import Qt
from .ui.main_window import MainWindow


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)

    window = MainWindow(app)
    window.show()

    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()