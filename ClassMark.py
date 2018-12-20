#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Created on 19. 12. 2018
ClassMark is a classifier benchmark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import sys

from PySide2.QtWidgets import QApplication
from ui.MainWindow import MainWindow



if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = MainWindow(app)
    window.show()

    sys.exit(app.exec_())