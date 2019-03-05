"""
Created on 19. 12. 2018
ClassMark is a classifier benchmark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import sys

from PySide2.QtWidgets import QApplication, QMessageBox
from PySide2.QtCore import Qt
from .ui.main_window import MainWindow
import traceback

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    try:
        
        window = MainWindow(app)
        window.show()
    
        sys.exit(app.exec_())
    except Exception as e: 
        errTxt="--------------------\nError detail:\n\n"+traceback.format_exc()\
                +"--------------------\nText:\n\n"+str(e)+"\n--------------------"
        print(errTxt, file=sys.stderr)
        emsg = QMessageBox()
        emsg.setWindowTitle("Error occurred :(")
        emsg.setText(str(e))
        emsg.setDetailedText(errTxt)
        emsg.setIcon(QMessageBox.Critical)
        emsg.show()
        sys.exit(app.exec_())

        
        
    
if __name__ == '__main__':
    main()