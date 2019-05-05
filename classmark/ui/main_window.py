"""
Created on 19. 12. 2018
Module for main window of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from functools import partial

from .widget_manager import WidgetManager
from .home_section import HomeSection
from .experiment_section import ExperimentSection
from .section_router import SectionRouter

from PySide2.QtWidgets import QFileDialog, QMessageBox, QShortcut
from PySide2.QtGui import QKeySequence
from PySide2.QtCore import Qt




class MainWindow(WidgetManager,SectionRouter):
    """
    Main window manager of that application.
    """
    
    TEMPLATE="window"
    """Corresponding template name."""
    

    def __init__(self, app):
        """
        Initializes main window. Initial section is set to home.
        
        :param app: The QT app.
        :type app: QApplication
        """
        super().__init__()
        
        self._widget=self._loadTemplate(self.TEMPLATE)
        
        self._home=HomeSection(self, self._widget.mainContent)
        self._experiment=ExperimentSection(self, self._widget.mainContent)
        
        
        #Main content is Stacked Widget.
        self._widget.mainContent.addWidget(self._home.widget)
        self._widget.mainContent.addWidget(self._experiment.widget)
        
        #register click events
        
        self._widget.menuFileNewExperiment.triggered.connect(partial(self.goExperiment, None))
        self._widget.menuFileLoadExperiment.triggered.connect(self.goLoadExperiment)
        self._widget.menuFileSaveExperiment.triggered.connect(self.goSaveExperiment)
        self._widget.menuFileSaveAsExperiment.triggered.connect(partial(self.goSaveExperiment,True))
        
        self._widget.menuFileExit.triggered.connect(app.quit)
        
        #shortcuts
        self._saveShortcut = QShortcut(QKeySequence(Qt.CTRL + Qt.Key_S), self._widget)

        self.goHome()
        
    
    def show(self):
        """
        Show loaded widget.
        """
        self._widget.show()
        
    def goHome(self):
        """
        Go to home section.
        """
        self._widget.mainContent.setCurrentIndex(self._widget.mainContent.indexOf(self._home.widget))
        self._widget.menuFileSaveExperiment.setEnabled(False)
        self._widget.menuFileSaveAsExperiment.setEnabled(False)
        self._saveShortcut.activated.connect(None)
        
    def goExperiment(self, load=None):
        """
        Go to experiment section.
        
        :param load: Path to file containing experiment configuration.
            None means that new experiment should be loaded.
        :type load: string|None
        """
        
        self._goExperiment(load)
        
        
    def _goExperiment(self, load=None):
        """
        Go to experiment section. Actually changes sections and load the experiment.
        
        :param load: Path to file containing experiment configuration.
            None means that new experiment should be loaded.
        :type load: string|None
        """

        self._experiment.loadExperiment(load)

        self._widget.mainContent.setCurrentIndex(self._widget.mainContent.indexOf(self._experiment.widget))
        self._widget.menuFileSaveExperiment.setEnabled(True)
        self._widget.menuFileSaveAsExperiment.setEnabled(True)
        self._saveShortcut.activated.connect(self.goSaveExperiment)
        
    def goSaveExperiment(self, saveAs:bool=False):
        """
        Selection of path where experiment should be saved and saving it. If experiment
        was already saved or loaded from a path than this method automatically selects the path for saving.
        
        :param saveAs: If this parameter is true. Than path selection is forced.
        :type saveAs: bool
        """
        
        if saveAs or self._experiment.experiment.loadSavePath is None:
            file=QFileDialog.getSaveFileName(self._widget, self.tr("Save experiment"), ".e", self.tr("Any files (*)"))
            file=file[0]
        else:
            file=self._experiment.experiment.loadSavePath
        if file:
            #use selected file
            try:
                self._experiment.saveExperiment(file)
                
            except Exception as e:
                emsg = QMessageBox()
                emsg.setWindowTitle(self.tr("There is a problem with saving your experiment :("))
                emsg.setText(str(e))
                emsg.setIcon(QMessageBox.Critical)
                emsg.exec()
            
    def goLoadExperiment(self):
        """
        Selection of experiment file.
        """

        file=QFileDialog.getOpenFileName(self._widget, self.tr("Load experiment"), "~", self.tr("Any files (*)"))
        if file[0]:
            #use selected file
            try:
                self._goExperiment(file[0])
            except Exception as e:
                emsg = QMessageBox()
                emsg.setWindowTitle(self.tr("There is a problem with loading your experiment :("))
                emsg.setText(str(e))
                emsg.setIcon(QMessageBox.Critical)
                emsg.exec()

        