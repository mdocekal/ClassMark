"""
Created on 19. 12. 2018
Module for main window of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""


from .WidgetManager import WidgetManager
from .HomeSection import HomeSection
from .ExperimentSection import ExperimentSection
from .SectionRouter import SectionRouter

class MainWindow(WidgetManager, SectionRouter):
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
        super(MainWindow, self).__init__()
        
        self._widget=self._loadTemplate(self.TEMPLATE)
        
        self._home=HomeSection(self, self._widget.mainContent)
        self._experiment=ExperimentSection(self, self._widget.mainContent)

        
        #Main content is Stacked Widget.
        self._widget.mainContent.addWidget(self._home.widget)
        self._widget.mainContent.addWidget(self._experiment.widget)
        
        #register click events
        self._widget.menuFileNewExperiment.triggered.connect(self.goExperiment)
        self._widget.menuFileLoadExperiment.triggered.connect(self.goLoadExperiment)
        self._widget.menuFileExit.triggered.connect(app.quit)
        
        
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
    
    def goExperiment(self, load=None):
        """
        Go to experiment section.
        
        :param load: Path to file containing experiment configuration.
            None means that new experiment should be loaded.
        :type load: string|None
        """
        if load is not None:
            self._experiment.loadExperiment(load)
            
        self._widget.mainContent.setCurrentIndex(self._widget.mainContent.indexOf(self._experiment.widget))
        
    def goLoadExperiment(self):
        """
        Selection of experiment file.
        """
        pass
        