"""
Created on 19. 12. 2018
Module for main window of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""


from .WidgetManager import WidgetManager
from .Home import Home

class MainWindow(WidgetManager):
    """
    Main window manager of that application.
    """
    
    WINDOW_TEMPLATE=WidgetManager.UI_FOLDER+"/templates/window.ui"
    HOME_PAGE_INDEX=0


    def __init__(self):
        """
        Initializes main window. Initial section is set to home.
        """
        super(MainWindow, self).__init__()
        
        self._widget=self._loadWidget(self.WINDOW_TEMPLATE)
        
        self._home=Home(self._widget.mainContent)
        #Main content is Stacked Widget.
        #Home must be first, or change goHome method.
        self._widget.mainContent.addWidget(self._home.widget)
        
        
        #self._setTemplateAsLayout(self._template)
    
    def show(self):
        """
        Show loaded widget.
        """
        self._widget.show()
        
    def goHome(self):
        """
        Go to home section.
        """
        self._widget.mainContent.setCurrentIndex(self.HOME_PAGE_INDEX)
    
    def goExperiment(self):
        """
        Go to experiment section.
        """
        pass
        
        
        
    
        