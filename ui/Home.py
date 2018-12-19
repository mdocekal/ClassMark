"""
Created on 19. 12. 2018
Module for home section of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .WidgetManager import WidgetManager

class Home(WidgetManager):
    """
    Home section manager class.
    """
    
    HOME_TEMPLATE=WidgetManager.UI_FOLDER+"/templates/home.ui"


    def __init__(self, parent=None):
        """
        Initializes home section.
        
        :param parent: Parent widget
        :type parent: QWidget
        """
        super(Home, self).__init__()
        self._widget=self._loadWidget(self.HOME_TEMPLATE, parent)
        