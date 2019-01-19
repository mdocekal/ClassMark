"""
Created on 19. 12. 2018
Module for home section of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .WidgetManager import WidgetManager
from ui.SectionRouter import SectionRouter

class HomeSection(WidgetManager):
    """
    HomeSection manager class.
    """
    
    TEMPLATE="home"
    """Corresponding template name."""


    def __init__(self, sectionRouter:SectionRouter, parent=None):
        """
        Initializes home section.
        
        :param sectionRouter: Router for in app navigation.
        :type sectionRouter: SectionRouter
        :param parent: Parent widget
        :type parent: QWidget
        """
        super(HomeSection, self).__init__()
        self._widget=self._loadTemplate(self.TEMPLATE, parent)
        self._router=sectionRouter
        
        
        #register click events
        self._widget.toolNewExperiment.clicked.connect(self._router.goExperiment)
        #self._widget.findChild(QToolButton, 'toolNewExperiment')