"""
Created on 19. 12. 2018
Module for home section of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .widget_manager import WidgetManager
from .section_router import SectionRouter
from functools import partial
from .models import ListLastExperiments
from PySide2.QtCore import Qt

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
        super().__init__()
        self._widget=self._loadTemplate(self.TEMPLATE, parent)
        self._router=sectionRouter
        
        
        #register click events
        self._widget.toolNewExperiment.clicked.connect(partial(self._router.goExperiment, None))
        self._widget.toolLoadExperiment.clicked.connect(self._router.goLoadExperiment)
        #self._widget.findChild(QToolButton, 'toolNewExperiment')
        
        
        #last experiements
        self._widget.lastExpList.setModel(ListLastExperiments())
        
        self._widget.lastExpList.doubleClicked.connect(self.lastExperSel)
        
    def lastExperSel(self, index):
        """
        Last experiment was selected.
        
        :param index: Index.
        :type index: QModelIndex
        """
        self._router.goExperiment(self._widget.lastExpList.model().data(index, Qt.DisplayRole))
        