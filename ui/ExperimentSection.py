"""
Created on 20. 12. 2018
Module for experiment section of the application.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .WidgetManager import WidgetManager
from ui.SectionRouter import SectionRouter

class ExperimentSection(WidgetManager):
    """
    Experiment section manager class.
    """

    TEMPLATE="experiment"
    """Corresponding template name."""

    def __init__(self, sectionRouter:SectionRouter, parent=None, load=None):
        """
        Initializes experiment section.
        
        :param sectionRouter: Router for in app navigation.
        :type sectionRouter: SectionRouter
        :param parent: Parent widget
        :type parent: QWidget
        :param load: Path to file containing experiment configuration.
            None means that new experiment should be loaded.
        :type load: string|None
        """
        
        super(ExperimentSection, self).__init__()
        self._widget=self._loadTemplate(self.TEMPLATE, parent)
        self._router=sectionRouter
        
    def loadExperiment(self, load):
        """
        Loads given experiment.
        
        :param load: Path to file containing experiment configuration.
        :type load: string
        """
        pass