"""
Created on 19. 1. 2018
Module of router interface that is usefull for navigation between sections..

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from abc import ABC, abstractmethod

class SectionRouter(ABC):
    """
    Interface that should implement every router, that wants to be used 
    for navigation inside this application.
    """
    
    @abstractmethod
    def goHome(self):
        """
        Go to home section.
        """
        pass
    
    @abstractmethod
    def goExperiment(self, load=None):
        """
        Go to experiment section.
        
        :param load: Path to file containing experiment configuration.
            None means that new experiment should be loaded.
        :type load: string|None
        """
        pass
    
    @abstractmethod
    def goLoadExperiment(self):
        """
        Selection of experiment file.
        """
        pass
        