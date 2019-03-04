"""
Created on 28. 2. 2019
Modules defining plugins types and interfaces.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from abc import ABC, abstractmethod
import pkg_resources

class Plugin(ABC):
    """
    Abstract class that defines plugin and its interface.
    """

    @staticmethod
    @abstractmethod
    def getAttributes():
        """
        Configurable attributes of plugin.
        """
        pass
    
    @staticmethod
    @abstractmethod
    def getName():
        """
        Plugin name.
        
        :return: Name of the plugin.
        :rtype: str
        """
        pass
    
    @staticmethod
    @abstractmethod
    def getNameAbbreviation():
        """
        Plugin name abbreviation.
        
        :return: Name abbreviation of the plugin.
        :rtype: str
        """
        pass
    
    @staticmethod
    @abstractmethod
    def getInfo():
        """
        Informations about this plugin.
        
        :return: Text description of this plugin.
        :rtype: str
        """
        pass
    
class Classifier(Plugin):
    """
    Abstract class that defines classifier type plugin and its interface.
    """
    
    @abstractmethod
    def train(self):
        """
        Train classifier on provided data.
        """
        pass
    
    @abstractmethod
    def predict(self):
        """
        Predict label on provided data.
        """
        pass
    
class FeatureExtractor(Plugin):
    """
    Abstract class that defines feature extractor type plugin and its interface.
    """
    
    @abstractmethod
    def fit(self):
        """
        Prepare feature extractor with given data.
        """
        pass
    
    @abstractmethod
    def extract(self):
        """
        Extract features from given data.
        """
        pass


CLASSIFIERS={entry_point.name: entry_point.load() \
                for entry_point in pkg_resources.iter_entry_points('classmark.plugins.classifiers')}
"""All classifiers plugins."""

FEATURE_EXTRACTORS={entry_point.name: entry_point.load() \
                for entry_point in pkg_resources.iter_entry_points('classmark.plugins.features_extractors')}
"""All feature extractors plugins."""
