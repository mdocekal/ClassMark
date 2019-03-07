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
    def train(self, data, labels):
        """
        Train classifier on provided data.
        
        :param data: Data that will be used for training.
        :type data: ArrayLike
        :param labels: Labels for the training data.
        :type labels: ArrayLike
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        Predict label on provided data.
        
        :param data: Data for classification.
        :type data: ArrayLike
        :return: Predicted labels.
        :rtype: ArrayLike
        """
        pass
    
class FeatureExtractor(Plugin):
    """
    Abstract class that defines feature extractor type plugin and its interface.
    """
    
    @abstractmethod
    def fit(self, data, labels=None):
        """
        Prepare feature extractor with given data.
        Something like classifier training.
        
        :param data: Data for preparation.
        :type data: ArrayLike
        :param labels: Labels for preparation.
        :type labels: ArrayLike
        """
        pass
    
    @abstractmethod
    def extract(self, data):
        """
        Extract features from given data.
        
        :param data: Original data for features extraction.
        :type data: ArrayLike
        :return: Extracted features.
        :rtype: ArrayLike
        """
        pass


CLASSIFIERS={entry_point.name: entry_point.load() \
                for entry_point in pkg_resources.iter_entry_points('classmark.plugins.classifiers')}
"""All classifiers plugins."""

FEATURE_EXTRACTORS={entry_point.name: entry_point.load() \
                for entry_point in pkg_resources.iter_entry_points('classmark.plugins.features_extractors')}
"""All feature extractors plugins."""
