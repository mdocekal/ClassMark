"""
Created on 28. 2. 2019
Modules defining plugins types and interfaces.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from abc import ABC, abstractmethod
import pkg_resources
from enum import Enum
from typing import List, Callable, Any

class PluginAttribute(object):
    """
    Wrapper for plugin attribute. You could use this for auto creating of attributes widget.
    Just create member of that class and let the default implementation of getAttributesWidget.
    """
    class PluginAttributeType(Enum):
        CHECKABLE=0
        """Good for boolean attributes. In UI it appears as checkbox."""
        
        VALUE=1
        """Ordinary attribute with value. In UI appears as input."""
        
        SELECTABLE=2
        """If you have attribute when user should choose one choice from multiple choices, than
        this is the type you are looking for. In UI appears as combo box."""
        
        GROUP_VALUE=3
        """Same as value, but user could increase number of inputs."""
    
    def __init__(self, name, t, valT=None, selVals:List[str]=[]):
        """
        Creates new attribute of an plugin.
        
        :param name: Name of attribute.
        :type name: str
        :param t: Type of attribute.
        :type t: PluginAttributeType
        :param valT: Type of attribute value. Could be used for type controls and 
            auto type conversion. None means any type. Attribute value that is None or empty string
            is allways ok reregardless valT.
        :type valT: Type
        :param selVals: Values for combobox if SELECTABLE type is used.
        :type selVals: List[str]
        """
        self._name=name
        self._type=t
        self._valT=valT
        self._value=None
        self._selVals=selVals
        
    @property
    def selVals(self):
        """
        Values for combobox if SELECTABLE type is used.
        """
        return self._selVals
        
    @property
    def name(self):
        """
        Attribute name.
        """
        return self._name
    
    @property
    def type(self): 
        """
        Attribute type.
        """
        return self._type
    
    @property
    def value(self):
        """
        Attribute value.
        """
        return self._value
    
    @value.setter
    def value(self, nVal):
        """
        Set new value of attribute.
        
        :param nVal: The new value.
        :type nVal: According to provided valT in __init__.
        :raise ValueError: When the type of new value is invalid.
        """
        self.setValue(nVal)
        
    def setValue(self, nVal):
        """
        Set new value of attribute.
        
        :param nVal: The new value.
        :type nVal: According to provided valT in __init__.
        :raise ValueError: When the type of new value is invalid.
        """
        if nVal is None or nVal=="":
            self._value=None
        else:
            self._value=nVal if self._valT is None else self._valT(nVal)
        
    def setValueBind(self, bind:Callable[[str],Any]):
        """
        Same as setValue, but when ValueError exception is raised, than
        it calls bind callable with old value (as str).
        
        :param bind: Callable that should be called when exception raise.
        :type bind: Callable[[str],Any]
        """
        
        def setV(val):
            try:
                self.setValue(val)
            except ValueError:
                bind(str(self._value))
            
        return setV
        

class Plugin(ABC):
    """
    Abstract class that defines plugin and its interface.
    """

    def getAttributesWidget(self, parent=None):
        """
        UI widget for configuration of plugin attributes.
        Default implementation returns None and Widget is automatically
        created from attributes from getAttributes method.
        
        Override this method only in case when you want to create the widget
        completely be your own.
        
        :param parent: Parent for widget.
        :type parent: QWidget
        :return: Widget for settings.
        :rtype: QWidget
        """
        return None
        
        
    def getAttributes(self):
        """
        Default implementation searches members of PluginAttribute type
        and returns them.
        
        :return: Searched attributes.
        :rtype: List[PluginAttribute]
        """
        
        return [value for value in self.__dict__.values() if isinstance(value, PluginAttribute)]
    
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

