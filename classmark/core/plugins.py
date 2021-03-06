"""
Created on 28. 2. 2019
Modules defining plugins types and interfaces.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from abc import ABC, abstractmethod

import numpy as np
import pkg_resources
from enum import Enum
from typing import List, Callable, Any, Set, Optional, Tuple
from .utils import Logger, Observable
import copy

import sys


class PluginAttributeValueChecker(object):
    """
    Just formal base class for value checker functors.
    """

    class IntermediateValue(Exception):
        """
        Signalises that value is not valid, but could be if user wil change it litle bit.
        """
        pass

    pass


class PluginAttributeStringChecker(PluginAttributeValueChecker):
    """
    Checks string values.
    """

    def __init__(self, valid: Set[str] = None, couldBeNone=False):
        """
        Initialization of checker.
        
        :param valid: Set of valid strings.
        :type valid: Set[str]
        :param couldBeNone: True means that value could be also None.
            For empty string or just passed None instead of string for conversion.
        :type couldBeNone: bool
        """
        self._valid = valid
        self._couldBeNone = couldBeNone

    def __call__(self, val: str):
        """
        Checks if val contains valid string.
        
        :param val: String value.
        :type val: str
        :return: str
        :rtype: str | None
        :raise ValueError: On invalid value. 
        """

        if self._couldBeNone and (val is None or val == ""):
            return None

        if self._valid is not None and val not in self._valid:
            if val is not None and len(val) > 0:
                if any(val in s for s in self._valid):
                    # is substring
                    raise self.IntermediateValue()

            raise ValueError("Invalid value: {}".format(val))

        return val


class PluginAttributeIntChecker(PluginAttributeValueChecker):
    """
    Checks and creates integer values from string.
    """

    def __init__(self, minV: int = None, maxV: int = None, couldBeNone=False):
        """
        Initialization of checker/creater. You can specify valid value interval.
        
        :param minV: Min value that is valid.
        :type minV: int
        :param maxV: Max valus that is valid.
        :type maxV: int
        :param couldBeNone: True means that value could be also None.
            For empty string or just passed None instead of string for conversion.
        :type couldBeNone: bool
        """
        self._min = minV
        self._max = maxV
        self._couldBeNone = couldBeNone

    def __call__(self, val: str):
        """
        Checks if val contains integer  value.
        If it is ok, than integer value is created else ValueError is raised.
        
        :param val: String value representing int.
        :type val: str
        :return: integer
        :rtype: int | None
        :raise ValueError: On invalid value. | self.IntermediateValue when value is not valid but could be.
        """
        if self._couldBeNone and (val is None or val == ""):
            return None

        if val == "-":
            # user starts typing negative value.
            res = -1
        else:
            res = int(val)

        if self._min is not None and res < self._min:
            # ok, the min does not match, but maybe user is just still typing
            raise self.IntermediateValue()

        if self._max is not None and res > self._max:
            raise ValueError("Maximal value could be {}.".format(self._max))

        return res


class PluginAttributeFloatChecker(PluginAttributeValueChecker):
    """
    Checks and creates float values from string.
    """

    def __init__(self, minV: float = None, maxV: float = None, couldBeNone=False):
        """
        Initialization of checker/creater. You can specify valid value interval.
        
        :param minV: Min value that is valid.
        :type minV: float
        :param maxV: Max valus that is valid.
        :type maxV: float
        :param couldBeNone: True means that value could be also None.
            For empty string or just passed None instead of string for conversion.
        :type couldBeNone: bool
        """
        self._min = minV
        self._max = maxV
        self._couldBeNone = couldBeNone

    def __call__(self, val: str):
        """
        Checks if val contains float  value.
        If it is ok, than float value is created else ValueError is raised.
        
        :param val: String value representing float.
        :type val: str
        :return: float
        :rtype: float | None
        :raise ValueError: On invalid value. | self.IntermediateValue when value is not valid but could be.
        """

        if self._couldBeNone and (val is None or val == ""):
            return None

        if val == "-":
            # user starts typing negative value.
            res = -1.0
        else:
            res = float(val)

        if self._min is not None and res < self._min:
            # ok, the min does not match, but maybe user is just still typing
            raise self.IntermediateValue()

        if self._max is not None and res > self._max:
            raise ValueError("Maximal value could be {}.".format(self._max))

        return res


class PluginAttribute(Observable):
    """
    Wrapper for plugin attribute. You could use this for auto creating of attributes widget.
    Just create member of that class and let the default implementation of getAttributesWidget.
    """

    class PluginAttributeType(Enum):
        CHECKABLE = 0
        """Good for boolean attributes. In UI it appears as checkbox."""

        VALUE = 1
        """Ordinary attribute with value. In UI appears as input."""

        SELECTABLE = 2
        """If you have attribute when user should choose one choice from multiple choices, then
        this is the type you are looking for. In UI appears as combo box."""

        SELECTABLE_PLUGIN = 3
        """When you have attribute that can contain one from one kind of plugins, then use that type of attribute.
        In UI appears as combo box and widget with attributes of actually selected plugin."""

        GROUP_PLUGINS = 4
        """Group of plugins. User could add or remove plugins in groups.
            Do not set values programmatically. 
            Usage: Create Plugin that will store your values and than read that values when you need.
            """

    def __init__(self, name, t, valT=None, selVals=None):
        """
        Creates new attribute of an plugin.
        
        :param name: Name of attribute.
        :type name: str
        :param t: Type of attribute.
        :type t: PluginAttributeType
        :param valT: Type of attribute value. Could be used for type controls and 
            auto type conversion. None means any type.
            
            If you use GROUP_PLUGINS, than pass to this type class of plugin that should be used.
            
            You can pass your own function that will be checking valid values
            and raises ValueError on invalid value and returns created value else.
            
            You can use one of PluginAttributeValueChecker or get inspiration from them.
        :type valT: Type | Callable[[str], Any]
        :param selVals: Values for combobox if SELECTABLE type is used.
        :type selVals: List[Any]
        """
        super().__init__()
        if selVals is None:
            selVals = []

        self._name = name
        self._type = t

        self._valT = valT
        self._value = [] if t == self.PluginAttributeType.GROUP_PLUGINS else None
        self._selVals = selVals
        self._groupItemLabel = None

    @property
    def groupItemLabel(self):
        """
        Label for item in group.
        """
        return self._groupItemLabel

    @groupItemLabel.setter
    def groupItemLabel(self, v):
        """
        Label for item in group.
        If you want to add position of item to label, when used with AttributesWidgetManager (default), than use {}.
        Example: Layer {}    ->     Layer 1
                                    ...
                                    Layer x                
        :param v: New label.
        :type v: str
        """
        self._groupItemLabel = v

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
    def valType(self):
        """
        Attribute value type.
        """
        return self._valT

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

    @Observable._event("VALUE_CHANGED", True)
    def setValue(self, nVal, index=None):
        """
        Set new value of attribute.
        
        :param nVal: The new value.
        :type nVal: According to provided valT in __init__.
        :param index: This parameter is for GROUP attributes and
            defines index in list where the value should be set.
        :type index: int
        :raise ValueError: When the type of new value is invalid. | self.IntermediateValue when value is not valid but could be.
        """

        if index is None:
            if self.type == PluginAttribute.PluginAttributeType.GROUP_PLUGINS:
                self._value = [x if x is None else self._valT(x) for x in nVal]
            else:
                self._value = nVal if self._valT is None else self._valT(nVal)
        else:
            self._value[index] = nVal if self._valT is None else self._valT(nVal)

    def setValueBind(self, bind: Callable[[str], Any], index=None):
        """
        Creates setter that is same as setValue, but when ValueError exception is raised, than
        it calls bind callable with old value (as str).
        
        :param bind: Callable that should be called when exception raise.
        :type bind: Callable[[str],Any]
        :param index:This parameter is for GROUP attributes and
            defines index in list where the value should be set.
        :type index: int
        :return:  Setter that sets value and handles exception
        :rtype: Callable[[Any],Any]
        """

        def setV(val):
            try:
                self.setValue(val, index)
            except ValueError:
                if index is None:
                    bind("" if self._value is None else str(self._value))
                else:
                    bind("" if self._value[index] is None else str(self._value[index]))
            except PluginAttributeValueChecker.IntermediateValue:
                # probably just intermediate value
                pass

        return setV


class PluginStub(object):
    """
    Plugin stub is used as a copy of Plugin that only consists of descriptive informations such as
    marking, name and plugina attributes.
    """

    def __init__(self, plugin):
        """
        Initialize stub.
        
        :param plugin: Plugin that stub you want.
        :type plugin: Plugin
        """

        self._attributes = copy.deepcopy(plugin.getAttributes())
        self._name = plugin.getName()
        self._nameAbber = plugin.getNameAbbreviation()
        self._marking = plugin.marking

    @property
    def marking(self):
        """
        Marking of original Plugin.
        """
        return self._marking

    def getAttributes(self):
        """
        Attributes of original plugin.
        
        :return: Searched attributes.
        :rtype: List[PluginAttribute]
        """

        return self._attributes

    def getName(self):
        """
        Name of original Plugin.
        
        :return: Name of the plugin.
        :rtype: str
        """
        return self._name

    def getNameAbbreviation(self):
        """
        Plugin name abbreviation.
        
        :return: Name abbreviation of the plugin.
        :rtype: str
        """
        return self._nameAbber

    def __repr__(self):
        """
        String representation in consisting of names and atributes of plugin.
        """

        return self.getName() + ", ".join(a.name + "=" + str(a.value.getName() + ":" + ", ".join(
            [pa.name + "->" + str(pa.value) for pa in a.value.getAttributes()]) if isinstance(a.value,
                                                                                              Plugin) else a.value) for
                                          a in self.getAttributes())

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self._marking == other._marking
        return False

    def __hash__(self):
        return self._marking

    def stub(self):
        """
        This is already a stub.
        """
        return self


class Plugin(ABC):
    """
    Abstract class that defines plugin and its interface.
    """

    _logger = Logger()
    """If you want to log some information that should be shown in GUI than use this
    logger. (logger.log("message"))"""

    __MARKING_CNT = 0
    """This is used as unique identifier/marking of plugin."""

    def __new__(cls, *args, **kwargs):
        """
        Just add marking.
        """
        if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
            # In Python 3.3 and later there could not be any arguments.
            instance = super().__new__(cls)
        else:
            instance = super().__new__(cls, *args, **kwargs)
        instance._marking = Plugin.__MARKING_CNT
        Plugin.__MARKING_CNT += 1

        return instance

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

    def __repr__(self):
        """
        String representation in consisting of names and atributes of plugin.
        """

        return self.getName() + ", ".join(a.name + "=" + str(a.value.getName() + ":" + ", ".join(
            [pa.name + "->" + str(pa.value) for pa in a.value.getAttributes()]) if isinstance(a.value,
                                                                                              Plugin) else a.value) for
                                          a in self.getAttributes())

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self._marking == other._marking
        return False

    def __hash__(self):
        return self._marking

    @property
    def marking(self):
        """
        Identifier of plugin.
        May not be unique if copy of plugin is made.
        """
        return self._marking

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

    def stub(self):
        """
        Make stub of plugin. The stub has copy of marking, name and attributes.
        """
        return PluginStub(self)


class Classifier(Plugin):
    """
    Abstract class that defines classifier type plugin and its interface.
    """

    @abstractmethod
    def train(self, data, labels):
        """
        Train classifier on provided data.
        
        :param data: Data that will be used for training.
        :type data: scipy.sparse matrix
        :param labels: Labels for the training data.
            Classifier will always receive encoded labels to integers (0...NUMBER_OF_CLASSES-1).
        :type labels: np.array
        """
        pass

    @abstractmethod
    def classify(self, data):
        """
        Classify label on provided data.
        
        :param data: Data for classification.
        :type data: scipy.sparse matrix
        :return: Predicted labels.
        :rtype: ArrayLike
        """
        pass

    def classifyShowTopFeatures(self, data, featuresNames: np.array):
        """
        Classify label on provided data and provides top important features that were used for decision.

        Number of most important features determines each classifier itself.
        Consider adding a user editable attribute for it.

        :param data: Data for classification.
        :type data: scipy.sparse matrix
        :param featuresNames: Name for each feature in an input data vector that was passed to the model during training.
        :type featuresNames: np.array
        :return: Predicted labels, array of features names with array of importance scores. Both arrays
        (names, importance) are in descending order according to importance
        :rtype: Tuple[ArrayLike, ArrayLike, ArrayLike]
        """
        raise NotImplementedError()

    def topFeatures(self, featuresNames: np.array) -> List[Tuple[str, float]]:
        """
        Returns importance of features as a float number. Bigger the number is, the more important the feature
        should be for decision making.

        Number of most important features determines each classifier itself.
        Consider adding a user editable attribute for it.

        This method should be called after training.

        :param featuresNames: Name for each feature [string] in an input data vector that was passed to the model during training.
        :type featuresNames: np.array
        :return:  Features names and their importance. Should be order from the most important feature.
        :rtype:  np.array, np.array
        """

        raise NotImplementedError()

    def featureImportanceShouldBeShown(self) -> bool:
        """
        True if classifier is able to provide importance score for a feature and user asked for it.
        False otherwise.
        """
        return False


class FeatureExtractor(Plugin):
    """
    Abstract class that defines feature extractor type plugin and its interface.
    """

    class DataTypes(Enum):
        """
        Data types that should be passed to extract methods.
        All this types of data will be wrapped in numpy array.
        """

        STRING = 0
        """default"""
        IMAGE = 1
        """LazyImageFileReader
        If you want np.array, call getRGB."""

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
            In numpy array:
                example: ["first document", "second document"]
        :type data: ArrayLike
        :return: Extracted features.
            Example:
                [
                [1,1,0],    first document features
                [0,1,1]    second document features
                ]
        :rtype: scipy.sparse matrix
        """
        pass

    @abstractmethod
    def fitAndExtract(self, data, labels=None):
        """
        Should be the same as calling fit and than extract on the same data.
        
        :param data:Original data for features extraction.
            In numpy array:
                example: ["first document", "second document"]
        :type data: np.array
        :param labels: Labels for preparation.
        :type labels: ArrayLike
        :return: Extracted features.
            Example:
                [
                [1,1,0],    first document features
                [0,1,1]    second document features
                ]
        :rtype: scipy.sparse matrix
        """
        pass

    @classmethod
    def expDataType(cls):
        """
        Expected data type for extraction.
        Overwrite this method if you want to use different data type.
        Beware that this is just type hint for reading external files that haves their path
        in attributes marked as path attribute.
        """
        return cls.DataTypes.STRING

    def featuresNames(self) -> Optional[List[str]]:
        """
        List of features names. The order of features names must correspond to the order of features value in feature
        vector.

        :return: features names or None when our extractor is not using naming
        :rtype: Optional[List[str]]
        """
        return None


CLASSIFIERS = {entry_point.name: entry_point.load() \
               for entry_point in pkg_resources.iter_entry_points('classmark.plugins.classifiers')}
"""All classifiers plugins."""

FEATURE_EXTRACTORS = {entry_point.name: entry_point.load() \
                      for entry_point in pkg_resources.iter_entry_points('classmark.plugins.features_extractors')}
"""All feature extractors plugins."""
