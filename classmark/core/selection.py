"""
Created on 14. 4. 2019
Module for feature selection methods.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
import numpy

from .plugins import Plugin, PluginAttribute, PluginAttributeStringChecker, PluginAttributeFloatChecker

from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.tree import DecisionTreeClassifier

from abc import abstractmethod

class FeaturesSelector(Plugin):
    """
    Base class for all feature selectors.
    """
    
    @abstractmethod
    def fit(self, features, labels=None):
        """
        Prepare feature selector.
        
        :param features: Features for preparation.
        :type features: ArrayLike
        :param labels: Labels for preparation.
        :type labels: ArrayLike
        """
        pass
    
    @abstractmethod
    def select(self, features):
        """
        Selects features.
        
        :param features: Features for selection.
        :type features: scipy.sparse matrix
        :return: Selected features.
        :rtype: scipy.sparse matrix
        """
        pass

    @abstractmethod
    def getSupport(self, indices: bool = False) -> numpy.array:
        """
        Get a mask, or integer index, of the features selected

        :param indices: If True, the return value will be an array of integers, rather than a boolean mask.
        :type indices: bool
        :return: An index that selects the retained features from a feature vector.
        If indices is False, this is a boolean array of shape [# input features], in which an element is
        True iff its corresponding feature is selected for retention.
        If indices is True, this is an integer array of shape [# output features] whose values are indices into the
        input feature vector.
        :rtype: np.array
        """
        pass
    
    
class VarianceSelector(FeaturesSelector):
    """
    Selects features according to variance threshold.
    """

    def __init__(self, threshold:float=0):
        """
        Initialize VarianceSelector.
        
        :param threshold: Variance threshold.
        :type threshold: float
        """
        
        self._threshold=PluginAttribute("Threshold", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeFloatChecker(minV=0))
        self._threshold.value=threshold
        
        self._sel=None
        
    def fit(self, features, labels=None):
        self._sel = VarianceThreshold(threshold=self._threshold.value)
        self._sel.fit(features)
        
    def select(self, features):
        return self._sel.transform(features)
    
    @staticmethod
    def getName():
        return "Variance Selector"
 
    @staticmethod
    def getNameAbbreviation():
        return "VS"
    
    @staticmethod
    def getInfo():
        return ""

    def getSupport(self, indices: bool = False):
        return self._sel.get_support(indices)

    
class TreeBasedFeatureImportanceSelector(FeaturesSelector):
    """
    Selects features according to tree based feature importance.
    """
    
    def __init__(self, threshold: str = None):
        """
        Initialize TreeBasedFeatureImportanceSelector.
        
        :param threshold: Importance threshold.
        :type threshold: str
        """

        self._threshold=PluginAttribute("Threshold", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeStringChecker(couldBeNone=True))
        self._threshold.value=threshold
        
        self._sel=None
    
    def fit(self, features, labels=None):
        clf = DecisionTreeClassifier()
        clf.fit(features, labels)
        
        try:
            thres=float(self._threshold.value)
        except:
            thres=self._threshold.value
        
        self._sel = SelectFromModel(clf, prefit=True, threshold=thres)
        
    def select(self, features):
        return self._sel.transform(features)
    
    @staticmethod
    def getName():
        return "Tree Based Feature Importance Selector"
 
    @staticmethod
    def getNameAbbreviation():
        return "TBFIS"
    
    @staticmethod
    def getInfo():
        return ""

    def getSupport(self, indices: bool = False):
        return self._sel.get_support(indices)
    