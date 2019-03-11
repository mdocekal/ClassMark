"""
Created on 11. 3. 2019
Module for features preprocessing techniques.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .plugins import Plugin, PluginAttribute
from abc import abstractmethod

from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler, RobustScaler
from scipy.sparse import spmatrix

class BaseNormalizer(Plugin):
    """
    Base class for scalers and normalizers.
    """
    
    @abstractmethod
    def fitTransform(self, data):
        """
        Normalizes/scales values, but also gets informations from input data that need to learned.
        
        :param data: Data for preparation.
        :type data: ArrayLike
        """
        pass
    
    @abstractmethod
    def transform(self, data):
        """
        Normalizes/scales values.
        
        :param data: Data for preparation.
        :type data: ArrayLike
        """
        pass
    
class NormalizerPlugin(BaseNormalizer):
    """
    Scale input vectors individually to unit norm (vector length).
    """
    
    @staticmethod
    def getName():
        return "Normalizer"
    
    @staticmethod
    def getNameAbbreviation():
        return "N"
    
    @staticmethod
    def getInfo():
        return ""
    
    def fitTransform(self, data):
        return normalize(data)
    
    def transform(self, data):
        return normalize(data)
        
class MinMaxScalerPlugin(BaseNormalizer):
    """
    Transforms features by scaling each feature to a given range.
    """
    
    def __init__(self, minV:float=0, maxV:float=1):
        """
        MinMaxScaler initialization.
        
        :param minV: Minimal value.
        :type minV: int
        :param maxV: Maximal value.
        :type maxV: int
        """
 
        self._min=PluginAttribute("Min", PluginAttribute.PluginAttributeType.VALUE, int)
        self._min.value=minV
        
        self._max=PluginAttribute("Max", PluginAttribute.PluginAttributeType.VALUE, int)
        self._max.value=maxV
    
    @staticmethod
    def getName():
        return "MinMaxScaler"
    
    @staticmethod
    def getNameAbbreviation():
        return "MM"
    
    @staticmethod
    def getInfo():
        return ""
    
    def fitTransform(self, data):
        self._scaler = MinMaxScaler(feature_range=(self._min.value, self._max.value))
        if isinstance(data,spmatrix):
            #because it throws error otherwise
            data=data.toarray()

        return self._scaler.fit_transform(data)
    
    def transform(self, data):
        if isinstance(data,spmatrix):
            #because it throws error otherwise
            data=data.toarray()
        return self._scaler.transform(data)

class RobustScalerPlugin(BaseNormalizer):
    """
    Scale features using statistics that are robust to outliers.
    """
    
    def __init__(self, centering:bool=True):
        """
        RobustScalerPlugin initialization.
        
        :param centering: If True, center the data before scaling
        :type centering: bool
        """
 
        self._centering=PluginAttribute("centering", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._centering.value=centering

    @staticmethod
    def getName():
        return "RobustScaler"
    
    @staticmethod
    def getNameAbbreviation():
        return "RS"
    
    @staticmethod
    def getInfo():
        return ""
    
    def fitTransform(self, data):
        self._scaler =RobustScaler(with_centering=self._centering.value)
        if self._centering.value and isinstance(data,spmatrix):
            #because it throws error otherwise
            data=data.toarray()
        return self._scaler.fit_transform(data)
    
    def transform(self, data):
        if self._centering.value and isinstance(data,spmatrix):
            #because it throws error otherwise
            data=data.toarray()
        return self._scaler.transform(data)

class StandardScalerPlugin(BaseNormalizer):
    """
    Standardize features by removing the mean and scaling to unit variance
    """
    
    def __init__(self, centering:bool=True):
        """
        StandardScalerPlugin initialization.
        
        :param centering: If True, center the data before scaling.
        :type centering: bool
        """
 
        self._centering=PluginAttribute("centering", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._centering.value=centering
        
    @staticmethod
    def getName():
        return "StandardScaler"
    
    @staticmethod
    def getNameAbbreviation():
        return "SS"
    
    @staticmethod
    def getInfo():
        return ""

    def fitTransform(self, data):
        self._scaler = StandardScaler(with_mean=self._centering.value)
        if self._centering.value and isinstance(data,spmatrix):
            #because it throws error otherwise
            data=data.toarray()

        return self._scaler.fit_transform(data)
    
    def transform(self, data):
        if self._centering.value and isinstance(data,spmatrix):
            #because it throws error otherwise
            data=data.toarray()
        return self._scaler.transform(data)