"""
Created on 18. 3. 2019
CEEF classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin



class CEEF(Classifier):
    """
    Classification by evolutionary estimated functions (or CEEF) is classification method that uses
    something like probability density functions, one for each class, to classify input data.
    """
    
    def __init__(self, normalizer:BaseNormalizer=NormalizerPlugin(), maxIter:int=1000):
        """
        Classifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer
        :param maxIter: Maximum number of iterations.
        :type maxIter: int
        """

        
        self._maxIter=PluginAttribute("Max iterations", PluginAttribute.PluginAttributeType.VALUE, int)
        self._maxIter.value=maxIter

        #TODO: type control must be off here (None -> BaseNormalizer) maybe it will be good if one could pass
        #object
        self._normalizer=PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                         [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin])
        self._normalizer.value=normalizer
        
    @staticmethod
    def getName():
        return "Classification by evolutionary estimated functions"
    
    @staticmethod
    def getNameAbbreviation():
        return "CEEF"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self, data, labels):
        pass
    
    def predict(self, data):
        pass
        