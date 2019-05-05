"""
Created on 24. 4. 2019
Decision Tree classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin
    
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(Classifier):
    """
    Decision Tree classifier.
    """

    def __init__(self, normalizer:BaseNormalizer=None, randomSeed:int=None, criterion:str="gini"):
        """
        lassifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer
        :param randomSeed: If not None than fixed seed is used.
        :type randomSeed: int
        :param criterion: The function to measure the quality of a split. 
        :type criterion: str
        """
        
        #TODO: type control must be off here (None -> BaseNormalizer) maybe it will be good if one could pass
        #object
        self._normalizer=PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                         [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin])
        self._normalizer.value=normalizer

        self._randomSeed=PluginAttribute("Random seed", PluginAttribute.PluginAttributeType.VALUE, int)
        self._randomSeed.value=randomSeed
        
        self._criterion=PluginAttribute("Split criterion", PluginAttribute.PluginAttributeType.SELECTABLE, str,
                                        ["Gini Index", "Information Gain"])
        self._criterion.value=criterion
        
    @staticmethod
    def getName():
        return "Decision Tree Classifier"
    
    @staticmethod
    def getNameAbbreviation():
        return "DTC"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self, data, labels):
        if self._normalizer.value is not None:
            data=self._normalizer.value.fitTransform(data)
        
        self._cls=DecisionTreeClassifier(random_state=self._randomSeed.value, criterion=self._criterion.value)
            
        self._cls.fit(data,labels)
    
    def classify(self, data):
        if self._normalizer.value is not None:
            data=self._normalizer.value.transform(data)
        return self._cls.predict(data)