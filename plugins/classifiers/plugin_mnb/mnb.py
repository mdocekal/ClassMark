"""
Created on 24. 4. 2019
Multinomial Naive Bayes classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin
    
from sklearn.naive_bayes import MultinomialNB

class MultinomialNaiveBayes(Classifier):
    """
    MultinomialNaiveBayes classifier.
    """

    def __init__(self, normalizer:BaseNormalizer=None):
        """
        Classifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer

        """
        
        #TODO: type control must be off here (None -> BaseNormalizer) maybe it will be good if one could pass
        #object
        self._normalizer=PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                         [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin])
        self._normalizer.value=normalizer

        
    @staticmethod
    def getName():
        return "Multinomial Naive Bayes"
    
    @staticmethod
    def getNameAbbreviation():
        return "MNB"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self, data, labels):
        if self._normalizer.value is not None:
            data=self._normalizer.value.fitTransform(data)
        
        self._cls=MultinomialNB()
            
        self._cls.fit(data,labels)
    
    def predict(self, data):
        if self._normalizer.value is not None:
            data=self._normalizer.value.transform(data)
        return self._cls.predict(data)