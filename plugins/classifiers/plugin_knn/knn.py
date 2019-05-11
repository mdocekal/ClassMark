"""
Created on 24. 4. 2019
KNN classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute, PluginAttributeIntChecker
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin
    
from sklearn.neighbors import KNeighborsClassifier

class KNN(Classifier):
    """
    k-Nearest Neighbor classifier.
    """

    def __init__(self, normalizer:BaseNormalizer=None, neighbors:int=3):
        """
        lassifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer
        :param neighbors: Number of neighbors (k).
        :type neighbors: int
        """
        
        #TODO: type control must be off here (None -> BaseNormalizer) maybe it will be good if one could pass
        #object
        self._normalizer=PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                         [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin])
        self._normalizer.value=normalizer
        
        self._neighbors=PluginAttribute("Neighbors", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._neighbors.value=neighbors
        
    @staticmethod
    def getName():
        return "k-Nearest Neighbor"
    
    @staticmethod
    def getNameAbbreviation():
        return "KNN"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self, data, labels):
        if self._normalizer.value is not None:
            data=self._normalizer.value.fitTransform(data)
        
        self._cls=KNeighborsClassifier(n_neighbors=self._neighbors.value)
            
        self._cls.fit(data,labels)
    
    def classify(self, data):
        if self._normalizer.value is not None:
            data=self._normalizer.value.transform(data)
        return self._cls.predict(data)