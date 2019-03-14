"""
Created on 4. 3. 2019
SVM classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin
from sklearn.svm import LinearSVC



class SVM(Classifier):
    """
    SVM classifier plugin for ClassMark.
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
        return "Support Vector Machines"
    
    @staticmethod
    def getNameAbbreviation():
        return "SVM"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self, data, labels):
        if self._normalizer.value is not None:
            data=self._normalizer.value.fitTransform(data)
        
        #The documentation says:
        #    Prefer dual=False when n_samples > n_features.
        self._cls=LinearSVC(dual=data.shape[0]<=data.shape[1], max_iter=self._maxIter.value)
        self._cls.fit(data,labels)
    
    def predict(self, data):
        if self._normalizer.value is not None:
            data=self._normalizer.value.transform(data)
        return self._cls.predict(data)
        