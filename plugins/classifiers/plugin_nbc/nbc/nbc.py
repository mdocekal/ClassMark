"""
Created on 24. 4. 2019
Naive Bayes classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin
    
from scipy.sparse import spmatrix
    
from sklearn.naive_bayes import MultinomialNB, GaussianNB

class NaiveBayesClassifier(Classifier):
    """
    Naive Bayes classifier.
    """

    def __init__(self, normalizer:BaseNormalizer=None, typeV:str="gaussian"):
        """
        Classifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer
        :param typeV: Type of naive bayes.
            multinomial    -    Discrete non negative values like counts or something like that.
            gaussian    -    Continuous values. Likelihood of features is assumed to be gaussian.
        :type typeV: str
        """
        

        #object
        self._normalizer=PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                         [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin])
        self._normalizer.value=normalizer
        
        self._type=PluginAttribute("Type", PluginAttribute.PluginAttributeType.SELECTABLE, str,
                                         ["gaussian", "multinomial"])
        self._type.value=typeV

        
    @staticmethod
    def getName():
        return "Naive Bayes Classifier"
    
    @staticmethod
    def getNameAbbreviation():
        return "NBC"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self, data, labels):
        if self._normalizer.value is not None:
            data=self._normalizer.value.fitTransform(data)
        
        if self._type.value=="gaussian":
            self._cls=GaussianNB()
            if isinstance(data, spmatrix):
                data=data.A  #this classifier does not like sparse matrices
        else:
            self._cls=MultinomialNB()
        
        self._cls.fit(data,labels)
            
        
    
    def classify(self, data):
        if self._normalizer.value is not None:
            data=self._normalizer.value.transform(data)
            
        if (isinstance(self._cls, GaussianNB)):
            #this classifier does not like sparse matrices
            if isinstance(data, spmatrix):
                data=data.A #this classifier does not like sparse matrices
        return self._cls.predict(data)