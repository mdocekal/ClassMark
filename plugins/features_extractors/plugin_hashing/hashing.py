"""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor, PluginAttribute
from sklearn.feature_extraction.text import HashingVectorizer

class Hashing(FeatureExtractor):
    """
    Hashing feature extractor plugin for ClassMark. 
    Uses method called feature hashing.
    """

    def __init__(self, nonNegative:bool=True, nFeatures:int=65536):
        """
        Initialize Hashing features extractor.
        
        :param nonNegative: Generate only nonnegative values (True).
        :type nonNegative: bool
        :param nFeatures: Number of features
        :type nFeatures: int
        """
        
        self._nonNegative=PluginAttribute("Non negative", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._nonNegative.value=nonNegative
        self._nFeatures=PluginAttribute("Number of features", PluginAttribute.PluginAttributeType.VALUE, int)
        self._nFeatures.value=nFeatures
    
        self._vectorizer=None
        
    def _createVectorizer(self):
        """
        Creates hashing vectorizer.
        """
        self._vectorizer=HashingVectorizer(non_negative=self._nonNegative,n_features=self._nFeatures)
        
    @staticmethod
    def getName():
        return "Hashing"
    
    @staticmethod
    def getNameAbbreviation():
        return "Hashing"
 
    @staticmethod
    def getInfo():
        return ""
    
    def fit(self, data, labels=None):
        """
        This function is there only for compatibility with FeatureExtractor.
        There is really no need to call it, but if you do so, than this is just empty
        operation.
        
        :param data: It does not matter.
        :type data: It does not matter.
        :param labels: It does not matter.
        :type labels: It does not matter.
        """
        pass

    def extract(self, data):
        """
        Extracts feature with hashing trick method.
        
        :param data: Original data for features extraction.
        :type data: ArrayLike
        :return: Converted data.
        :rtype: scipy.sparse matrix
        """
        try:
            return self._vectorizer.transform(data)
        except AttributeError:
            #vectorizer is probably not created yet.
            self._createVectorizer()
            return self._vectorizer.transform(data)
        
        