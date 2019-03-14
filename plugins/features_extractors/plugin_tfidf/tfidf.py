"""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor, PluginAttribute
from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDF(FeatureExtractor):
    """
    TF-IDF feature extractor plugin for ClassMark.
    """
    
    def __init__(self, maxFeatures:int=None, caseSensitive:bool=False):
        """
        Feature extractor initialization.
        
        :param maxFeatures: Limit to the maximum number of features. None means unlimited.
        :type maxFeatures: None | int
        :param caseSensitive: True means that we want to be case senstive.
        :type caseSensitive: bool

        """
        
        self._maxFeatures=PluginAttribute("Max number of features", PluginAttribute.PluginAttributeType.VALUE, int)
        self._maxFeatures.value=maxFeatures
        
        self._caseSensitive=PluginAttribute("Case sensitive", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._caseSensitive.value=caseSensitive
    
    @staticmethod
    def getName():
        return "Term Frequency–Inverse Document Frequency"
    
    @staticmethod
    def getNameAbbreviation():
        return "TF-IDF"
 
    @staticmethod
    def getInfo():
        return ""
    
    def _initExt(self):
        """
        Creates and initializes extractor according to current attributes values.
        """
        self._ext=TfidfVectorizer(max_features=self._maxFeatures.value,lowercase=not self._caseSensitive.value)
    
    def fit(self, data, labels=None):
        self._initExt()
        self._ext.fit(data)
        
    def extract(self, data):
        return self._ext.transform(data)
    
    def fitAndExtract(self, data, labels=None):
        self._initExt()
        return self._ext.fit_transform(data)