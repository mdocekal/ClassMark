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

    def __init__(self, nonNegative:bool=False, nFeatures:int=65536, norm="l2"):
        """
        Initialize Hashing features extractor.
        
        :param nonNegative: Generate only nonnegative values (True).
        :type nonNegative: bool
        :param nFeatures: Number of features
        :type nFeatures: int
        :param norm: Type of normalization.
        :type norm: str|None
        """
        
        self._nonNegativ=PluginAttribute("Non negative values", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._nonNegativ.value=nonNegative
        
        self._nFeatures=PluginAttribute("Number of features", PluginAttribute.PluginAttributeType.VALUE, int)
        self._nFeatures.value=nFeatures
        
        self._norm=PluginAttribute("Normalization", PluginAttribute.PluginAttributeType.SELECTABLE, str,
                                            [None,"l1","l2",])
        self._norm.value=norm
    
        self._vectorizer=None
        
    @staticmethod
    def getName():
        return "Hashing"
    
    @staticmethod
    def getNameAbbreviation():
        return "Hashing"
 
    @staticmethod
    def getInfo():
        return ""
    
    def _initExt(self):
        """
        Creates and initializes extractor according to current attributes values.
        """
        self._ext=HashingVectorizer(n_features=self._nFeatures.value, non_negative=self._nonNegativ.value,
                                    norm=self._norm.value)
    
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
        self._initExt()

    def extract(self, data):
        """
        Extracts feature with hashing trick method.
        
        :param data: Original data for features extraction.
        :type data: ArrayLike
        :return: Converted data.
        :rtype: scipy.sparse matrix
        """
        return self._ext.transform(data)

        
    def fitAndExtract(self, data, labels=None):
        self._initExt()
        return self._ext.fit_transform(data,labels)
        
        