"""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor

class TFIDF(FeatureExtractor):
    """
    TF-IDF feature extractor plugin for ClassMark.
    """
    @staticmethod
    def getAttributes():
        return []
    
    @staticmethod
    def getName():
        return "Term Frequency–Inverse Document Frequency"
    
    @staticmethod
    def getNameAbbreviation():
        return "TF-IDF"
 
    @staticmethod
    def getInfo():
        return ""
    
    def fit(self, data, labels=None):
        pass

    def extract(self, data):
        pass
    
    def fitAndExtract(self, data, labels=None):
        pass