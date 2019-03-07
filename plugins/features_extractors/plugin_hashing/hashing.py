"""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor

class Hashing(FeatureExtractor):
    """
    Hashing feature extractor plugin for ClassMark. 
    Uses method called feature hashing.
    """
    @staticmethod
    def getAttributes():
        return []
    
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
        pass

    def extract(self, data):
        pass
        