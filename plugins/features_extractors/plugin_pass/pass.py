"""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor

class Pass(FeatureExtractor):
    """
    Pass feature extractor plugin for ClassMark. This extractor
    just pass string value as it is and if it is number than string is converted
    to number.
    """
    @staticmethod
    def getAttributes():
        return []
    
    @staticmethod
    def getName():
        return "Pass"
    
    @staticmethod
    def getNameAbbreviation():
        return "Pass"
 
    @staticmethod
    def getInfo():
        return ""
    
    def fit(self):
        pass

    def extract(self):
        pass
        