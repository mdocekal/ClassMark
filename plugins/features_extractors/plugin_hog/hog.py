"""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor

class HOG(FeatureExtractor):
    """
    HOG feature extractor plugin for ClassMark.
    """
    @staticmethod
    def getAttributes():
        return []
    
    @staticmethod
    def getName():
        return "Histogram of Oriented Gradients"
    
    @staticmethod
    def getNameAbbreviation():
        return "HOG"
 
    @staticmethod
    def getInfo():
        return ""
    
    def fit(self):
        pass

    def extract(self):
        pass
        