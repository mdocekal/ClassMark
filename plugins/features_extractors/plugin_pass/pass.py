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
    to number. Even this behavior could be suppressed and than its do literally nothing with input data.
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
        pass
        