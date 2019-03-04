"""
Created on 4. 3. 2019
SVM classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier

class SVM(Classifier):
    """
    SVM classifier plugin for ClassMark.
    """
    @staticmethod
    def getAttributes():
        return []
    
    @staticmethod
    def getName():
        return "Support Vector Machines"
    
    @staticmethod
    def getNameAbbreviation():
        return "SVM"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self):
        pass
    
    def predict(self):
        pass
        