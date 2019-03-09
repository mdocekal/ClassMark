"""
Created on 4. 3. 2019
SVM classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute
from sklearn.svm import LinearSVC

class SVM(Classifier):
    """
    SVM classifier plugin for ClassMark.
    """
    
    def __init__(self):
        """
        Classifier initialization.
        """
        self._cls=LinearSVC()

    @staticmethod
    def getName():
        return "Support Vector Machines"
    
    @staticmethod
    def getNameAbbreviation():
        return "SVM"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self, data, labels):
        self._cls.fit(data,labels)
    
    def predict(self, data):
        return self._cls.predict(data)
        