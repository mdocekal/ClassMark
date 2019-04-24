"""
Created on 10. 3. 2019
Module for experiment results.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .plugins import Classifier
from sklearn.preprocessing import LabelEncoder
from typing import List, Any


class Results(object):
    """
    Storage of experiment results.
    """
    
    class ValidationStep(object):
        """
        Results for 
        """
        def __init__(self):
            self._results={}
            self.labels=None #True labels for test data in that validation step.
        
        def addResults(self, classifier:Classifier, predictions:List[Any]):
            """
            Add experiment results for classifier.
            
            :param classifier: The classifier.
            :type classifier: Classifier
            :param predictions: Classifier's predictions of labels.
            :type predictions:List[Any]
            """
            self._results[classifier.getName()]=[predictions]
                
                
        def resultsForCls(self,classifier:Classifier):
            """
            Get results for classifier.
            
            :param classifier: The classifier
            :type classifier: Classifier
            :return: Predicted labels for this step.
            :rtype: ArrayLike
            """
            
            return self._results[classifier.getName()]
            

    def __init__(self, numOfSteps:int, labelEncoder:LabelEncoder):
        
        """
        Initialization of results.
        
        :param numOfSteps: Total number of steps to complete validation process for one classifier.
        :type numOfSteps: int
        :param labelcEncoder: We assume that providet labels are in encoded form. This encoder
            serves for getting real labels.
        :type labelcEncoder:LabelEncoder
        """
        
        self.steps=[self.ValidationStep() for _ in range(numOfSteps)]
        self.encoder=labelEncoder
        
    
            
