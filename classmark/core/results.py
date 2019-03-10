"""
Created on 10. 3. 2019
Module for experiment results.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .plugins import Classifier

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
            

    def __init__(self, numOfSteps:int):
        """
        Initialization of results.
        
        :param numOfSteps: Total number of steps to complete validation process for one classifier.
        :type numOfSteps:int
        """
        
        self.steps=[self.ValidationStep() for _ in range(numOfSteps)]
        
    
            
