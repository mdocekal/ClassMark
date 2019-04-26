"""
Created on 10. 3. 2019
Module for experiment results.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .plugins import Classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from typing import List, Any
import numpy as np


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
        
        def addResults(self, classifier:Classifier, predictions:np.array):
            """
            Add experiment results for classifier.
            
            :param classifier: The classifier.
            :type classifier: Classifier
            :param predictions: Classifier's predictions of labels.
            :type predictions:np.array
            """
            self._results[classifier]=predictions
                
                
        def resultsForCls(self,classifier:Classifier):
            """
            Get results for classifier.
            
            :param classifier: The classifier
            :type classifier: Classifier
            :return: Predicted labels for this step.
            :rtype: np.array
            """
            
            return self._results[classifier]
        
        def score(self):
            """
            Calculates score for this step.
            
            :return: Score.
                {
                    classifier:{
                        'accuracy':val,
                        'macroPrecision':val,
                        'macroRecall':val,
                        'macroF1score':val,
                        'microPrecision':val,
                        'microRecall':val,
                        'microF1score':val,
                        'weightedPrecision':val,
                        'weightedRecall':val,
                        'weightedF1score':val
                    }
                }
            :rtype: Dict[Classifier,Dict[str,float]]
            """
            
            res={}
            for c, predictions in self._results.items():
                macroPrecision, macroRecall, macroF1score,_=precision_recall_fscore_support(self.labels, predictions, average='macro')
                microPrecision, microRecall, microF1score,_=precision_recall_fscore_support(self.labels, predictions, average='micro')
                weightedPrecision, weightedRecall, weightedF1score,_=precision_recall_fscore_support(self.labels, predictions, average='weighted')
                res[c]={
                    'accuracy':accuracy_score(self.labels, predictions),
                    'macroPrecision':macroPrecision,
                    'macroRecall':macroRecall,
                    'macroF1score':macroF1score,
                    'microPrecision':microPrecision,
                    'microRecall':microRecall,
                    'microF1score':microF1score,
                    'weightedPrecision':weightedPrecision,
                    'weightedRecall':weightedRecall,
                    'weightedF1score':weightedF1score
                    }
            
            return res

    def __init__(self, numOfSteps:int, classifiers:List[Classifier], labelEncoder:LabelEncoder):
        """
        Initialization of results.
        
        :param numOfSteps: Total number of steps to complete validation process for one classifier.
        :type numOfSteps: int
        :param classifiers: Classifiers that will be tested. This parameter is mainly used to determinate classifier order.
        :type classifiers: List[Classifier]
        :param labelcEncoder: We assume that providet labels are in encoded form. This encoder
            serves for getting real labels.
        :type labelcEncoder:LabelEncoder
        """
        
        self.steps=[self.ValidationStep() for _ in range(numOfSteps)]
        self._classifiers=classifiers
        self.encoder=labelEncoder
        self._finalize=False
        self._finalScore=None
        
    def finalize(self):
        """
        Mark this result as complete.
        When you mark the result than repeated calling of score method will be faster, because
        the score will be computed only for the first time and next time the saved one will be used.
        """
        self._finalize=True
       
    @property
    def classifiers(self):
        """
        Tested classifiers.
        """
        return self._classifiers
        

    def scores(self):
        """
        Average scores for all classifiers among all steps.
        
        :return: Score.
                {
                    classifier:{
                        'accuracy':val,
                        'macroPrecision':val,
                        'macroRecall':val,
                        'macroF1score':val,
                        'microPrecision':val,
                        'microRecall':val,
                        'microF1score':val,
                        'weightedPrecision':val,
                        'weightedRecall':val,
                        'weightedF1score':val
                    }
                }
        :rtype: Dict[Classifier,Dict[str,float]] | None
        """
        if len(self.steps)==0:
            return None
        
        if self._finalScore is not None:
            #use cached
            return self._finalScore
        
        res={}
        #let's calc sum, it will be used for avg calculation, over all scores
        for s in self.steps:
            for c, scores in s.score().items():
                for scoreType, scoreVal in scores.items():
                    try:
                        res[c][scoreType]+=scoreVal  #calc sum
                    except KeyError:
                        if c not in res:
                            res[c]={}
                        res[c][scoreType]=scoreVal
                    
            
        #calc avg from sum and number of steps
        for scores in res.values():
            for scoreType, scoreVal in scores.items():
                scores[scoreType]=scoreVal/len(self.steps)
        
        if self._finalize:
            #these are final results so we can cache the scores
            self._finalScore=res
        return res
            
