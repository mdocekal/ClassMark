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
from enum import Enum, auto
import numpy as np
import copy

class Results(object):
    """
    Storage of experiment results.
    """

    class ScoreType(Enum):
        """
        Average scores.
        """
        
        ACCURACY=auto()
        """Average accuracy among all steps."""
        
        MACRO_PRECISION=auto()
        """Average macro precision among all steps."""
        
        MACRO_RECALL=auto()
        """Average macro recall among all steps."""
        
        MACRO_F1=auto()
        """Average macro F1 among all steps."""
        
        MICRO_PRECISION=auto()
        """Average micro precision among all steps."""
        
        MICRO_RECALL=auto()
        """Average micro recall among all steps."""
        
        MICRO_F1=auto()
        """Average micro F1 among all steps."""
        
        WEIGHTED_PRECISION=auto()
        """Average weighted precision among all steps."""
        
        WEIGHTED_RECALL=auto()
        """Average weighted recall among all steps."""
        
        WEIGHTED_F1=auto()
        """Average weighted F1 among all steps."""
    
    class ValidationStep(object):
        """
        Results for 
        """
        def __init__(self):
            self._predictions={}
            self._times={}
            self._stats={}
            self.labels=None #True labels for test data in that validation step.
        
        def addResults(self, classifier:Classifier, predictions:np.array, times, stats):
            """
            Add experiment results for classifier.
            
            :param classifier: The classifier.
            :type classifier: Classifier
            :param predictions: Classifier's predictions of labels.
            :type predictions:np.array
            :param times: dict of times
            :type times: Dict[Validator.TimeDuration,float]
            :param stats: samples stats
            :type stats: Dict[Validator.SamplesStats,float]
            """
            self._predictions[classifier.stub()]=predictions
            self._times[classifier.stub()]=times
            self._stats[classifier.stub()]=stats
                
                
        def predictionsForCls(self,classifier:Classifier):
            """
            Get predictions for classifier.
            
            :param classifier: The classifier
            :type classifier: Classifier
            :return: Predicted labels for this step.
            :rtype: np.array
            """
            
            return self._predictions[classifier.stub()]
        
        def timesForCls(self,classifier:Classifier):
            """
            Get times for classifier.
            
            :param classifier: The classifier
            :type classifier: Classifier
            :return: Times for classifier.
            :rtype: Dict[Validator.TimeDuration,float]
            """
            
            return self._times[classifier.stub()]
        
        def statsForCls(self,classifier:Classifier):
            """
            Get stats like number of samples and features.
            
            :param classifier: The classifier.
            :type classifier: Classifier
            :return: samples stats
            :rtype: Dict[Validator.SamplesStats,float]
            """
            
            return self._stats[classifier.stub()]
        
        def score(self):
            """
            Calculates score for this step.
            
            :return: Score.
                Dict key is classifier stub.
            :rtype: Dict[PluginStub,Dict[ScoreType,float]]
            """
            
            res={}
            for c, predictions in self._predictions.items():
                macroPrecision, macroRecall, macroF1score,_=precision_recall_fscore_support(self.labels, predictions, average='macro')
                microPrecision, microRecall, microF1score,_=precision_recall_fscore_support(self.labels, predictions, average='micro')
                weightedPrecision, weightedRecall, weightedF1score,_=precision_recall_fscore_support(self.labels, predictions, average='weighted')
                res[c]={
                    Results.ScoreType.ACCURACY:accuracy_score(self.labels, predictions),
                    Results.ScoreType.MACRO_PRECISION:macroPrecision,
                    Results.ScoreType.MACRO_RECALL:macroRecall,
                    Results.ScoreType.MACRO_F1:macroF1score,
                    Results.ScoreType.MICRO_PRECISION:microPrecision,
                    Results.ScoreType.MICRO_RECALL:microRecall,
                    Results.ScoreType.MICRO_F1:microF1score,
                    Results.ScoreType.WEIGHTED_PRECISION:weightedPrecision,
                    Results.ScoreType.WEIGHTED_RECALL:weightedRecall,
                    Results.ScoreType.WEIGHTED_F1:weightedF1score
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
        self._classifiers=[c.stub() for c in classifiers]
        self.encoder=labelEncoder
        self._finalize=False
        self._finalScore=None
        self._finalTimes=None

    def finalize(self):
        """
        Mark this result as complete.
        When you mark the result than repeated calling of score (and other metrics) method will be faster, because
        the score will be computed only for the first time and next time the saved one will be used.
        """
        self._finalize=True
       
    @property
    def classifiers(self):
        """
        Tested classifiers plugin stubs.
        """
        return self._classifiers
    
    @property
    def times(self):
        """
        Average times for all classifiers among all steps.
        
        :return: Average times.
        :rtype: Dict[PluginStub,Dict[Validator.TimeDuration,float]] | None
        """
        
        if len(self.steps)==0:
            return None
        
        if self._finalTimes is not None:
            #use cached
            return self._finalTimes
        
        stepIter=iter(self.steps)
        try:
            #get the sum for avg
            res=copy.copy(next(stepIter)._times)
            while True:
                for c,cVals in next(stepIter)._times.items():
                    for t,v in cVals.items():
                        res[c][t]+=v
        except StopIteration:
            #ok we have all steps
            #let's calc the avg
            
            
            for c,cVals in res.items():
                for t,v in cVals.items():
                    res[c][t]=v/len(self.steps)
        
        if self._finalize:
            #these are final results so we can cache the avg times
            self._finalTimes=res
            
        return res
        
    @property
    def scores(self):
        """
        Average scores for all classifiers among all steps.
        
        :return: Score.
            Dict key is classifier plugin stub.
        :rtype: Dict[PluginStub,Dict[ScoreType,float]] | None
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
            
