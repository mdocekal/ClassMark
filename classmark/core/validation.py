"""
Created on 9. 3. 2019
Module for validation methods.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from scipy.sparse import hstack
import numpy as np
from abc import abstractmethod
from typing import List, Callable
from .plugins import Plugin, PluginAttribute, Classifier, FeatureExtractor
from PySide2.QtCore import Signal
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold

class EvaluationMethod(object):
    """
    Evaluation method for experiment.
    """
    pass

class Validator(Plugin):
    """
    Base class for all validation methods.
    """
    
    subStep=Signal()
    """Sub. step was performed. (features extraction, training classifier ...)"""
    
    actStepDesc=Signal(str)
    """Is emitted when step begins and sends description of actual step."""
    

    def run(self, classifier:Classifier, data:np.array, labels:np.array, extMap:List[FeatureExtractor]):
        """
        Run whole validation process on given classifier with
        given data. It is implemented as generator that provides predicted labels and real labels for test set on the
        end of validation step.
        
        :param classifier: Classifier you want to test.
        :type classifier: Classifier
        :param data: Data which will be used for validation.
        :type data: np.array
        :param labels: Labels which will be used for validation.
        :type labels: np.array
        :param extMap: Index of a FeatureExtractor, in that list corresponds
            to column, in data matrix, to which the extractor will be used.
        :type extMap: List[FeatureExtractor]
        """
        
        for trainIndices, testIndices in self.splitter(data, labels):
            """
            self.actInfo.emit("Extracting features {} on samples. Step {}/{}.".format(
                trainIndices.shape[0],classifier.getName(), step+1, self.numOfSteps(data, labels)))
            """
            #feature extraction for training set
            trainLabels=labels[trainIndices]
            trainFeatures=self._featuresStep(data, trainIndices, extMap, trainLabels)

            #train classifier
            classifier.train(trainFeatures, trainLabels)
            
            #free memory
            del trainFeatures
            del trainLabels
            
            #feature extraction for test set
            testFeatures=self._featuresStep(data, testIndices, extMap)

            #predict the labels
            predictedLabels=classifier.predict(testFeatures)
            
            yield (predictedLabels, labels[testIndices])
            
    def _featuresStep(self, data:np.array, useSamples, extMap:List[FeatureExtractor], labels:np.array=None):
        """
        Extracting features step.
        
        :param data: Samples for extraction.
        :type data: np.array
        :param useSamples: Indices of samples that should be used.
        :type useSamples:array-like int
        :param extMap:
        :type extMap:
        :param labels: Labels for labeled samples.
        :type labels: np.array
        """
        features=None
        for i,extractor in enumerate(extMap):
            """
            self.actStepDesc.emit("Extracting features from {} samples with {} for attribute {}".format(
                useSamples.shape[0], extractor.getName()))
                """
            actF=extractor.fitAndExtract(data[useSamples,None,i],labels) 
            #append the features to make one shared vector
            features= actF if features is None else hstack([features,actF])
            #self.subStep.emit()
        return features
    
    @abstractmethod
    def step(self):
        """
        Run one step of validation process.
        """
        pass
    
    @property
    @abstractmethod
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    @abstractmethod
    def numOfSteps(self, data:np.array=None, labels:np.array=None):
        """
        Total number of steps to complete validation process for one classifier.
        
        :param data: Data which will be used for validation.
        :type data: np.array
        :param labels: Labels which will be used for validation.
        :type labels: np.array
        """
        pass
    
    @property
    @abstractmethod
    def splitter(self):
        """
        Train/test set splitter.
        Should be callable that takes two arguments data,labels and returns
        touple (indices for train set, indices for test set).
        """
        pass
        
    
class ValidatorStratifiedKFold(Validator):
    """
    Validation process that uses StratifiedKFold for getting
    train and test sets.
    """
    
    def __init__(self, folds:int=5):
        """
        Initialize StratifiedKFold validation
        
        :param folds: Number of folds.
        :type folds: int
        """
        
        self._folds=PluginAttribute("Folds", PluginAttribute.PluginAttributeType.VALUE, int)
        self._folds.value=folds
        
        self._shuffle=PluginAttribute("Shuffle", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._shuffle.value=False
        
        self._randomSeed=PluginAttribute("Shuffle - random seed", PluginAttribute.PluginAttributeType.VALUE, int)
        self._randomSeed.value=None
        
    @property
    def splitter(self):
        self._spliter=StratifiedKFold(n_splits=self._folds.value,
                                      shuffle=self._shuffle.value,
                                      random_state=self._randomSeed.value)
        return self._spliter.split
        
        
    @staticmethod
    def getName():
        return "stratified-k-fold"
 
    @staticmethod
    def getNameAbbreviation():
        return "SKF"
    
    @staticmethod
    def getInfo():
        return ""
    
    def step(self):
        """
        Run one step of validation process.
        """
        pass
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    def numOfSteps(self, data:np.array=None, labels:np.array=None):
        return self._folds.value

class ValidatorKFold(Validator):
    """
    Validation process that uses KFold for getting
    train and test sets.
    """
    
    def __init__(self, folds:int=5):
        """
        Initialize ValidationKFold validation
        
        :param folds: Number of folds.
        :type folds: int
        """
        
        self._folds=PluginAttribute("Folds", PluginAttribute.PluginAttributeType.VALUE, int)
        self._folds.value=folds
    
    @staticmethod
    def getName():
        return "k-fold"
 
    @staticmethod
    def getNameAbbreviation():
        return "KF"
    
    @staticmethod
    def getInfo():
        return ""
    
    def run(self):
        """
        Run whole validation process.
        """
        pass

    def step(self):
        """
        Run one step of validation process.
        """
        pass
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    def numOfSteps(self, data:np.array=None, labels:np.array=None):
        return self._folds   

class ValidatorLeaveOneOut(Validator):
    """
    Validation process that uses LeaveOneOut for getting
    train and test sets.
    """
    
    @staticmethod
    def getName():
        return "leave-one-out"
 
    @staticmethod
    def getNameAbbreviation():
        return "LOO"
    
    @staticmethod
    def getInfo():
        return ""
    
    def run(self):
        """
        Run whole validation process.
        """
        pass

    def step(self):
        """
        Run one step of validation process.
        """
        pass
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    

    def numOfSteps(self, data:np.array=None, labels:np.array=None):
        return labels.shape[0]