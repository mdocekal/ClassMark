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
from .selection import FeaturesSelector
from ..data.data_set import DataSet
from PySide2.QtCore import Signal
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold
from enum import Enum, auto
import time
from pip._internal.utils.misc import enum

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

    
    class SamplesStats(Enum):
        """
        Statistics of samples.
        """
        
        NUM_SAMPLES_TRAIN=auto()
        """Number of samples used for training."""
        
        NUM_SAMPLES_TEST=auto()
        """Number of samples used for testing."""
        
        NUM_FEATURES=auto()
        """Used number of features."""
        
    
    class TimeDuration(Enum):
        """
        All time durations that are measure inside one step.
        """
        FEATURE_EXTRACTION_TRAIN=auto()
        """Duration of feature extracting for train set measured with time.time()."""
        
        FEATURE_EXTRACTION_TRAIN_PROC=auto()
        """Duration of feature extracting for train set measured with time.process_time()."""
        
        FEATURE_SELECTION_TRAIN=auto()
        """Duration of feature selection for train set measured with time.time()."""
        
        FEATURE_SELECTION_TRAIN_PROC=auto()
        """Duration of feature selection for train set measured with time.process_time()."""
        
        TRAINING=auto()
        """Duration of training measured with time.time()."""
        
        TRAINING_PROC=auto()
        """Duration of training measured with time.process_time()."""
        
        FEATURE_EXTRACTION_TEST=auto()
        """Duration of feature extracting for test set measured with time.time()."""
        
        FEATURE_EXTRACTION_TEST_PROC=auto()
        """Duration of feature extracting for test set measured with time.process_time()."""
        
        FEATURE_SELECTION_TEST=auto()
        """Duration of feature selection for test set measured with time.time()."""

        FEATURE_SELECTION_TEST_PROC=auto()
        """Duration of feature selection for test set measured with time.process_time()."""
                
        TEST=auto()
        """Duration of testing measured with time.time()."""
        
        TEST_PROC=auto()
        """Duration of testing measured with time.process_time()."""
        
        

    def run(self, dataSet:DataSet, classifier:Classifier, data:np.array, labels:np.array, extMap:List[FeatureExtractor], \
            featSel:List[FeaturesSelector]=[]):
        """
        Run whole validation process on given classifier with
        given data. It is implemented as generator that provides predicted labels, real labels and times for train/test set on the
        end of validation step.
        
        :param dataSet: Dataset where the data come from.
            Some validators may used it to get additional informations.
        :type dataSet: DataSet 
        :param classifier: Classifier you want to test.
        :type classifier: Classifier
        :param data: Data which will be used for validation.
        :type data: np.array
        :param labels: Labels which will be used for validation.
        :type labels: np.array
        :param extMap: Index of a FeatureExtractor, in that list corresponds
            to column, in data matrix, to which the extractor will be used.
        :type extMap: List[FeatureExtractor]
        :param featSel: Features selectors. Selectors will be applied in same order
            as they appear in that list.
        :type featSel: List[FeaturesSelector]
        :return: Generator of tuple Tuple[predictedLabels,realLabels, dict of times, samples stats]
        :rtype: Iterator[Tuple[np.array,np.array,Dict[TimeDuration, float],Dict[SamplesStats,int]]]
        """
        
        self._lastDataSet=dataSet
        
        for trainIndices, testIndices in self._splitter(data, labels):
            s=time.time()
            """
            self.actInfo.emit("Extracting features {} on samples. Step {}/{}.".format(
                trainIndices.shape[0],classifier.getName(), step+1, self.numOfSteps(data, labels)))
            """
            #feature extraction for training set
            trainLabels=labels[trainIndices]

            times={self.TimeDuration.FEATURE_EXTRACTION_TRAIN:time.time(),
                   self.TimeDuration.FEATURE_EXTRACTION_TRAIN_PROC:time.process_time()}
            
            stats={}
            
            trainFeatures=self._featuresStep(data[trainIndices], extMap, trainLabels, fit=True)
            
            times[self.TimeDuration.FEATURE_EXTRACTION_TRAIN_PROC]=time.process_time()-times[self.TimeDuration.FEATURE_EXTRACTION_TRAIN_PROC]
            times[self.TimeDuration.FEATURE_EXTRACTION_TRAIN]=time.time()-times[self.TimeDuration.FEATURE_EXTRACTION_TRAIN]
            
            #feature selection
            times[self.TimeDuration.FEATURE_SELECTION_TRAIN]=time.time()
            times[self.TimeDuration.FEATURE_SELECTION_TRAIN_PROC]=time.process_time()
            for s in featSel:
                s.fit(trainFeatures, trainLabels)
                trainFeatures=s.select(trainFeatures)
            times[self.TimeDuration.FEATURE_SELECTION_TRAIN_PROC]=time.process_time()-times[self.TimeDuration.FEATURE_SELECTION_TRAIN_PROC]
            times[self.TimeDuration.FEATURE_SELECTION_TRAIN]=time.time()-times[self.TimeDuration.FEATURE_SELECTION_TRAIN]
            
            stats[self.SamplesStats.NUM_SAMPLES_TRAIN]=trainFeatures.shape[0]
            stats[self.SamplesStats.NUM_FEATURES]=trainFeatures.shape[1]
            
            times[self.TimeDuration.TRAINING]=time.time()
            times[self.TimeDuration.TRAINING_PROC]=time.process_time()
            #train classifier
            classifier.train(trainFeatures, trainLabels)

            times[self.TimeDuration.TRAINING_PROC]=time.process_time()-times[self.TimeDuration.TRAINING_PROC]
            times[self.TimeDuration.TRAINING]=time.time()-times[self.TimeDuration.TRAINING]

            #free memory
            del trainFeatures
            del trainLabels

            times[self.TimeDuration.FEATURE_EXTRACTION_TEST]=time.time()
            times[self.TimeDuration.FEATURE_EXTRACTION_TEST_PROC]=time.process_time()
            #feature extraction for test set
            testFeatures=self._featuresStep(data[testIndices], extMap)
            
            times[self.TimeDuration.FEATURE_EXTRACTION_TEST_PROC]=time.process_time()-times[self.TimeDuration.FEATURE_EXTRACTION_TEST_PROC]
            times[self.TimeDuration.FEATURE_EXTRACTION_TEST]=time.time()-times[self.TimeDuration.FEATURE_EXTRACTION_TEST]

            stats[self.SamplesStats.NUM_SAMPLES_TEST]=testFeatures.shape[0]
            
            #feature selection for test set
            times[self.TimeDuration.FEATURE_SELECTION_TEST]=time.time()
            times[self.TimeDuration.FEATURE_SELECTION_TEST_PROC]=time.process_time()
            for s in featSel:
                testFeatures=s.select(testFeatures)
            times[self.TimeDuration.FEATURE_SELECTION_TEST_PROC]=time.process_time()-times[self.TimeDuration.FEATURE_SELECTION_TEST_PROC]
            times[self.TimeDuration.FEATURE_SELECTION_TEST]=time.time()-times[self.TimeDuration.FEATURE_SELECTION_TEST]
            
            times[self.TimeDuration.TEST]=time.time()
            times[self.TimeDuration.TEST_PROC]=time.process_time()
            #predict the labels
            predictedLabels=classifier.predict(testFeatures)
            
            times[self.TimeDuration.TEST_PROC]=time.process_time()-times[self.TimeDuration.TEST_PROC]
            times[self.TimeDuration.TEST]=time.time()-times[self.TimeDuration.TEST]

            yield (predictedLabels, labels[testIndices], times, stats)
            
    def _featuresStep(self, data:np.array,extMap:List[FeatureExtractor], labels:np.array=None, fit=False):
        """
        Extracting features step.
        
        :param data: Samples for extraction.
        :type data: np.array
        :param extMap: Index of a FeatureExtractor, in that list corresponds
            to column, in data matrix, to which the extractor will be used.
        :type extMap: List[FeatureExtractor]
        :param labels: Labels for labeled samples.
        :type labels: np.array
        :param fit: Should do fit step ("train" on input).
        :type fit: bool
        """
        features=None
        
        #because hstack is quite costly operation and Pass can work on multiple attributes at once,
        #than lets make groups of attribute that should use Pass.
        concatenatePass=[]
        for i,extractor in enumerate(extMap):
            if extractor.getName()=="Pass":
                concatenatePass.append(i)
                continue
            elif len(concatenatePass)>0:
                if fit:
                    actF=extractor.fitAndExtract(data[:,[concatenatePass]],labels) 
                else:
                    actF=extractor.extract(data[:,[concatenatePass]]) 
                concatenatePass=[]
            else:
                if fit:
                    actF=extractor.fitAndExtract(data[:,i],labels) 
                else:
                    actF=extractor.extract(data[:,i]) 
            """
            self.actStepDesc.emit("Extracting features from {} samples with {} for attribute {}".format(
                useSamples.shape[0], extractor.getName()))
                """
            #append the features to make one shared vector
            features= actF if features is None else hstack([features,actF])

            #self.subStep.emit()
            
        if len(concatenatePass)>0:
            extractor=extMap[concatenatePass[0]]
            if len(concatenatePass)==len(extMap):
                if fit:
                    features=extractor.fitAndExtract(data,labels)
                else:
                    features=extractor.extract(data)
            else:
                if fit:
                    actF=extractor.fitAndExtract(data[:,[concatenatePass]],labels)
                else:
                    actF=extractor.extract(data[:,[concatenatePass]])
                    
                features= actF if features is None else hstack([features,actF])
            
        return features.tocsr()
    
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
    def numOfSteps(self, dataSet:DataSet=None, data:np.array=None, labels:np.array=None):
        """
        Total number of steps to complete validation process for one classifier.
        
        :param dataSet: Dataset where the data come from.
            Some validators may used it to get additional informations.
        :type dataSet: DataSet 
        :param data: Data which will be used for validation.
        :type data: np.array
        :param labels: Labels which will be used for validation.
        :type labels: np.array
        """
        pass
    
    @property
    @abstractmethod
    def _splitter(self):
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
    def _splitter(self):
        return StratifiedKFold(n_splits=self._folds.value,
                                      shuffle=self._shuffle.value,
                                      random_state=self._randomSeed.value).split
        
        
    @staticmethod
    def getName():
        return "stratified-k-fold"
 
    @staticmethod
    def getNameAbbreviation():
        return "SKF"
    
    @staticmethod
    def getInfo():
        return ""
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    def numOfSteps(self, dataSet:DataSet=None,data:np.array=None, labels:np.array=None):
        return self._folds.value

class ValidatorKFold(Validator):
    """
    Validation process that uses KFold for getting
    train and test sets.
    """
    
    def __init__(self, folds:int=5, shuffle:bool=False, randomSeed:int=None):
        """
        Initialize ValidationKFold validation
 
        :param folds: Number of folds.
        :type folds: int
        :param shuffle: Sould shuffle the data?
        :type shuffle: bool
        :param randomSeed: Given seed.
        :type randomSeed: None| int
        """
        
        self._folds=PluginAttribute("Folds", PluginAttribute.PluginAttributeType.VALUE, int)
        self._folds.value=folds
        
        self._shuffle=PluginAttribute("Shuffle", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._shuffle.value=shuffle
        
        self._randomSeed=PluginAttribute("Shuffle - random seed", PluginAttribute.PluginAttributeType.VALUE, int)
        self._randomSeed.value=randomSeed
        
    @property
    def _splitter(self):
        return KFold(n_splits=self._folds.value,
                                      shuffle=self._shuffle.value,
                                      random_state=self._randomSeed.value).split
    
    @staticmethod
    def getName():
        return "k-fold"
 
    @staticmethod
    def getNameAbbreviation():
        return "KF"
    
    @staticmethod
    def getInfo():
        return ""
    
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    def numOfSteps(self,dataSet:DataSet=None, data:np.array=None, labels:np.array=None):
        return self._folds.value

class ValidatorPeparedSets(Validator):
    """
    Validation process that uses prepared train/test sets.
    """
    
    class Splitter(object):
        """
        Functor that makes split according to selected attribute that contains
        marking.
        Marking example:
                1    First validation step train set.
                1t    First validation step test set.
                2    Second validation step train set.
                2t    Second validation step test set.
        """
        
        def __init__(self, dataSet:DataSet, attr:str):
            """
            Initialization of splitter.
            
            :param dataSet: This data set will be used for getting the attr values.
            :type dataSet: dataSet
            :param attr: Attribute name that contains marking.
            :type attr: str
            """
            
            self.dataSet=dataSet
            self.attr=attr
            self.splits={}
            for i, row in enumerate(self.dataSet):
                try:
                    m=row[self.attr]
                except KeyError:
                    raise ValueError("Non existing attribute: "+str(self.attr))
                    
                isTest=0
                if m.endswith("t"):
                    #test set
                    v=float(m[:-1])
                    isTest=1
                else:
                    #train set
                    v=float(m)
                    
                try:
                    splits[v][isTest].append(i)
                except KeyError:
                    splits[v]=([],[])
                    splits[v][isTest].append(i)
            
        def __callable__(self, data:np.array, labels:np.array):
            """
            One split step.
            
            :param data: Data which will be used for validation.
            :type data: np.array
            :param labels: Labels which will be used for validation.
            :type labels: np.array
            :return: (indices for train set, indices for test set)
            :rtype: Tuple[Tuple,Tuple]
            """
 
            for split in sorted(self._splits):
                yield split

    def __init__(self, attribute:str=None):
        """
        Initialize ValidationKFold validation
        
        :param attribute: Attribute that contains train/test set marking
            Marking example:
                1    First validation step train set.
                1t    First validation step test set.
                2    Second validation step train set.
                2t    Second validation step test set.
        :type attribute: str
        """

        self._attribute=PluginAttribute("Attribute", PluginAttribute.PluginAttributeType.VALUE, str)
        self._attribute.value=attribute
        
    @property
    def _splitter(self):
        """
        Beware that this splitter is using attribute column for experiment data set.
        So do not use other data than the one from the experiment.
        
        Also the splitter expects that member _lastDataSet is set.
        """
        if not hasattr(self, "_splitterObj") or self._splitterObj is None or \
            self._lastDataSet != self._splitterObj.dataSet or \
            self._splitterObj.attr!=self._attribute.value:
            
            if self._attribute.value is None:
                raise ValueError("Please fill field "+self._attribute.name+" for validator: "+self.getName()+".")
            self._splitterObj=self.Splitter(self._lastDataSet,self._attribute.value)

        return self._splitterObj
    
    @staticmethod
    def getName():
        return "Prepared sets"
 
    @staticmethod
    def getNameAbbreviation():
        return "PS"
    
    @staticmethod
    def getInfo():
        return ""
    
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    def numOfSteps(self, dataSet:DataSet=None,data:np.array=None, labels:np.array=None):
        self._lastDataSet=dataSet
        return len(self._splitter.splits)
    
class ValidatorLeaveOneOut(Validator):
    """
    Validation process that uses LeaveOneOut for getting
    train and test sets.
    """
    
    @property
    def _splitter(self):

        return LeaveOneOut().split
    
    @staticmethod
    def getName():
        return "leave-one-out"
 
    @staticmethod
    def getNameAbbreviation():
        return "LOU"
    
    @staticmethod
    def getInfo():
        return ""

    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    

    def numOfSteps(self, dataSet:DataSet=None,data:np.array=None, labels:np.array=None):
        return labels.shape[0]