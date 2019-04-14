"""
Created on 18. 2. 2019
Module for experiment representation and actions.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from ..data.data_set import DataSet
from enum import Enum
import time
from _collections import OrderedDict
from ..core.plugins import Plugin, CLASSIFIERS, FEATURE_EXTRACTORS
from ..core.validation import Validator
from builtins import isinstance
from PySide2.QtCore import QThread, Signal
from .results import Results
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import multiprocessing
import queue
from ..core.utils import sparseMatVariance, Observable
import copy
import statistics 
from pip._internal.utils.misc import enum
from functools import partial

class PluginSlot(object):
    """
    Slot that stores informations about selected plugin. 
    """
    
    def __init__(self, slotID):
        """
        Creates empty classifier slot
        
        :param slotID: Unique identifier of slot.
        :type slotID: int
        """
        self._id=slotID
        self.plugin=None
    
    @property
    def id(self):
        """
        Slot id.
        
        :return: id
        :rtype: int
        """
        return self._id
    
    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return self._id==other._id
        
    def __hash__(self):
        return self._id
    
    
class ExperimentDataStatistics(Observable):
    """
    Data statistics of experiment.
    """
    

    def __init__(self):
        super().__init__()
        self._classSamples={}
        self._classActivity={}
        self._classes=[]    #just to give order to the classes
        self._attributes=[]
        self._attributesFeatures={}
        self._attributesAVGFeatureVariance={}
        
    
    def isActive(self,c):
        """
        Determines if given class is active.
        
        :param c: Class for check.
        :type c: Any
        :return: True active. False otherwise.
        :rtype: bool
        """
        
        return self._classActivity[c]
    
    @Observable._event("SAMPLES_CHANGED")
    def samplesChangeEvent(self):
        """
        Informs all observers that samples stats changed.
        
        This method should be called inside this class only. For others it is just for
        observing.
        """
        pass

    
    def deactivateClass(self,c):
        """
        Deactivating a class means that this class is hidden and is no longer
        in properties like classes, classSamples,numberOfSamples, numberOfSamples,
        maxSamplesInClass and minSamplesInClass.
        
        :param c: Class from classes.
        :type c: Any
        """
        
        self._classActivity[c]=False
        self.samplesChangeEvent()

    def activateClass(self, c):
        """
        Activate class c.
        
        :param c: Class from classes.
        :type c: Any
        """
        self._classActivity[c]=True
        self.samplesChangeEvent()
        
    @property
    def attributes(self):
        """
        All attributes we have stats for.
        
        :return: List of attributes.
        :rtype: List[str]
        """
        
        return self._attributes
    
    @attributes.setter
    def attributes(self, attr):
        """
        Set all attributes. Is used mainly for the order of these attributes.
        WARNING: Cleares attributes metrics.
        
        :param attr: List of attributes.
        :type attr: List[str]
        """
        
        self._attributesFeatures={}
        self._attributesAVGFeatureVariance={}
        
        for a in attr:
            self._attributesFeatures[a]=0
            self._attributesAVGFeatureVariance[a]=0
            
        self._attributes=attr
        
    @property
    def classes(self):
        """
        Gives list of all classes.
        
        :return: classes
        :rtype: List[Any]
        """
        
        return self._classes
        
    @property
    def activeClasses(self):
        """
        Gives list of only active classes.
        
        :return: active classes
        :rtype: List[Any]
        """
        return [c for c in self._classes  if self._classActivity[c]]
    
    @property
    def classSamples(self):
        """
        Number of samples per class in form of dict where key is the class and value
        is the number of samples.
        """
        return { c:self._classSamples[c] for c in self.activeClasses}
    
    @classSamples.setter
    def classSamples(self, newCS:Dict[Any, int]):
        """
        Set new class samples.
        
        :param newCS: New number of samples per class.
        :type newCS: Dict[Any, int]
        """
        
        self._classSamples=newCS
        self._classActivity={c:True for c in newCS}
        self._classes=list(newCS.keys())
        self.samplesChangeEvent()
        
    def changeSamplesInClass(self,c,n):
        """
        Changes number of samples per class.
        
        :param c: The class.
        :type c: Any
        :param n: New number of samples in class.
        :type n: int
        """
        self._classSamples[c]=n
        self.samplesChangeEvent()
        
    @property
    def numberOfSamples(self):
        """
        Number of samples in whole data set.
        Is calculated from classSamples.
        """
        
        return sum( self._classSamples[c] for c in self.activeClasses)
        
    @property
    def maxSamplesInClass(self):
        """
        Gives maximal number of samples in class and also returns the class itself.
        
        :return: maximal number of samples in class and that class
        :rtype: Tuple[int, Any]
        """
        if len(self.activeClasses)>0:
            mC=max(self.activeClasses, key=lambda x: self._classSamples[x])
            return (self._classSamples[mC],mC)
        
        return (0,"")
    
    @property
    def minSamplesInClass(self):
        """
        Gives minimal number of samples in class and also returns the class itself.
        
        :return: minimal number of samples in class and that class
        :rtype: Tuple[int, Any]
        """
        
        if len(self.activeClasses)>0:
            mC=min(self.activeClasses, key=lambda x: self._classSamples[x])
            return (self._classSamples[mC],mC)
        
        return (0,"")
        
    @property
    def AVGSamplesInClass(self):
        """
        Average number of samples in class.
        
        :return: avg samples
        :rtype: float
        """
        if len(self.activeClasses)>0:
            return sum(self._classSamples[c] for c in self.activeClasses)/len(self.activeClasses)
        return 0
    
    @property
    def SDSamplesInClass(self):
        """
        Standard deviation of number of class samples.
        
        :return: SD of number of class samples
        :rtype: float
        """
        if len(self.activeClasses)>1:
            return statistics.stdev(self._classSamples[c] for c in self.activeClasses)
        return 0
        
    @property
    def attributesFeatures(self):
        """
        Number of features for every attribute.
        """
        return self._attributesFeatures
    
    
    
    @property
    def attributesAVGFeatureVariance(self):
        """
        Average variance of features for each attribute.
        """
        return self._attributesAVGFeatureVariance
    
    
class Experiment(Observable):
    """
    This class represents experiment.
    """
    
    DEFAULT_FEATURE_EXTRACTOR_NAME="Pass"
    """Name of default feature extractor that is set to attribute.
    If exists."""
    
    class AttributeSettings(Enum):
        """
        Possible settings types that could be set to an attribute.
        """
        USE=0
        PATH=1
        FEATURE_EXTRACTOR=2
        LABEL=3
        

    def __init__(self, filePath:str=None):
        """
        Creation of new experiment or loading of saved.
        
        :param filePath: Path to file. If None than new experiment is created, else
            saved experiment is loaded.
        :type filePath: str| None
        :raise RuntimeError: When there is problem with plugins.
        """
        super().__init__()
        self._dataset=None
        self._attributesSet={}
        self._label=None
        self._featuresSele=[]   
        self._classifiers=[]    #classifiers for testing
        self._evaluationMethod=None
        
        #let's load the plugins that are now available
        self._loadPlugins()
        #TODO: loading
        
        self._dataStats=None
        self._origDataStats=None
        
        self._attributesThatShouldBeUsedCache={}
        
    def useDataSubset(self):
        """
        Use only defined subset of data.
        Subset is defined by selected samples.
        Samples are selected according to constraints defined in dataStats.
        """
        if self._dataStats is not None:
            self._dataset.useSubset(None)#clear the old one
            subset=np.empty(self._dataStats.numberOfSamples)
            counters=copy.copy(self._dataStats.classSamples)

            cnt=0
            for i,sample in enumerate(self._dataset):
                l=sample[self._label]
                try:
                    if counters[l]>0:
                        counters[l]-=1
                        subset[cnt]=i
                        cnt+=1
                except KeyError:
                    #probably class that we want to omit
                    pass

            self.dataset.useSubset(subset)
        
    @property    
    def dataStats(self):
        """
        The data stats. Working copy of original data stats.
        
        :return: Actual stats.
        :rtype: ExperimentDataStatistics | None
        """
        return self._dataStats
    
    @property
    def origDataStats(self):
        """
        Original data stats. Maybe you are looking for working copy 
        of data stats that you can get with dataStats.
        
        :return: Original data stats.
        :rtype: ExperimentDataStatistics | None
        """
        return copy.copy(self._origDataStats)
    
    
    @Observable._event("NEW_DATA_STATS")
    def setDataStats(self, stats, actOnly=False):
        """
        Set the data stats. This method overrides working copy
        and original data stats.
        
        :param stats: New stats.
        :type stats: ExperimentDataStatistics
        :param actOnly: If true than overrides only working copy.
            If false than overrides original data to.
            If no original data was set (origData is None) than
            this parameter is ignored and origData is set too.
        :type actOnly: bool
        """

        self._dataStats=copy.deepcopy(stats)
        if self._origDataStats is None or not actOnly:
            self._origDataStats=stats
        else:
            #We must add classes that were filtered out.
            classSamples=self._dataStats.classSamples
            deactivate=[]
            for c in self._origDataStats.classes:
                if c not in classSamples:
                    #we set the max, but we must deactivate it
                    #The max is set because if user will decide
                    #that she/he wants to use this class, than
                    #we must set somu initial number of samples.
                    classSamples[c]=self._origDataStats.classSamples[c]
                    deactivate.append(c)
                    
            self._dataStats.classSamples=classSamples
            #lets deactivate it
            for c in deactivate:
                self._dataStats.deactivateClass(c)
    
    def _loadPlugins(self):
        """
        Loads available plugins.
        
        :raise RuntimeError: When there is problem with plugins.
        """
        #available features extractors
        if len(FEATURE_EXTRACTORS)==0:
            #TODO: Maybe multilanguage message will be better.
            raise RuntimeError("There are no features extractors plugins.")
        
        feTmp={}
        for fe in FEATURE_EXTRACTORS.values():
            if fe.getName() in feTmp:
                #wow, name collision
                #TODO: Maybe multilanguage message will be better.
                raise RuntimeError("Collision of features extractors names. For name: "+fe.getName())
            feTmp[fe.getName()]=fe
        
        #lets put the default feature extractor as the first if exists
        if self.DEFAULT_FEATURE_EXTRACTOR_NAME in feTmp:
            cont=[(self.DEFAULT_FEATURE_EXTRACTOR_NAME,feTmp[self.DEFAULT_FEATURE_EXTRACTOR_NAME])]
            #add the rest
            cont+=[(n,p) for n,p in feTmp.items() if n!=self.DEFAULT_FEATURE_EXTRACTOR_NAME]
            self._featuresExt=OrderedDict(cont)
        else:
            self._featuresExt=OrderedDict(feTmp)
        
        #available classifiers
        if len(CLASSIFIERS)==0:
            #TODO: Maybe multilanguage message will be better.
            raise RuntimeError("There are no classifiers plugins.")
        
        clsTmp=set()
        for cls in CLASSIFIERS.values():
            if cls.getName() in clsTmp:
                #wow, name collision
                #TODO: Maybe multilanguage message will be better.
                raise RuntimeError("Collision of classifiers names. For name: "+cls.getName())
            clsTmp.add(cls.getName())
            
        #available Validators
        self.availableEvaluationMethods = []
        stackValidators = [Validator]
        while len(stackValidators):
            base = stackValidators.pop()
            for child in base.__subclasses__():
                if child not in self.availableEvaluationMethods:
                    self.availableEvaluationMethods.append(child)
                    stackValidators.append(child)
                    
        self._evaluationMethod=self.availableEvaluationMethods[0]()  #add default
        
    @property
    def featuresSelectors(self):
        """
        Features selectors for feature selecting.
        """
        
        return [ s.plugin for s in self._featuresSele]
    
    @property
    def classifiers(self):
        """
        Classifiers for testing.
        """
        
        return [ s.plugin for s in self._classifiers]
    
    def newClassifierSlot(self):
        """
        Creates new slot for classifier that should be tested.
        
        :return: Classifier slot
        :rtype: PluginSlot
        """
        return self._addPluginSlot(self._classifiers)
    
    def removeClassifierSlot(self, slot:PluginSlot):
        """
        Remove classifier slot.
        
        :param slot: Slot for classifier.
        :type slot:PluginSlot
        """
        self._removePluginSlot(self._classifiers, slot)
        
    def newFeaturesSelectorSlot(self):
        """
        Creates new slot for features selector that should be tested.
        
        :return: Features selector slot
        :rtype: PluginSlot
        """
        return self._addPluginSlot(self._featuresSele)
    
    def removeFeaturesSelectorSlot(self, slot:PluginSlot):
        """
        Remove features selector slot.
        
        :param slot: Slot for features selector.
        :type slot: PluginSlot
        """
        self._removePluginSlot(self._featuresSele, slot)
            
    def _addPluginSlot(self, bank):
        """
        Creates new slot in given slot bank.
        
        :param bank: Slot bank
        :type bank: List[PluginSlot]
        :return: New slot
        :rtype: PluginSlot
        """
        #lets find first empty id
        slotId=0 if len(bank)==0 else max(p.id for p in bank)+1
        bank.append(PluginSlot(slotId))
        return bank[-1]
    
    def _removePluginSlot(self, bank:List[PluginSlot], slot:PluginSlot):
        """
        Creates new slot in given slot bank.
        
        :param bank: Slot bank
        :type bank: List[PluginSlot]
        :param slot: Slot that should be removed.
        :type slot: PluginSlot
        """
        bank.remove(slot)
    
    @property
    def availableClassifiers(self):
        """
        Available classifiers plugins.
        """
        return CLASSIFIERS

    @property
    def featuresExt(self):
        """
        Available features extractors plugins.
        Stored in OrderedDict (name -> plugin). Because it is handy to have default extractor as first
        (if exists). 
        """
        return self._featuresExt
        
    @Observable._event("NEW_DATA_SET")
    def loadDataset(self, filePath:str):
        """
        Loades dataset.
        
        :param filePath: Path to file with dataset.
        :type filePath: str
        """
        self._dataset=DataSet(filePath)
        #prepare new attribute settings
        self._attributesSet={
            name:{self.AttributeSettings.USE:True, self.AttributeSettings.PATH:False,
                  self.AttributeSettings.FEATURE_EXTRACTOR:next(iter(self._featuresExt.values()))()} 
                          for name in self._dataset.attributes}
        self._label=None
        self._dataStats=None
        self._attributesThatShouldBeUsedCache={}
        
    @property
    def evaluationMethod(self):
        """
        Validator used for evaluation.
        """
        return self._evaluationMethod
    
    @evaluationMethod.setter
    def evaluationMethod(self,val):
        """
        Validator used for evaluation.
        
        :param val: Validtor or name of validator class.
            If name of validator is provided than new  object of it's corresponding class is created.
        :type val:str|Validator
        :raise ValueError: When invalid value is given (unknown name).
        """
        if isinstance(val, Validator):
            self._evaluationMethod=val
        else:
            #self.availableEvaluationMethods is a list because we want to preserve order and therefore
            #we have no other choice than to iterate over it and find the right by name.
            for v in self.availableEvaluationMethods:
                if v.getName()==val:
                    self._evaluationMethod=v()
                    return
            
            raise ValueError("Unknown Validator name: "+val)
        
    def setEvaluationMethod(self,val):
        """
        Same as evaluationMethod but can be used as callable
        
        :param val: Validtor or name of validator class.
            If name of validator is provided than new  object of it's corresponding class is created.
        :type val:str|Validator
        :raise ValueError: When invalid value is given (unknown name).
        """
        self.evaluationMethod=val
        
    @property
    def label(self):
        """
        Attribute name that is set as label or None.
        """
        return self._label
    
    @label.setter
    def label(self, attribute:str):
        """
        Set new label attribute
        :param attribute: Name of attribute that should be used as new label.
        :type attribute: str | None
        """
        
        self._label=attribute
        
    def getAttributeSetting(self, attribute:str,t):
        """
        Get attribute setting of given type.
        
        :param attribute: The attribute.
        :type attribute: str
        :param t: The setting type.
        :type t: Experiment.AttributeSettings
        """
        if t == Experiment.AttributeSettings.LABEL:
            return self._label == attribute
        
        return self._attributesSet[attribute][t]
        
    @Observable._event("ATTRIBUTES_CHANGED")
    def attributesChangedEvent(self):
        """
        This event exists for informing observers that some attribute is no longer used
        or started to be used or when attribute is marked as label.
        
        """
        pass
        
    def setAttributeSetting(self, attribute:str,t, val):
        """
        Set attribute setting of given type.
        
        :param attribute: The attribute.
        :type attribute: str
        :param t: The setting type.
        :type t: Experiment.AttributeSettings
        :param val: New value.
        :type val: bool | Plugin
        :raise KeyError: When the name of attribute is uknown.
        """
        
        if t == Experiment.AttributeSettings.LABEL:
            self._label= attribute if val else None
        else:
            self._attributesSet[attribute][t]=val
            
        if t == Experiment.AttributeSettings.PATH:
            #we must inform the dataset object
            if val:
                self._dataset.addPathAttribute(attribute)
            else:
                self._dataset.removePathAttribute(attribute)
            
            
        if t==Experiment.AttributeSettings.USE or t==Experiment.AttributeSettings.LABEL:
            self._attributesThatShouldBeUsedCache={}
            self.attributesChangedEvent()

    def attributesThatShouldBeUsed(self, label:bool=True):
        """
        Names of attributes that should be used.
        
        :param label: True means that label attribute should be among them.
        :type label: bool
        """
        #we are preserving original attribute order
        try:
            return self._attributesThatShouldBeUsedCache[label]
        except KeyError:
            res=[a for a in self.dataset.attributes \
                if self._attributesSet[a][Experiment.AttributeSettings.USE] and (label or a!=self._label)]
            
            self._attributesThatShouldBeUsedCache[label]=res
        
            return res
    
    @property
    def dataset(self):
        """
        Loaded dataset.
        """
        return self._dataset

        
class ExperimentBackgroundWorker(QThread):
    numberOfSteps = Signal(int)
    """Signalizes that we now know the number of steps. Parameter is number of steps."""
    
    step = Signal()
    """Next step finished"""
    
    actInfo = Signal(str)
    """Sends information about what thread is doing now."""
    
    class MultPMessageType(Enum):
        """
        Message type for multiprocessing communication.
        """
        
        NUMBER_OF_STEPS_SIGNAL=0
        """Signalizes that we now know the number of steps. Value is number of steps (int)."""
        
        STEP_SIGNAL=1
        """Next step finished. Value None."""
        
        ACT_INFO_SIGNAL=2
        """Sends information about what process is doing now. Value is string."""
        
        
    
    def __init__(self, experiment:Experiment):
        """
        Initialization of background worker.
        
        :param experiment: Work on that experiment.
        :type experiment: Experiment
        """
        QThread.__init__(self)
        
        self._experiment=experiment
    

class ExperimentStatsRunner(ExperimentBackgroundWorker):
    """
    Runs the stats calculation in it's own thread.
    """

    calcStatsResult = Signal(ExperimentDataStatistics)
    """Sends calculated statistics."""
    
    def run(self):
        """
        Run the stat calculation.
        """
        statsExp=ExperimentDataStatistics()
        statsExp.attributes=self._experiment.attributesThatShouldBeUsed(False)
        #reading, samples, attributes
        self.numberOfSteps.emit(2+len(statsExp.attributes))
        
        #TODO: MULTILANGUAGE
        self.actInfo.emit("dataset reading") 
        if self.isInterruptionRequested():
            return
        #ok, lets first read the data
        data, labels=self._experiment.dataset.toNumpyArray([
            statsExp.attributes,
            [self._experiment.label]
            ])
        

        if data.shape[0]==0:
            #no data
            #TODO: ERROR MESSAGE FOR USER THIS IS TO FAST
            self.actInfo.emit("no data") 
            self.calcStatsResult.emit(statsExp)
            return
        
        labels=labels.ravel()   #we need row vector   
        
        self.step.emit()     
        self.actInfo.emit("samples counting") 
        if self.isInterruptionRequested():
            return
        
        classSamples={}
        
        classes, samples=np.unique(labels, return_counts=True)
        
        for actClass, actClassSamples in zip(classes, samples):
            if self.isInterruptionRequested():
                return
            classSamples[actClass]=actClassSamples
 
        statsExp.classSamples=classSamples
        #extractors mapping
        extMap=[(a, self._experiment.getAttributeSetting(a, Experiment.AttributeSettings.FEATURE_EXTRACTOR)) \
                for a in self._experiment.attributesThatShouldBeUsed(False)]
        
        self.step.emit() 
        
        #get the attributes values
        for i,(attr,extractor) in enumerate(extMap):
            self.actInfo.emit("attribute: "+attr) 
            if self.isInterruptionRequested():
                return
            actF=extractor.fitAndExtract(data[:,i],labels).tocsc()
            if self.isInterruptionRequested():
                return
            statsExp.attributesFeatures[attr]=actF.shape[1]
            statsExp.attributesAVGFeatureVariance[attr]=np.average(np.array([sparseMatVariance(actF[:,c]) for c in range(actF.shape[1])]))
            self.step.emit() 
            
        self.actInfo.emit("Done") 
        
        self.calcStatsResult.emit(statsExp)
        

        
class ExperimentRunner(ExperimentBackgroundWorker):
    """
    Runs experiment in it's own thread.
    """
        
    def run(self):
        """
        Run the experiment.
        """
        commQ = multiprocessing.Queue()
        p = multiprocessing.Process(target=partial(self.work, self._experiment, commQ))
        p.start()
        while not self.isInterruptionRequested() and p.is_alive():
            try:
                msgType, msgValue=commQ.get(False, 0.5)#nonblocking
                
                if msgType==self.MultPMessageType.NUMBER_OF_STEPS_SIGNAL:
                    self.numberOfSteps.emit(msgValue)
                elif msgType==self.MultPMessageType.STEP_SIGNAL:
                    self.step.emit()
                elif msgType==self.MultPMessageType.ACT_INFO_SIGNAL:
                    self.actInfo.emit(msgValue)
                
            except queue.Empty:
                #nothing here
                pass
            
        if p.is_alive() and self.isInterruptionRequested():
            p.terminate()
        
    @classmethod
    def work(cls, experiment:Experiment, commQ:multiprocessing.Queue):
        """
        The actual work of that thread
        
        :param experiment: Work on that experiment.
        :type experiment: Experiment
        :param commQ: Communication queue.
        :type commQ: multiprocessing.Queue
        """
        #TODO: MULTILANGUAGE
        commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL,"dataset reading"))

        #ok, lets first read the data
        data, labels=experiment.dataset.toNumpyArray([
            experiment.attributesThatShouldBeUsed(False),
            [experiment.label]
            ])
        
        if data.shape[0]==0:
            #no data
            #TODO: ERROR MESSAGE FOR USER THIS IS TO FAST
            commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL,"no data"))
            return
        
        labels=labels.ravel()   #we need row vector       
        #extractors mapping
        extMap=[experiment.getAttributeSetting(a, Experiment.AttributeSettings.FEATURE_EXTRACTOR) \
                for a in experiment.attributesThatShouldBeUsed(False)]
        
    
        
        #create storage for results
        steps=experiment.evaluationMethod.numOfSteps(data,labels)
        
        commQ.put((cls.MultPMessageType.NUMBER_OF_STEPS_SIGNAL,    #+1 reading
                   len(experiment.classifiers)*(steps)+1))

        #TODO catch memory error
        resultsStorage=Results(steps)

        commQ.put((cls.MultPMessageType.STEP_SIGNAL,None))
        for c in experiment.classifiers:
            print(c.getName(), ", ".join( a.name+"="+str(a.value.getName()+":"+", ".join([pa.name+"->"+str(pa.value) for pa in a.value.getAttributes()]) if isinstance(a.value,Plugin) else a.value) for a in c.getAttributes()))
            commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL,
                       "testing {} {}/{}".format(c.getName(), 1, steps)))

            start = time.time()
            startProc=time.process_time()
            for step, (predicted, realLabels, stepTimes, stats) in enumerate(experiment.evaluationMethod.run(c, data, labels, extMap, experiment.featuresSelectors)):
                endProc=time.process_time()
                end = time.time()
                
                if resultsStorage.steps[step].labels is None:
                    #because it does not make much sense to have true labels stored for each predictions
                    #we store labels just once for each validation step
                    resultsStorage.steps[step].labels=realLabels
                resultsStorage.steps[step].addResults(c, predicted)
                commQ.put((cls.MultPMessageType.STEP_SIGNAL,None))
                if step+2<=steps:
                    #TODO: MULTILANGUAGE
                    commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL,
                               "testing {} {}/{}".format(c.getName(), step+2, steps)))
                
                cls.writeConfMat(predicted, realLabels)

                print(classification_report(realLabels, predicted))
                
                print("accuracy\t{}".format(accuracy_score(realLabels, predicted)))
                
                print("Samples for training:", stats[Validator.SamplesStats.NUM_SAMPLES_TRAIN])
                print("Samples for testing:", stats[Validator.SamplesStats.NUM_SAMPLES_TEST])
                print("Number of features:", stats[Validator.SamplesStats.NUM_FEATURES])
                
                print("Feature extraction time for train set:",
                      stepTimes[Validator.TimeDuration.FEATURE_EXTRACTION_TRAIN])
                print("Feature extraction process time for train set:",
                      stepTimes[Validator.TimeDuration.FEATURE_EXTRACTION_TRAIN_PROC])
                
                print("Feature extraction time for test set:",
                      stepTimes[Validator.TimeDuration.FEATURE_EXTRACTION_TEST])
                print("Feature extraction process time for test set:",
                      stepTimes[Validator.TimeDuration.FEATURE_EXTRACTION_TEST_PROC])
                
                print("Feature extraction time for train and test set:",
                      stepTimes[Validator.TimeDuration.FEATURE_EXTRACTION_TRAIN]
                      +
                      stepTimes[Validator.TimeDuration.FEATURE_EXTRACTION_TEST])
                print("Feature extraction process time train and test set:",
                      stepTimes[Validator.TimeDuration.FEATURE_EXTRACTION_TRAIN_PROC]
                      +
                      stepTimes[Validator.TimeDuration.FEATURE_EXTRACTION_TEST_PROC])
                
                print("---")
                print("Feature selection time for train set:",
                      stepTimes[Validator.TimeDuration.FEATURE_SELECTION_TRAIN])
                print("Feature selection process time for train set:",
                      stepTimes[Validator.TimeDuration.FEATURE_SELECTION_TRAIN_PROC])
                
                print("Feature selection time for test set:",
                      stepTimes[Validator.TimeDuration.FEATURE_SELECTION_TEST])
                print("Feature selection process time for test set:",
                      stepTimes[Validator.TimeDuration.FEATURE_SELECTION_TEST_PROC])
                
                print("Feature selection time for train and test set:",
                      stepTimes[Validator.TimeDuration.FEATURE_SELECTION_TRAIN]
                      +
                      stepTimes[Validator.TimeDuration.FEATURE_SELECTION_TEST])
                print("Feature selection process time train and test set:",
                      stepTimes[Validator.TimeDuration.FEATURE_SELECTION_TRAIN_PROC]
                      +
                      stepTimes[Validator.TimeDuration.FEATURE_SELECTION_TRAIN_PROC])          
                print("---")
                print("Train time:",
                      stepTimes[Validator.TimeDuration.TRAINING])
                print("Train process time:",
                      stepTimes[Validator.TimeDuration.TRAINING_PROC])
                
                print("Test time:",
                      stepTimes[Validator.TimeDuration.TEST])
                print("Test process time:",
                      stepTimes[Validator.TimeDuration.TEST_PROC])
                
                print("Step time:",end-start)
                print("Step process time:",endProc-startProc)
                print("\n\n")
                start = time.time()
                startProc=time.process_time()
        
    @staticmethod
    def writeConfMat(predicted, labels):

        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', len(max(predicted, key=len)) if len(max(predicted, key=len))>len(max(labels, key=len)) else len(max(labels, key=len)))
        
        print(str(pd.crosstab(pd.Series(labels), pd.Series(predicted), rownames=['Real'], colnames=['Predicted'], margins=True)))
