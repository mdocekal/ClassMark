"""
Created on 18. 2. 2019
Module for experiment representation and actions.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from ..data.data_set import DataSet
from enum import Enum, auto
import time
from _collections import OrderedDict
from ..core.plugins import Plugin, CLASSIFIERS, FEATURE_EXTRACTORS
from ..core.validation import Validator
from PySide2.QtCore import QThread, Signal
from .results import Results
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import multiprocessing
import queue
import copy
import statistics 
from functools import partial
from .utils import getAllSubclasses, sparseMatVariance, Observable,Logger,Singleton
from .selection import FeaturesSelector
import pickle
from classmark.core.plugins import Classifier
from sklearn.preprocessing import LabelEncoder
import traceback
import os


class LastUsedExperiments(Observable, metaclass=Singleton):
    """
    Last used experiments manager. (singleton)
    """
    
    DATA_FILE_PATH=os.path.join(os.path.dirname(os.path.realpath(__file__)), "data","lastUsed")
    MAX_IN_HISTORY=100

    def __init__(self):
        """
        Initialization of singleton instance
        """
        super().__init__()
        self._list=[]
        if os.path.exists(self.DATA_FILE_PATH):
            with open(self.DATA_FILE_PATH, "r+") as file:
                filtered=False
                for line in file:
                    p=line.rstrip('\n')
                    if os.path.exists(p):
                        self._list.append(p)
                    else:
                        #filtered
                        filtered=True
                
                if filtered:
                    #write changes
                    self._listChange()
      
    @property  
    def list(self):
        """
        Get list of last used experiments.
        """
        return self._list
    
    def _listChange(self):
        """
        Update list content in file.
        """
        with open(self.DATA_FILE_PATH, "w") as f:
            for p in self._list:
                print(p, file=f)
    
    @Observable._event("CHANGE")
    def used(self, pathToExp:str):
        """
        Inform that experiment on given path was used.
        
        :param pathToExp: Path of given experiment.
        :type pathToExp: str
        """
        
        try:
            self._list.insert(0, self._list.pop(self._list.index(pathToExp)))
        except ValueError:
            #not in list yet
            self._list.insert(0,pathToExp)
            
            if len(self._list)>self.MAX_IN_HISTORY:
                del self._list[-1]
                
        #list changed
        self._listChange()
        
        
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
        self._attributesAVGFeatureSD={}
        
    
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
        self._attributesAVGFeatureSD={}
        
        for a in attr:
            self._attributesFeatures[a]=0
            self._attributesAVGFeatureSD[a]=0
            
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
    def attributesAVGFeatureSD(self):
        """
        Average standard deviation of features for each attribute.
        """
        return self._attributesAVGFeatureSD
    
class ExperimentLoadException(Exception):
    """
    There are some troubles when we are loading experiment.
    """
    pass    

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
        :raise RuntimeError: When there is a problem with plugins.
        :raise ExperimentLoadException: When there is a problem with loading.
        """
        super().__init__()
        
        self._dataset=None
        self._attributesSet={}
        self._label=None
        self._featuresSele=[]   
        self._classifiers=[]    #classifiers for testing
        self._evaluationMethod=None
        self.loadSavePath=None  #stores path from which this exp was loaded or where is saved
        self.results=None
            
        #let's load the plugins that are now available
        #must be called before experiment loading
        #because sets default values
        self._loadPlugins()
        
        if filePath is not None:
            #load saved experiment
            self._load(filePath)
            self.loadSavePath=filePath

        
        self._dataStats=None
        self._origDataStats=None
        
        self._attributesThatShouldBeUsedCache={}
        
    def save(self, filePath):
        """
        Saves experiment configuration to given file.
        
        :param filePath: Path to experiment file.
        :type filePath: str
        """
        with open(filePath,"wb") as saveF:
            #let's create Experiment version for saving
            data={
                "dataSet":self._dataset,
                "attributesSet":self._attributesSet,
                "label":self._label,
                "featuresSele":self._featuresSele,
                "classifiers":self._classifiers,
                "evaluationMethod":self._evaluationMethod,
                "results":self.results
                }
            #save it
            pickle.dump(data,saveF)
            self.loadSavePath=filePath
            
            LastUsedExperiments().used(filePath)
            
    def setResults(self, r):
        """
        Sets results. Suitable for use as callback.
        
        :param r: new results.
        :type r: Results
        """
        self.results=r
    
    def _load(self, filePath):
        """
        Loads saved experiment configuration from given file.
        
        :param filePath: Path to experiment file.
        :type filePath: str
        :raise ExperimentLoadException: When there is a problem with loading.
        """
        with open(filePath,"rb") as loadF:
            try:
                lE=pickle.load(loadF)
            except:
                raise ExperimentLoadException("Couldn't load given experiment.")
            
            if not isinstance(lE, dict):
                raise ExperimentLoadException("Couldn't load given experiment.")
            
            #check that we have loaded all attributes
            
            for a in ["dataSet","attributesSet","label", \
                      "featuresSele","classifiers","evaluationMethod"]:
                if a not in lE:
                    raise ExperimentLoadException("Couldn't load given experiment.")
                
            if not isinstance(lE["dataSet"], DataSet):
                raise ExperimentLoadException("Couldn't load given experiment.")
            
            self._dataset=lE["dataSet"]
            
            if not isinstance(lE["attributesSet"], dict):
                raise ExperimentLoadException("Couldn't load given experiment.")
                
            self._attributesSet=lE["attributesSet"]
            
            if not isinstance(lE["label"], str) and lE["label"] is not None :
                raise ExperimentLoadException("Couldn't load given experiment.")
            
            self._label=lE["label"]
            
            
            if not isinstance(lE["featuresSele"], list) and \
                any(not isinstance(fs, FeaturesSelector) for fs in lE["featuresSele"]):
                raise ExperimentLoadException("Couldn't load given experiment.")
            
            self._featuresSele=lE["featuresSele"]
            
            
            if not isinstance(lE["classifiers"], list) and \
                any(not isinstance(c, Classifier) for c in lE["classifiers"]):
                raise ExperimentLoadException("Couldn't load given experiment.")
            
            self._classifiers=lE["classifiers"]
            
            
            
            if not isinstance(lE["evaluationMethod"], Validator):
                raise ExperimentLoadException("Couldn't load given experiment.")
            self._evaluationMethod=lE["evaluationMethod"]
            
            if lE["results"] is not None and not isinstance(lE["results"], Results):
                raise ExperimentLoadException("Couldn't load given experiment.")
            self.results=lE["results"]
            
            LastUsedExperiments().used(filePath)

    def useDataSubset(self):
        """
        Use only defined subset of data.
        Subset is defined by selected samples.
        Samples are selected according to constraints defined in dataStats.
        """
        self._dataset.useSubset(None)#clear the old one
        if self._dataStats is not None:
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
        return self._origDataStats
    
    
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
        Adds default.
        
        :raise RuntimeError: When there is problem with plugins.
        """
        #available features extractors
        if len(FEATURE_EXTRACTORS)==0:
            raise RuntimeError("There are no features extractors plugins.")
        
        feTmp={}
        for fe in FEATURE_EXTRACTORS.values():
            if fe.getName() in feTmp:
                #wow, name collision
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
            raise RuntimeError("There are no classifiers plugins.")
        
        clsTmp=set()
        for cls in CLASSIFIERS.values():
            if cls.getName() in clsTmp:
                #wow, name collision
                raise RuntimeError("Collision of classifiers names. For name: "+cls.getName())
            clsTmp.add(cls.getName())
            
        #available Validators
        self.availableEvaluationMethods = getAllSubclasses(Validator)
                    
        self._evaluationMethod=self.availableEvaluationMethods[0]()  #add default
        
        #available Features selectors
        self.availableFeatureSelectors = getAllSubclasses(FeaturesSelector)
        
    @property
    def featuresSelectors(self):
        """
        Features selectors for feature selecting.
        """
        
        return [ s.plugin for s in self._featuresSele]
    @property
    def featuresSelectorsSlots(self):
        """
        All used features selectors slots.
        """
        
        return self._featuresSele
    
    @property
    def classifiersSlots(self):
        """
        All curently used classifiers slots.
        """
        return self._classifiers
    
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
        :param val: New value. For setting new label val must be true, because if you pass false than
        label will be set to None.
        :type val: bool | Plugin
        :raise KeyError: When the name of attribute is uknown.
        """
        
        if t == Experiment.AttributeSettings.LABEL:
            self._label= attribute if val else None
            #setting new label invalidates data stats
            self.setDataStats(None)
        else:
            self._attributesSet[attribute][t]=val
            
        if t == Experiment.AttributeSettings.PATH:
            #we must inform the data set object
            if val:
                self._dataset.addPathAttribute(attribute,
                    self._attributesSet[attribute][Experiment.AttributeSettings.FEATURE_EXTRACTOR].expDataType())
            else:
                self._dataset.removePathAttribute(attribute)
                
        if t ==Experiment.AttributeSettings.FEATURE_EXTRACTOR and \
            attribute in self._dataset.pathAttributes:
            
            #we must inform the data set object
            self._dataset.addPathAttribute(attribute, val.expDataType())
            
            
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

        
class ExperimentBackgroundRunner(QThread):
    """
    Base class for background tasks.
    Defines all mandatory signals.
    """
    
    numberOfSteps = Signal(int)
    """Signalizes that we now know the number of steps. Parameter is number of steps."""
    
    step = Signal()
    """Next step finished"""
    
    actInfo = Signal(str)
    """Sends information about what thread is doing now."""
    
    error = Signal(str,str)
    """Sends information about error that cancels background worker.
    First string is short error message and second is detailed error description."""
    
    log= Signal(str)
    """Sends log message that should be shown in GUI."""
    
    class MultPMessageType(Enum):
        """
        Message type for multiprocessing communication.
        """
        
        NUMBER_OF_STEPS_SIGNAL=auto()
        """Signalizes that we now know the number of steps. Value is number of steps (int)."""
        
        STEP_SIGNAL=auto()
        """Next step finished. Value None."""
        
        ACT_INFO_SIGNAL=auto()
        """Sends information about what process is doing now. Value is string."""
        
        LOG_SIGNAL=auto()
        """Sends log message that should be shown in GUI."""
        
        RESULT_SIGNAL=auto()
        """Sends experiment results."""
        
        ERROR_SIGNAL=auto()
        """Sends information about error that cancels background worker.
        To queue pass tuple with message and detailed message in that order.
        If you want to pass just message, than pass it just like regular string."""
        
    
    def __init__(self, experiment:Experiment):
        """
        Initialization of background worker.
        
        :param experiment: Work on that experiment.
        :type experiment: Experiment
        """
        QThread.__init__(self)
        
        self._experiment=copy.copy(experiment)
        #remove thinks that are no longer useful
        self._experiment.clearObservers()
        
        if self._experiment.dataStats is not None:
            self._experiment.dataStats.clearObservers()
            
        if self._experiment.origDataStats is not None:
            self._experiment.origDataStats.clearObservers()
        
        
    def run(self):
        """
        Run the background work.
        Default implementation that creates new process with which communicates via queue.
        The true work must be implemented in work method.
        """
        
        try:
            commQ = multiprocessing.Queue()
            p = multiprocessing.Process(target=partial(self.work, self._experiment, commQ))
            p.start()
            while not self.isInterruptionRequested() and p.is_alive():
                try:
                    msgType, msgValue=commQ.get(True, 0.5)#blocking
                    self.processMultPMsg(msgType, msgValue)
                except queue.Empty:
                    #nothing here
                    pass
                
            if p.is_alive():
                p.terminate()
                
            while True:#is something still in queue?
                try:
                    msgType, msgValue=commQ.get(True, 0.5)#blocking
                    self.processMultPMsg(msgType, msgValue)
                except queue.Empty:
                    #nothing here
                    break
            
        except Exception as e:
            #error
            self.error.emit(str(e), traceback.format_exc())
            
    @classmethod
    def work(cls, experiment:Experiment, commQ:multiprocessing.Queue):
        """
        The actual work of that thread
        
        :param experiment: Work on that experiment.
        :type experiment: Experiment
        :param commQ: Communication queue.
        :type commQ: multiprocessing.Queue
        """
        
        raise NotImplemented("Please implement the work method")
        
        
    def processMultPMsg(self,msgType,msgVal:Any):
        """
        Processes message received from another process.
        Sends appropriate signals to UI thread.
        
        :param msgType: Type of received message.
        :type msgType: MultPMessageType
        :param msgVal: The message.
        :type msgVal: Any
        :return: Returns True if emits an signal. False otherwise.
        :rtype: bool
        """

        if msgType==self.MultPMessageType.NUMBER_OF_STEPS_SIGNAL:
            self.numberOfSteps.emit(msgVal)
        elif msgType==self.MultPMessageType.STEP_SIGNAL:
            self.step.emit()
        elif msgType==self.MultPMessageType.ACT_INFO_SIGNAL:
            self.actInfo.emit(msgVal)
        elif msgType==self.MultPMessageType.LOG_SIGNAL:
            self.log.emit(msgVal)
        elif msgType==self.MultPMessageType.ERROR_SIGNAL:
            if isinstance(msgVal, str):
                self.error.emit(msgVal, None)
            else:
                self.error.emit(msgVal[0],msgVal[1])
        else:
            return False
        return True
    

class ExperimentStatsRunner(ExperimentBackgroundRunner):
    """
    Runs the stats calculation in it's own thread.
    """

    calcStatsResult = Signal(ExperimentDataStatistics)
    """Sends calculated statistics."""
    
    
    

    @classmethod
    def work(cls, experiment:Experiment, commQ:multiprocessing.Queue):
        """
        Stats calculation.
        
        :param experiment: Work on that experiment.
        :type experiment: Experiment
        :param commQ: Communication queue.
        :type commQ: multiprocessing.Queue
        """
        
        try:
            experiment.useDataSubset()
            statsExp=ExperimentDataStatistics()
            statsExp.attributes=experiment.attributesThatShouldBeUsed(False)

            #reading, samples, attributes
            
            commQ.put((cls.MultPMessageType.NUMBER_OF_STEPS_SIGNAL, 2+len(statsExp.attributes)))

            commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL, "dataset reading"))
            
            #ok, lets first read the data
            data, labels=experiment.dataset.toNumpyArray([
                statsExp.attributes,
                [experiment.label]
                ])

            if data.shape[0]==0:
                #no data
                commQ.put((cls.MultPMessageType.ERROR_SIGNAL, ("no data", "Given data set does not have any samples.")))
                return
            
            labels=labels.ravel()   #we need row vector   
            
            commQ.put((cls.MultPMessageType.STEP_SIGNAL, None))
 
            commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL, "samples counting"))
            
            
            classSamples={}
            
            classes, samples=np.unique(labels, return_counts=True)
            
            for actClass, actClassSamples in zip(classes, samples):
                classSamples[actClass]=actClassSamples
     
            statsExp.classSamples=classSamples
            #extractors mapping
            extMap=[(a, experiment.getAttributeSetting(a, Experiment.AttributeSettings.FEATURE_EXTRACTOR)) \
                    for a in experiment.attributesThatShouldBeUsed(False)]
            
            commQ.put((cls.MultPMessageType.STEP_SIGNAL, None))
            
            #get the attributes values
            for i,(attr,extractor) in enumerate(extMap):
                commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL, "attribute: "+attr))

                actF=extractor.fitAndExtract(data[:,i],labels).tocsc()

                statsExp.attributesFeatures[attr]=actF.shape[1]
                statsExp.attributesAVGFeatureSD[attr]=np.average(np.array([(sparseMatVariance(actF[:,c]))**0.5 for c in range(actF.shape[1])]))
                commQ.put((cls.MultPMessageType.STEP_SIGNAL, None))
                
            commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL, "Done"))
            commQ.put((cls.MultPMessageType.RESULT_SIGNAL, statsExp))

        except Exception as e:
            #error
            commQ.put((cls.MultPMessageType.ERROR_SIGNAL, (str(e),traceback.format_exc())))
        finally:
            commQ.close()
            commQ.join_thread()
            
    def processMultPMsg(self,msgType,msgVal:Any):
        """
        Processes message received from another process.
        Sends appropriate signals to UI thread.
        
        :param msgType: Type of received message.
        :type msgType: MultPMessageType
        :param msgVal: The message.
        :type msgVal: Any
        :return: Returns True if emits an signal. False otherwise.
        :rtype: bool
        """
        
        if not super().processMultPMsg(msgType,msgVal):
            if msgType==self.MultPMessageType.RESULT_SIGNAL:
                self.calcStatsResult.emit(msgVal)
            else:
                return False
            return True
                
        return False

        
class ExperimentRunner(ExperimentBackgroundRunner):
    """
    Runs experiment in it's own thread.
    
    """

    result= Signal(Results)
    """Send signal with experiment results."""
        
    @classmethod
    def work(cls, experiment:Experiment, commQ:multiprocessing.Queue):
        """
        The actual work of that thread.
        
        :param experiment: Work on that experiment.
        :type experiment: Experiment
        :param commQ: Communication queue.
        :type commQ: multiprocessing.Queue
        """
        try:
            experiment.useDataSubset()
            #remove thinks that are no longer useful
            experiment.setDataStats(None)
            
            
            commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL,"dataset reading"))
    
            logger=Logger() #get singleton instance of logger
            
            #reg event
            logger.registerObserver("LOG", 
                lambda logMsg: commQ.put((cls.MultPMessageType.LOG_SIGNAL, logMsg)))
            
            #ok, lets first read the data
            data, labels=experiment.dataset.toNumpyArray([
                experiment.attributesThatShouldBeUsed(False),
                [experiment.label]
                ])
            
            if data.shape[0]==0:
                #no data
                commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL,"no data"))
                return
            
            labels=labels.ravel()   #we need row vector       
            lEnc=LabelEncoder()
            #let's encode labels to save some memory space
            #also this is more suitable representation for some classifiers such as neural networks
            labels=lEnc.fit_transform(labels)
            
            
            #extractors mapping
            extMap=[experiment.getAttributeSetting(a, Experiment.AttributeSettings.FEATURE_EXTRACTOR) \
                    for a in experiment.attributesThatShouldBeUsed(False)]
            
        
            
            #create storage for results
            steps=experiment.evaluationMethod.numOfSteps(experiment.dataset,data,labels)
            
            commQ.put((cls.MultPMessageType.NUMBER_OF_STEPS_SIGNAL,    
                       (len(experiment.classifiers)*2+experiment.evaluationMethod.NUMBER_OF_FEATURES_STEP)*(steps)+1))
                    #+1 reading
                    #len(experiment.classifiers)*2    one step for testing and one for training of one classifier
            
            
            commQ.put((cls.MultPMessageType.STEP_SIGNAL,None))  #because reading is finished
    
            resultsStorage=Results(steps,experiment.classifiers,lEnc)
    
            
            for step, c, predicted, realLabels, testIndices, stepTimes, stats in experiment.evaluationMethod.run(experiment.dataset,experiment.classifiers, 
                                data, labels, extMap, experiment.featuresSelectors, cls.nextSubStep(commQ)):
                if resultsStorage.steps[step].labels is None:
                    #because it does not make much sense to have true labels stored for each predictions
                    #we store labels just once for each validation step
                    resultsStorage.steps[step].labels=realLabels
                    
                resultsStorage.steps[step].addResults(c, predicted, testIndices, stepTimes, stats)
       
                
                transRealLabels=lEnc.inverse_transform(realLabels)
                transPredictedLabels=lEnc.inverse_transform(predicted)
                cls.writeConfMat(transPredictedLabels, transRealLabels)
    
                logger.log(classification_report(transRealLabels, 
                                                 transPredictedLabels))
                
                logger.log("accuracy\t{}".format(accuracy_score(realLabels, predicted)))
                logger.log("\n\n")
            resultsStorage.finalize()   #for better score calculation
            commQ.put((cls.MultPMessageType.RESULT_SIGNAL,resultsStorage))
            
        except Exception as e:
            #error
            commQ.put((cls.MultPMessageType.ERROR_SIGNAL, (str(e), traceback.format_exc())))
        finally:
            commQ.close()
            commQ.join_thread()
    @classmethod
    def nextSubStep(cls,commQ:multiprocessing.Queue):
        """
        Informs UI about next substep.
        
        :param commQ: Communication queue.
        :type commQ: multiprocessing.Queue
        """   
        
        def x(msg):
            commQ.put((cls.MultPMessageType.ACT_INFO_SIGNAL,msg))
            commQ.put((cls.MultPMessageType.STEP_SIGNAL,None))
        return x
            
    
    def processMultPMsg(self,msgType,msgVal:Any):
        """
        Processes message received from another process.
        Sends appropriate signals to UI thread.
        
        :param msgType: Type of received message.
        :type msgType: MultPMessageType
        :param msgVal: The message.
        :type msgVal: Any
        :return: Returns True if emits an signal. False otherwise.
        :rtype: bool
        """
        if not super().processMultPMsg(msgType,msgVal):
            if  msgType==self.MultPMessageType.RESULT_SIGNAL:

                self.result.emit(msgVal)
            else:
                return False
            return True
                
        return False
            
    @staticmethod
    def writeConfMat(predicted, labels):

        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', len(max(predicted, key=len)) if len(max(predicted, key=len))>len(max(labels, key=len)) else len(max(labels, key=len)))
        
        Logger().log(str(pd.crosstab(pd.Series(labels), pd.Series(predicted), rownames=['Real'], colnames=['Predicted'], margins=True)))
