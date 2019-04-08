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
from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from multiprocessing import Process
from sklearn.feature_selection import chi2
from ..core.utils import sparseMatVariance
import copy

class ClassifierSlot(object):
    """
    Slot that stores informations about classifier that should be tested. 
    """
    
    def __init__(self, slotID):
        """
        Creates empty classifier slot
        
        :param slotID: Unique identifier of slot.
        :type slotID: int
        """
        self._id=slotID
        self.classifier=None
    
    def __eq__(self, other):
        if not isinstance(other, __class__):
            return False
        return self._id==other._id
        
    def __hash__(self):
        return self._id
    
    
class ExperimentDataStatistics(object):
    """
    Data statistics of experiment.
    """
    
    def __init__(self):
        self._classSamples={}
        self._classActivity={}
        self._classes=[]    #just to give order to the classes
        self._attributesFeatures={}
        self._attributesAVGFeatureVariance={}
        self._attributesAVGFeatureChiSquare={}
        
    def isActive(self,c):
        """
        Determines if given class is active.
        
        :param c: Class for check.
        :type c: Any
        :return: True active. False otherwise.
        :rtype: bool
        """
        
        return self._classActivity[c]
    
    def deactivateClass(self,c):
        """
        Deactivating a class means that this class is hidden and is no longer
        in properties like classes, classSamples,numberOfSamples, numberOfSamples,
        maxSamplesInClass and minSamplesInClass.
        
        :param c: Class from classes.
        :type c: Any
        """
        
        self._classActivity[c]=False
        
    def activateClass(self, c):
        """
        Activate class c.
        
        :param c: Class from classes.
        :type c: Any
        """
        self._classActivity[c]=True
        
    @property
    def classes(self):
        """
        Gives list of all classes.
        
        :return: classes
        :rtype: List[Any]
        """
        
        return [c for c in self._classes  if self._classActivity[c]]
        
    @property
    def classSamples(self):
        """
        Number of samples per class in form of dict where key is the class and value
        is the number of samples.
        """
        return { c:v for c,v in self._classSamples.items() if self._classActivity[c] }
    
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
        
    def changeSamplesInClass(self,c,n):
        """
        Changes number of samples per class.
        
        :param c: The class.
        :type c: Any
        :param n: New number of samples in class.
        :type n: int
        """
        self._classSamples[c]=n
        
    @property
    def numberOfSamples(self):
        """
        Number of samples in whole data set.
        Is calculated from classSamples.
        """
        
        return sum( v for c,v in self._classSamples if self._classActivity[c])
        
    @property
    def maxSamplesInClass(self):
        """
        Gives maximal number of samples in class and also returns the class itself.
        
        :return: maximal number of samples in class and that class
        :rtype: Tuple[int, Any]
        """
        
        mC=max([ c for c in self._classSamples if self._classActivity[c]], 
                key=lambda x: self._classSamples[x])
        return (self._classSamples[mC],mC)
    
    @property
    def minSamplesInClass(self):
        """
        Gives minimal number of samples in class and also returns the class itself.
        
        :return: minimal number of samples in class and that class
        :rtype: Tuple[int, Any]
        """
        
        mC=min([ c for c in self._classSamples if self._classActivity[c]], 
                key=lambda x: self._classSamples[x])
        return (self._classSamples[mC],mC)
        
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
    
    @property
    def attributesAVGFeatureChiSquare(self):
        """
        Average Chi^2 of features for each attribute.
        """
        return self._attributesAVGFeatureChiSquare
    
class Experiment(object):
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
        self._dataset=None
        self._attributesSet={}
        self._label=None
        self._classifiers=[]    #classifiers for testing
        self._evaluationMethod=None
        
        #let's load the plugins that are now available
        self._loadPlugins()
        #TODO: loading
        
        self._dataStats=None
        self._origDataStats=None
        
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
    
    def setDataStats(self, stats):
        """
        Set the data stats. This method overrides working copy
        and original data stats.
        
        :param stats: New stats.
        :type stats: ExperimentDataStatistics
        """
        
        self._dataStats=copy.deepcopy(stats)
        self._origDataStats=stats
        
        
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
    def classifiers(self):
        """
        Classifiers for testing.
        """
        
        return [ c.classifier for c in self._classifiers if c is not None]
    
    def newClassifierSlot(self):
        """
        Creates new slot for classifier that should be tested.
        
        :return: Classifier slot
        :rtype: ClassifierSlot
        """
        #lets find first empty id
        slotId=len(self._classifiers)
        for i, x in enumerate(self._classifiers):
            if x is None:
                #there is empty id
                self._classifiers[i]=ClassifierSlot(i)
                return self._classifiers[i]
            
        self._classifiers.append(ClassifierSlot(slotId))
        return self._classifiers[-1]
    
    def removeClassifierSlot(self, slot:ClassifierSlot):
        """
        Remove classifier slot.
        
        :param slot: Slot fot classifier.
        :type slot:ClassifierSlot
        """

        i=self._classifiers.index(slot)
        self._classifiers[i]=None
        
        #remove all None from the end of list
        for i in range(len(self._classifiers)-1,-1,-1):
            if self._classifiers[i] is None:
                self._classifiers.pop()
            else:
                break
    
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
            
            
        

    def attributesThatShouldBeUsed(self, label:bool=True):
        """
        Names of attributes that should be used.
        
        :param label: True means that label attribute should be among them.
        :type label: bool
        """
        #we are preserving original attribute order
        return [a for a in self.dataset.attributes \
                if self._attributesSet[a][Experiment.AttributeSettings.USE] and (label or a!=self._label)]
    
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
        #reading, samples, attributes
        self.numberOfSteps.emit(3)
        
        #TODO: MULTILANGUAGE
        self.actInfo.emit("dataset reading") 
        if self.isInterruptionRequested():
            return
        #ok, lets first read the data
        data, labels=self._experiment.dataset.toNumpyArray([
            self._experiment.attributesThatShouldBeUsed(False),
            [self._experiment.label]
            ])
        
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
        self.actInfo.emit("attributes") 
        #get the attributes values
        for i,(attr,extractor) in enumerate(extMap):
            if self.isInterruptionRequested():
                return
            actF=extractor.fitAndExtract(data[:,i],labels).tocsc()
            if self.isInterruptionRequested():
                return
            statsExp.attributesFeatures[attr]=actF.shape[1]
            statsExp.attributesAVGFeatureChiSquare[attr]=np.average(chi2(actF, labels))
            if self.isInterruptionRequested():
                return
            statsExp.attributesAVGFeatureVariance[attr]=np.average(np.array([sparseMatVariance(actF[:,c]) for c in range(actF.shape[1])]))
            

        self.step.emit() 
        
        self.calcStatsResult.emit(statsExp)
        

        
class ExperimentRunner(ExperimentBackgroundWorker):
    """
    Runs experiment in it's own thread.
    """
        
    def run(self):
        """
        Run the experiment.
        """
        p = Process(target=self.work)
        p.start()
        while not self.isInterruptionRequested() and p.is_alive():
            time.sleep(1)
            
        if p.is_alive() and self.isInterruptionRequested():
            p.terminate()
        
    def work(self):
        """
        The actual work of that thread
        """
        #TODO: MULTILANGUAGE
        self.actInfo.emit("dataset reading") 
        #ok, lets first read the data
        data, labels=self._experiment.dataset.toNumpyArray([
            self._experiment.attributesThatShouldBeUsed(False),
            [self._experiment.label]
            ])
        
        labels=labels.ravel()   #we need row vector       
        #extractors mapping
        extMap=[self._experiment.getAttributeSetting(a, Experiment.AttributeSettings.FEATURE_EXTRACTOR) \
                for a in self._experiment.attributesThatShouldBeUsed(False)]
        
        #create storage for results
        steps=self._experiment.evaluationMethod.numOfSteps(data,labels)
        self.numberOfSteps.emit(len(self._experiment.classifiers)*(steps)+1)    #+1 reading
        
        #TODO catch memory error
        resultsStorage=Results(steps)
        self.step.emit() 
        for c in self._experiment.classifiers:
            print(c.getName(), ", ".join( a.name+"="+str(a.value.getName()+":"+", ".join([pa.name+"->"+str(pa.value) for pa in a.value.getAttributes()]) if isinstance(a.value,Plugin) else a.value) for a in c.getAttributes()))
            self.actInfo.emit("testing {} {}/{}".format(c.getName(), 1, steps))
            start = time.time()
            startProc=time.process_time()
            for step, (predicted, realLabels) in enumerate(self._experiment.evaluationMethod.run(c, data, labels, extMap)):
                end = time.time()
                endProc=time.process_time()
                if resultsStorage.steps[step].labels is None:
                    #because it does not make much sense to have true labels stored for each predictions
                    #we store labels just once for each validation step
                    resultsStorage.steps[step].labels=realLabels
                resultsStorage.steps[step].addResults(c, predicted)
                self.step.emit()
                if step+2<=steps:
                    #TODO: MULTILANGUAGE
                    self.actInfo.emit("testing {} {}/{}".format(c.getName(), step+2, steps))
                
                self.writeConfMat(predicted, realLabels)

                print(classification_report(realLabels, predicted))
                print("accuracy\t{}".format(accuracy_score(realLabels, predicted)))
                print("Time:",end-start)
                print("Process time:",endProc-startProc)
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
