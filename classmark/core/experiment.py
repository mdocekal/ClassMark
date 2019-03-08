"""
Created on 18. 2. 2019
Module for experiment representation and actions.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from ..data.data_set import DataSet
from enum import Enum
from _collections import OrderedDict
from ..core.plugins import CLASSIFIERS, FEATURE_EXTRACTORS

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
        
        #let's load the plugins that are now available
        self._loadPlugins()
        #TODO: loading
        
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
        """
        
        if t == Experiment.AttributeSettings.LABEL:
            self._label= attribute if val else None
        else:
            self._attributesSet[attribute][t]=val
        
    
  
    @property
    def dataset(self):
        """
        Loaded dataset.
        """
        return self._dataset