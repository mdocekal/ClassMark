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
        :raise RuntimeError: Where collision of features extractors names occurs.
        """
        self._dataset=None
        self._attributesSet={}
        self._label=None
        
        #available features extractors
        feTmp={}
        for fe in FEATURE_EXTRACTORS.values():
            if fe.getName() in feTmp:
                #wow, name collision
                #TODO: Maybe multilanguage message will be better.
                raise RuntimeError("Collision of features extractors names. For name: "+fe.getName())
            feTmp[fe.getName()]=fe
        
        #TODO: raise exception when now plugins are available.
        
        #lets put the default feature extractor as the first if exists
        if self.DEFAULT_FEATURE_EXTRACTOR_NAME in feTmp:
            cont=[(self.DEFAULT_FEATURE_EXTRACTOR_NAME,feTmp[self.DEFAULT_FEATURE_EXTRACTOR_NAME])]
            #add the rest
            cont+=[(n,p) for n,p in feTmp.items() if n!=self.DEFAULT_FEATURE_EXTRACTOR_NAME]
            self._featuresExt=OrderedDict(cont)
        else:
            self._featuresExt=OrderedDict(feTmp)
        
        #TODO: loading

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
                  self.AttributeSettings.FEATURE_EXTRACTOR:next(iter(self._featuresExt.values()))} 
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