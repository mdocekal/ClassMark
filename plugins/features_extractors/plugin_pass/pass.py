"""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor, PluginAttribute

import numpy as np

class Pass(FeatureExtractor):
    """
    Pass feature extractor plugin for ClassMark.
    This extractor can work in one of these modes:
        convert data to float np.array (convToNums=True)
        convert data to np.array (convToNums=False,convToNp=True)
        just pass data (convToNums=False,convToNp=False)

    """
    
    def __init__(self, convToNums=True, convToNp=True):
        """
        Initialize Pass feature extractor.
        
        :param convToNums:If true than data are converted to float np array and
            convToNp attribute is not considered.
        :type convToNums: bool
        :param convToNp: If true than data are converted to float np array.
        :type convToNp: bool
        """
        
        self._convToNums=PluginAttribute("To float numpy array", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._convToNums.value=convToNums
        self._convToNp=PluginAttribute("To numpy array", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._convToNp.value=convToNp
        
        self._val=PluginAttribute("Value", PluginAttribute.PluginAttributeType.VALUE, int)
        self._val.value=10
        
        self._sel=PluginAttribute("Select", PluginAttribute.PluginAttributeType.SELECTABLE, str,["First", "Second", "Third"])
        self._sel.value="First"
        
        self._group=PluginAttribute("Hidden layers", PluginAttribute.PluginAttributeType.GROUP_VALUE, int)
        self._group.value=[]
    
    @staticmethod
    def getName():
        return "Pass"
    
    @staticmethod
    def getNameAbbreviation():
        return "Pass"
 
    @staticmethod
    def getInfo():
        return ""
    
    def fit(self, data, labels=None):
        """
        This function is there only for compatibility with FeatureExtractor.
        There is really no need to call it, but if you do so, than this is just empty
        operation.
        
        :param data: It does not matter.
        :type data: It does not matter.
        :param labels: It does not matter.
        :type labels: It does not matter.
        """
        pass

    def extract(self, data):
        """
        This extractor can work in one of these modes:
            convert data to float np.array (convToNums=True)
            convert data to np.array (convToNums=False,convToNp=True)
            just pass data (convToNums=False,convToNp=False)
        
        :param data: Original data for features extraction.
        :type data: ArrayLike
        :return: Converted data.
        :rtype: np.array
        """
        if self._convNums:
            return np.asfarray(data)
        elif self._convToNp:
            return np.array(data)
        else:
            return data

        