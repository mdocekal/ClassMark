"""
Created on 18. 2. 2019
Module for data set representation and actions.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import os
import csv
import numpy as np
from typing import List

class Sample(object):
    """
    Representation of one sample in data set.
    """
    pass

class LazyTextFileReader(str):
    """
    Lazy text reader for attributes that are marked as path.
    """
    
    def __init__(self, filePath:str):
        """
        Initializes lazy file reader.
         
        :param filePath: Path to file that should be read, at time it is absolutely necessary.
            It means when the content is actually required.
        :type filePath: str
        """
        self._filePath=filePath
        

    def __str__(self):
        """
        Get content of the file.
        """
        with open(self._filePath, "r", encoding="utf-8") as fileToRead:
            return fileToRead.read()

    def __repr__(self):
        """
        Representation of that reader
        """
        return "{}: {}".format(__class__.__name__, self._filePath)
 
        

class DataSet(object):
    """
    Representation of data set.
    """

    def __init__(self, filePath:str):
        """
        Loading of saved data set.
        
        :param filePath: Path to data set.
        :type filePath: str
        """
        
        self._samples=[]
        self._attributes=[]
        self._filePath=filePath
        self._folder=os.path.dirname(filePath)
        self._pathAttributes=set()  #attributes that points to content in different file
        #load just the attributes.
        with open(filePath, "r", encoding="utf-8") as opF:
            reader = csv.DictReader(opF)
            self._attributes=reader.fieldnames
            
    def addPathAttribute(self, attr:str):
        """
        Marks attribute as path to different file whichs content shold be read and
        used instead of the path.
        If path is written as relative than the base point is taken the directory in which this
        dataset file is stored.
        
        :param attr: The attribute name.
        :type attr: str
        :raise KeyError: When the name of attribute is uknown.
        """
        if attr not in self._attributes:
            raise KeyError("Unknown attribute.")
        
        self._pathAttributes.add(attr)
        
    def removePathAttribute(self, attr:str):
        """
        Removes path mark from attribute.
        
        :param attr: The attribute name.
        :type attr: str
        """
        self._pathAttributes.remove(attr)
        
            
    @property
    def filePath(self):
        """
        Path to file where dataset is saved.
        """
        return self._filePath
    
    @property
    def attributes(self):
        """
        Name of attributes in dataset.
        """
        return self._attributes
    
    def __iter__(self):
        """
        Iterate over each sample.
        """
        with open(self._filePath, "r", encoding="utf-8") as opF:
            reader = csv.DictReader(opF)
            for sample in reader:
                #convert the path attributes
                for pA in self._pathAttributes:
                    if not os.path.isabs(sample[pA]):
                        #it is relative path so let's 
                        #add as base folder of that data set
                        sample[pA]=LazyTextFileReader(os.path.join(self._folder,sample[pA]))
                        
                yield sample
        
    def toNumpyArray(self, useOnly:List[List[str]]=[],):
        """
        Stores all samples to numpy array.
        
        :param useOnly: Which samples attributes should be only used (stored in sub list).
            The order of attributes is given by this sub list.
            Because this attribute is list of list than multiple numpy arrays are created
            for each sub list.
        :type useOnly: List[List[str]]
        :return: Numpy array with samples values.
        :rtype: [np.array] | [np.array]
        """
        
        
        if useOnly:
            res=[[]for _ in range(len(useOnly))]
            for s in self:
                for i, u in enumerate(useOnly):
                    res[i].append([s[a] for a in u])
            return [np.array(r) for r in res]
        else:
            samples=[]
            for s in self:
                samples.append([v for v in s.value()])
        
            return np.array(samples)