"""
Created on 18. 2. 2019
Module for data set representation and actions.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import csv
import numpy as np
from typing import List

class Sample(object):
    """
    Representation of one sample in data set.
    """
    pass

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
        #load just the attributes.
        with open(filePath, "r") as opF:
            reader = csv.DictReader(opF)
            self._attributes=reader.fieldnames
            
    @property
    def filePath(self):
        """
        Path to file here dataset is saved.
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
        with open(self._filePath, "r") as opF:
            reader = csv.DictReader(opF)
            for sample in reader:
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