"""
Created on 18. 2. 2019
Module for data set representation and actions.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import csv

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
        
        #load just the attributes.
        with open(filePath, "r") as opF:
            reader = csv.DictReader(opF)
            self._attributes=reader.fieldnames
            
        
    @property
    def attributes(self):
        """
        Name of attributes in dataset.
        """
        return self._attributes
        