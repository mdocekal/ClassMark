"""
Created on 18. 2. 2019
Module for data set representation and actions.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

import os
import csv
import numpy as np
from typing import List, Dict
from ..core.plugins import FeatureExtractor
from skimage.io import imread
from abc import ABC


class LazyFileReaderFactory(object):
    """
    Lazy file reader factory.
    """
    
    @staticmethod
    def getReader(filePath:str, t:FeatureExtractor.DataTypes):
        """
        Create file reader.
        
        :param filePath: Path to file that should be read, at time it is absolutely necessary.
            It means when the content is actually required.
        :type filePath: str
        :param t: Data type.
        :type t: FeatureExtractor.DataTypes
        """
        
        if t ==FeatureExtractor.DataTypes.STRING:
            return LazyTextFileReader(filePath)
        if t==FeatureExtractor.DataTypes.IMAGE:
            return LazyImageFileReader(filePath)
        
        

class LazyFileReader(ABC):
    """
    Lazy reader for attributes that are marked as path.
    """
    
    def __init__(self, filePath:str):
        """
        Initializes lazy file reader.
         
        :param filePath: Path to file that should be read, at time it is absolutely necessary.
            It means when the content is actually required.
        :type filePath: str
        """
        self._filePath=filePath
        
    def __repr__(self):
        """
        Representation of that reader
        """
        return "{}: {}".format(__class__.__name__, self._filePath)
    
class LazyTextFileReader(LazyFileReader):
    """
    Lazy text reader for attributes that are marked as path.
    """
    def __init__(self, filePath:str):
        super().__init__(filePath)

        
    def __str__(self):
        """
        Get content of the file.
        """

        try:
            with open(self._filePath, "r", encoding="utf-8") as fileToRead:
                return fileToRead.read()
        except:
            raise RuntimeError("Couldn't read file: "+self._filePath)
        
    def __repr__(self):
        """
        Representation of that reader
        """
        return str(self)
    
 
class LazyImageFileReader(LazyFileReader):
    """
    Lazy image reader for attributes that are marked as path.
    """
    
    def __init__(self, filePath:str):
        super().__init__(filePath)

    def getRGB(self):
        try:
            return imread(self._filePath)
        except:
            raise RuntimeError("Couldn't read file: "+self._filePath)
        

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
        self._cntSamples=None
        self._folder=os.path.dirname(filePath)
        self._pathAttributes={}  #attributes that points to content in different file
        
        #load just the attributes.
        with open(filePath, "r", encoding="utf-8") as opF:
            reader = csv.DictReader(opF)
            self._attributes=reader.fieldnames
            
        self._subset=None   #determines if we want to use just subset
        
    def __eq__(self, other):
        return self._filePath==other._filePath and self._subset == other._subset and \
            self._pathAttributes==other._pathAttributes
    
    def useSubset(self, subset:np.array):
        """
        Use just subset of samples.
        
        :param subset: Subset of samples. Contains indexes of samples
            that should be used. Indexes must be sorted in ascending order.
            None means that you do not want to use subset anymore
        :type subset:np.array | None
        """
        self._subset=subset
        self._cntSamples=None
    
    def save(self, filePath, useOnly:List[str]=None):    
        """
        Saves data set.
        
        :param filePath: Path to file.
        :type filePath: str
        :param useOnly: attributes filter
        :type useOnly: List[str] | None
        """
        if useOnly is None:
            useOnly=self._attributes
        with open(filePath, "w", encoding="utf-8") as opF:
            writter=csv.DictWriter(opF, useOnly, extrasaction='ignore')
            writter.writeheader()
            
            for _, sample in self._goThrough():
                writter.writerow(sample)
            
    @property
    def pathAttributes(self):
        """
        List of attributes marked as path.
        """
        return set(self._pathAttributes.keys())
    
    def addPathAttribute(self, attr:str, t:FeatureExtractor.DataTypes):
        """
        Marks attribute as path to different file whichs content shold be read and
        used instead of the path.
        If path is written as relative than the base point is taken the directory in which this
        dataset file is stored.
        
        :param attr: The attribute name.
        :type attr: str
        :param t: Type of data for reading.
        :type t: FeatureExtractor.DataTypes
        :raise KeyError: When the name of attribute is uknown.
        """
        if attr not in self._attributes:
            raise KeyError("Unknown attribute.")
        
        self._pathAttributes[attr]=t
        
    def removePathAttribute(self, attr:str):
        """
        Removes path mark from attribute.
        
        :param attr: The attribute name.
        :type attr: str
        """
        del self._pathAttributes[attr]
        
            
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
        
        for sampleNum, sample in self._goThrough():
            try:
                self._prepareSample(sample)
                yield sample
            except Exception as e:
                raise RuntimeError("Error when reading file "+self._filePath+" on sample: "+str(sampleNum))
                        
    def _goThrough(self):
        """
        Just goes trough of selected samples in csv.
        
        :return: Generator with filter sample and sample number (number is counted among all samples not just the filtered).
        :rtype: Iterator[Tuple[int, Any]]
        """
        
        with open(self._filePath, "r", encoding="utf-8") as opF:
            reader = csv.DictReader(opF)
            if self._subset is None:
                for i, sample in enumerate(reader):
                    yield (i, sample)
                    
            else:
                if self._subset.shape[0]>0:
                    #user wants just the subset
                    subIter=iter(self._subset)
                    try:
                        nextFromSub=next(subIter)
                        
                        for i, sample in enumerate(reader):
                            if i==nextFromSub:
                                #we are in subset
                                yield (i, sample)
                                
                                nextFromSub=next(subIter)
                    except StopIteration:
                        #end of subset
                        pass

        
                        
    def _prepareSample(self, sample:Dict[str,str]):
        """
        Makes all necessary preparation for sample.
        
        :param sample: Data sample.
        :type sample: Dict[str,str]
        """
        #convert the path attributes
        for pA, pT in self._pathAttributes.items():
            
            if os.path.isabs(sample[pA]):
                sample[pA]=LazyFileReaderFactory.getReader(sample[pA], pT)
            else:
                #it is relative path so let's 
                #add, as base, folder of that data set
                
                sample[pA]=LazyFileReaderFactory.getReader(os.path.join(self._folder,sample[pA]), pT)

    
    def countSamples(self):
        """
        Get number of samples inside data set.
        """
        
        if self._cntSamples is None:
            self._cntSamples=sum( 1 for _ in self._goThrough())
                
        return self._cntSamples
        
    def toNumpyArray(self, useOnly:List[List[str]]):
        """
        Stores all samples to numpy array.
        All attributes must have value else ValueError is raised.
        
        :param useOnly: Which samples attributes should be only used (stored in sub list).
            The order of attributes is given by this sub list.
            Because this attribute is list of list than multiple numpy arrays are created
            for each sub list.
            
            Example use case:
                [["attribute1", ["attribute 2"]],["label"]]
                
                and you wil get two numpy array one for attributes and the other for labels
                
        :type useOnly: List[List[str]]

        :return: Numpy array with samples values.
        :rtype: [np.array]
        :raise ValueError: When attribute has None value.
        """

        res=[np.empty([self.countSamples(), len(u)], dtype=object) for u in useOnly]
        for n, s in enumerate(self):
            for i, u in enumerate(useOnly):
                for ui, a in enumerate(u):
                    if s[a] is None:
                        raise ValueError("Error when reading file "+self._filePath
                                         +" on sample: "+str(n)
                                         +". Attribute "+str(a)+" has None value.")

                    res[i][n][ui]=s[a]
                    
        return res
