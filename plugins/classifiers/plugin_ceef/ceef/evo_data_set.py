"""
Created on 1. 4. 2019
Module containing dataset that is designed for evolution.

@author: windionleaf
"""
    
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

class EvoDataSet(object):
    """
    Dataset for evolution that provides train and test set.
    Train is used for evolution itself and test is used for evaluation.
    """


    def __init__(self, data, labels, testSize, randomSeed, nChanges=0):
        """
        Initializes dataset for evolution.
        
        :param data: All data that are available.
        :type data:scipy.sparse matrix
        :param labels: Labels for the data.
        :type labels: np.array
        :param testSize: Size of test set, that is used for fitness score calculation. Number in interval 0..1.
        :type testSize: float
        :param randomSeed:If not None than fixed seed is used.
        :type randomSeed: int
        :param nChanges: How many times you will be changing the test set.
        :type nChanges: int
        """
        self._data=data
        
        self._labels=labels
        self._classes=np.unique(labels)
        
        #n_splits=nChanges+1    because we are using change at initialization
        self._shuffler=StratifiedShuffleSplit(n_splits=nChanges+1, test_size=testSize, random_state=randomSeed).split(data, labels)
        self.changeTestSet()
        
    def changeTestSet(self):
        """
        Randomly selects new samples for testing.
        """
        _, test=next(self._shuffler)
        self._testData=self._data[test].A
        self._testLabels=self._labels[test]
        
    @property
    def data(self):
        """
        All data that are available.
        
        :return: Data
        :rtype data:scipy.sparse matrix
        """
        return self._data
    
    @property
    def labels(self):
        """
        Labels for the data.
        
        :return: Labels
        :rtype np.array
        """
        return self._labels
    
    @property
    def classes(self):
        """
        All classes/labels.
        
        :return: classes
        :rtype: np.array
        """
        return self._classes
    
    @property
    def testData(self):
        """
        All data that should be used for testing.
        
        :return: Data in test set.
        :rtype data:np.array
        """
        
        return self._testData
    @property
    def testLabels(self):
        """
        All labels that should be used for testing.
        
        :return: Labels in test set.
        :rtype data:np.array
        """
        
        return self._testLabels