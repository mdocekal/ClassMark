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


    def __init__(self, data, labels, testSize, randomSeed):
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
        """
        self._data=data
        
        self._labels=labels
        self._train, self._test=next(StratifiedShuffleSplit(test_size=testSize, random_state=randomSeed).split(data, labels))
        self._classes=np.unique(labels)
        self._testData=self._data[self._test].A
        self._testLabels=self._labels[self._test]
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
    def trainIndices(self):
        """
        Indices of train set.
        
        :return: indices
        :rtype: np.array
        """
        
        return self._train
    
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