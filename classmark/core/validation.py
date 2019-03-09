"""
Created on 9. 3. 2019
Module for validation methods.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from abc import abstractmethod
from .plugins import Plugin, PluginAttribute

from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold

class EvaluationMethod(object):
    """
    Evaluation method for experiment.
    """
    pass

class Validator(Plugin):
    """
    Base class for all validation methods.
    """

    @abstractmethod
    def run(self):
        """
        Run whole validation process.
        """
        pass
        
    @abstractmethod
    def step(self):
        """
        Run one step of validation process.
        """
        pass
    
    @property
    @abstractmethod
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    @property
    @abstractmethod
    def numOfSteps(self):
        """
        Total number of steps to complete validation process.
        """
        pass
    
class ValidatorStratifiedKFold(Validator):
    """
    Validation process that uses StratifiedKFold for getting
    train and test sets.
    """
    
    def __init__(self, folds:int=5):
        """
        Initialize StratifiedKFold validation
        
        :param folds: Number of folds.
        :type folds: int
        """
        
        self._folds=PluginAttribute("Folds", PluginAttribute.PluginAttributeType.VALUE, int)
        self._folds.value=folds
    
    @staticmethod
    def getName():
        return "stratified-k-fold"
 
    @staticmethod
    def getNameAbbreviation():
        return "SKF"
    
    @staticmethod
    def getInfo():
        return ""
    
    def run(self):
        """
        Run whole validation process.
        """
        pass

    def step(self):
        """
        Run one step of validation process.
        """
        pass
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    @property
    def numOfSteps(self):
        """
        Total number of steps to complete validation process.
        """
        pass

class ValidatorKFold(Validator):
    """
    Validation process that uses KFold for getting
    train and test sets.
    """
    
    def __init__(self, folds:int=5):
        """
        Initialize ValidationKFold validation
        
        :param folds: Number of folds.
        :type folds: int
        """
        
        self._folds=PluginAttribute("Folds", PluginAttribute.PluginAttributeType.VALUE, int)
        self._folds.value=folds
    
    @staticmethod
    def getName():
        return "k-fold"
 
    @staticmethod
    def getNameAbbreviation():
        return "KF"
    
    @staticmethod
    def getInfo():
        return ""
    
    def run(self):
        """
        Run whole validation process.
        """
        pass

    def step(self):
        """
        Run one step of validation process.
        """
        pass
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    @property
    def numOfSteps(self):
        """
        Total number of steps to complete validation process.
        """
        pass   

class ValidatorLeaveOneOut(Validator):
    """
    Validation process that uses LeaveOneOut for getting
    train and test sets.
    """
    
    @staticmethod
    def getName():
        return "leave-one-out"
 
    @staticmethod
    def getNameAbbreviation():
        return "LOO"
    
    @staticmethod
    def getInfo():
        return ""
    
    def run(self):
        """
        Run whole validation process.
        """
        pass

    def step(self):
        """
        Run one step of validation process.
        """
        pass
    
    @property
    def results(self):
        """
        Get gathered results so far.
        """
        pass
    
    @property
    def numOfSteps(self):
        """
        Total number of steps to complete validation process.
        """
        pass