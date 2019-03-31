"""
Created on 18. 3. 2019
CEEF classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin
    
from sklearn.model_selection import StratifiedShuffleSplit
from inspect import getmembers, isfunction
from . import functions
from typing import Callable, List, Iterable, Any
import random
import numpy as np


class CEEF(Classifier):
    """
    Classification by evolutionary estimated functions (or CEEF) is classification method that uses
    something like probability density functions, one for each class, to classify input data.
    """
    
    def __init__(self, normalizer:BaseNormalizer=None, generations:int=1000, 
                 population:int=5, runs:int=1, randomSeed:int=None, maxMutations=5, testSetSize:float=0.25):
        """
        Classifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer
        :param generations: Maximum number of generations.
        :type generations: int
        :param population: Population size.
        :type population: int
        :param runs: Default is one, but if you want you can run evolution process multiple times.
        :type runs: int
        :param randomSeed: If not None than fixed seed is used.
        :type randomSeed: int
        :param maxMutations: Maximum number of changed genes in one mutation.
        :type maxMutations: int
        :param testSetSize: Size of test set, that is used for fitness score calculation.
        :type testSetSize: float
        """


        #TODO: type control must be off here (None -> BaseNormalizer) maybe it will be good if one could pass
        #object
        self._normalizer=PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                         [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin])
        self._normalizer.value=normalizer
        
        self._generations=PluginAttribute("Number of generations", PluginAttribute.PluginAttributeType.VALUE, int)
        self._generations.value=generations
        
        self._population=PluginAttribute("Population size", PluginAttribute.PluginAttributeType.VALUE, int)
        self._population.value=population

        self._runs=PluginAttribute("Evolution runs", PluginAttribute.PluginAttributeType.VALUE, int)
        self._runs.value=runs
        
        self._randomSeed=PluginAttribute("Random seed", PluginAttribute.PluginAttributeType.VALUE, int)
        self._randomSeed.value=randomSeed
        
        self._maxMutations=PluginAttribute("Max changed genes in mutation", PluginAttribute.PluginAttributeType.VALUE, int)
        self._maxMutations.value=maxMutations
        
        self._testSetSize=PluginAttribute("Test set size", PluginAttribute.PluginAttributeType.VALUE, float)
        self._testSetSize.value=testSetSize
        
    @staticmethod
    def getName():
        return "Classification by evolutionary estimated functions"
    
    @staticmethod
    def getNameAbbreviation():
        return "CEEF"
 
    @staticmethod
    def getInfo():
        return ""
    
    def train(self, data, labels):
        random.seed(self._randomSeed.value)
        
        #let's create train and test set
        
        train, test=next(StratifiedShuffleSplit(test_size=self._testSetSize.value, random_state=self._randomSeed.value).split(data, labels))
        
        #make reference for individuals
        self.actData=data
        self.actLabels=labels
        #get classes
        classes=list(np.unique(labels))

        for run in range(self._runs.value):
            #run the evolution for each class
            
            #create initial population
            population=[Individual.createInit(self, classes, train)]
            
            #evaluate
            popScores=[ i.score(test) for i in population]
            
                
                
        #reference is no longer needed
        self.actData=None
        self.actLabels=None
        
    def predict(self, data):
        pass
    
class Individual(object):
    """
    Representation of one individual in population.
    An individual is classifier.
    
    Each individual is composed by series of genes that are grouped together into a groups of genes that represents
    function for each class.
    """
    
    def __init__(self, parent:CEEF, classes:List[Any]):
        """
        Initialization of individual.
        
        :param parent: The parent classifier.
        :type parent: CEEF
        :param classes: Classes for classifier training.
        :type classes: List[Any]
        """
        self._score=None
        self._classes=classes
        self._parent=parent
        self._funGenes=[]
        
    @classmethod
    def createInit(cls, parent:CEEF, classes:List[Any], train:np.array):
        """
        Creates individual for initial population.
        
        :param parent: The parent classifier.
        :type parent: CEEF
        :param classes: Classes for classifier training.
        :type classes: List[Any]
        :param train: Indices of samples that should be used for training.
            Indices must corresponds to parent.actData and parent.actLabels arrays.
        :type train:np.array
        :return: The new individual.
        :rtype: Individual
        """
        
        theNewSelf=cls(parent,classes)
        
        #let's add fun genes for each class
        for c in classes:
            indices=[]
            outerIndices=[]
            for i in train:
                if parent.actLabels[i]==c:
                    indices.append(i)
                else:
                    outerIndices.append(i)
        
            theNewSelf._funGenes.append(FunGenes.createInit(parent,indices,outerIndices))
            
    @property
    def score(self, indices):
        """
        Fitness of that individual.
        
        :param indices: Indices of samples, in parents actData and actLabels member, that will be used for evaluation.
        :type indices: np.array
        """
        if self._score is None:
            self._score=0
            #ok we do not have it yet so lets calc it
            for sampleInd in indices:
                if self.predict(self._parent.actData[sampleInd])==self._parent.actLabels[sampleInd]:
                    self._score+=1
        
        
        return self._score
    
    def predict(self, sample):
        """
        Predicts class for given sample according to actual chromosome.
        
        :param sample: The data sample.
        :type sample: np.array
        :return: Predicted class.
        :rtype: Any
        """
        predicted = np.empty(len(self._funGenes))
        for i, fg in enumerate(self._funGenes):
            predicted[i] = fg.fenotype(sample)
        return self._classes[np.argmax(predicted)]
            
    
class FunGenes(object):
    """
    Genes of 
    Representation of one individual in population.
    An individual is function for some class.
    
    Each individual is composed from n data samples and compound function.
    """
    
    FUNCTIONS=[o for o in getmembers(functions) if isfunction(o[1])]
    """Functions that are useful for estimating 
    likelihood that given vector is in a class."""
    
    class CanNotFillSlotException(Exception):
        """
        This exception raises when there is no free sample that can fill empty slot.
        """
        super().__init__("There is no free sample that can fill empty slot.")
    
    def __init__(self, parent:CEEF, classDataIndices, outerIndices, function:Callable[[List[np.array],List[float]],float]=None):
        """
        Initializes genes.
        
        :param parent: The parent classifier.
        :type parent: CEEF
        :param classDataIndices: Indices of current class samples.
        :type classDataIndices: List[int]
        :param outerIndices: Indices of others classes samples.
        :type outerIndices: List[int]
        :param function: Function that should be used for estimating likelihood. None means
            that you just want select randomly one of FUNCTIONS
        :type function: Callable[[List[np.array],List[float]],float]
        """
        self._parent=parent
        self._classDataIndices=classDataIndices
        self._outerIndices=outerIndices
        
        self._fun=random.choice(self.FUNCTIONS) if function is None else function

        
        self._classSamples=[]
        self._classSamplesVal=[]
        
        self._outerSamples=[]
        self._outerSamplesVal=[]

        self._fenotypeGenerated=None

    @classmethod
    def createInit(cls, parent:CEEF, classDataIndices,outerIndices):
        """
        Creates individual for initial population.
        
        :param parent: The parent classifier.
        :type parent: CEEF
        :param classDataIndices: Indices of current class samples.
        :type classDataIndices: List[int]
        :param outerIndices: Indices of others classes samples.
        :type outerIndices: List[int]
        :return: The genes.
        :rtype: FunGenes
        """
        
        self=cls(parent,classDataIndices,outerIndices)
        
        #individuals in initial population haves
        #at least one class sample and one outer sample
        
        if self._parent._maxMutations.value<2:
            #at least one
            numberOfClassSlots=1
            numberOfOuterSlots=1
        else:
            numberOfClassSlots=random.randint(1,self._parent._maxMutations.value)
            numberOfOuterSlots=random.randint(1,self._parent._maxMutations.value)
        
        try:
            self._addNewSlots(numberOfClassSlots, True)
        except self.CanNotFillSlotException as e:
            #if we have at least one than it is ok
            if len(self._classSamples)==0:
                raise e
            
        try:
            self._addNewSlots(numberOfOuterSlots, False)
        except self.CanNotFillSlotException as e:
            #if we have at least one than it is ok
            if len(self._outerSamples)==0:
                raise e
        
        return self
        
            
    def _addNewSlots(self, n, classSamples):
        """
        Add new slots and fill them with unique samples.
        
        :param n: Number of new slots
        :type n: int
        :param classSamples: True means add class samples. False means outer samples.
        :type classSamples: bool
        :raise CanNotFillSlotException: This exception raises when there is no free sample that can fill empty slot.
        """
        slots, slotsVal=self._classSamples, self._classSamplesVal if classSamples else self._outerSamples, self._outerSamplesVal

        for _ in range(n):
            #select one
            actSel=self._selectUniqueSample(slots, classSamples)
            slots.append(actSel)
            slotsVal.append(self._occurences(actSel,classSamples))
                    
    def _selectUniqueSample(self, t, classSamples):
        """
        Selects unique sample.
        
        :param t: Slots for unique check.
        :type t: List
        :param classSamples: True means select from class samples. False means outer samples.
        :type classSamples: bool
        :raise CanNotFillSlotException: This exception raises when there is no free sample that can fill empty slot.
        :return: selected sample
        :rtype: np.array
        """
        indicies=self._classDataIndices if classSamples else self._outerIndices
        sel=random.randrange(indicies.shape[0])
        actSel=self._parent.actData[indicies[sel]].todense()
        if any(True for x in t if np.allclose(actSel,x[0])):
            #not unique sample
            #find next that is free
            tmpSel=(sel+1)%indicies.shape[0]
            while tmpSel!=sel:
                actSel=self._parent.actData[indicies[tmpSel]].todense()
                if not any(True for x in t if np.allclose(actSel,x[0])):
                    #unique
                    break
                tmpSel=(tmpSel+1)%indicies.shape[0]
                
            if tmpSel==sel:
                #couldn't get unique
                raise self.CanNotFillSlotException()
        
        return actSel
        
    def _occurences(self, sample,classSamples):
        """
        Counts occurrences of given sample in class data.
        
        :param sample: The sample.
        :type sample: np.array()
        :param classSamples: True means fill with class samples. False means outer samples (always returns zero).
        :type classSamples: bool
        :return: Number of occurrences of sample but for classSamples=False returns 0.
        :rtype: int
        """
        if not classSamples:
            return 0

        return sum(1 for i in self._classDataIndices if np.allclose(self._parent.actData[i].todense(),sample))
    
    def mutate(self, m:int):
        """
        Performs mutation. (randomly selects one mutation or none)
        
        :param m: Maximum number of mutations that can be performed on
            that genes.
        :type m: int
        :return: Number of mutations actually performed.
        :rtype: int
        """
        
        mutK=[
            self.mutateClassSamples,
            self.mutateOuterSamples,
            self.mutateClassSamplesSlots,
            self.mutateOuterSamplesSlots,
            self.mutateFunction
            ]
        
        #reload the mutations counter
        self._mutations=random.randint(0,m)
        
        numPer=self._mutations
        while self._mutations>0:
            #select one of mutation kind
            mutK[random.randrange(len(mutK))]()
        
        return numPer
    
    def _mutateClassSamples(self):
        """
        Mutates class samples.
        """
        self._mutateSample(True)
        
    
    def _mutateOuterSamples(self):
        """
        Mutates outer samples.
        """
        self._mutateSample(False)
    
    def _mutateSample(self, classSamples):
        """
        Mutates class or outer sample according to parameter.

        :param classSamples: True means mutate class samples. False means outer samples.
        :type classSamples: bool
        """
        samples=self._classSamples if classSamples else self._outerSamples
        samplesVal=self._classSamplesVal if classSamples else self._outerSamplesVal
        
        #we select the maximum according to number of slots
        #because we are trying to minimize chance, that one slot will be mutated multiple times.
        maxMut=len(samples) if self._mutations>len(samples) else self._mutations
        
        mut=random.randint(1,maxMut)
        for _ in range(mut):
            try:
                sel=random.randrange(len(samples))
                changeTo=self._selectUniqueSample(samples, classSamples)
                self._mutations-=1
                
                #overwrite
                samples[sel]=changeTo
                samplesVal[sel]=self._occurences(changeTo,classSamples)
            except self.CanNotFillSlotException:
                #ok we are out of data
                break
        
    
    def _mutateClassSamplesSlots(self):
        """
        Adds or removes slots for class samples.
        """
        self._mutateSlots(True)
    
    def _mutateOuterSamplesSlots(self):
        """
        Adds or removes slots for outer samples.
        """
        self._mutateSlots(False)
    
    def _mutateSlots(self, classSamples):
        """
        Mutates (adds/removes) class or outer sample slots according to parameter.

        :param classSamples: True means mutate class samples slots. False means outer samples slots.
        :type classSamples: bool
        """
        
        slots=self._classSamples if classSamples else self._outerSamples
        
        mut=random.randint(1,self._mutations)
        
        if len(slots)==1 or random.randint(0,1)==1:
            #add
            self._addNewSlots(mut, classSamples)
        else:
            slotsVal=self._classSamplesVal if classSamples else self._outerSamplesVal
            #remove
            for _ in range(mut):
                i=random.randrange(len(slots))
                del slots[i]
                del slotsVal[i]
    
    def _mutateFunction(self):
        """
        Changes actual function.
        """
        self._mutations-=1
        self._fun=random.choice(self.FUNCTIONS)
        
    
    @property
    def fenotype(self):
        """
        Fenotype of that individual.
        """
        
        if self._fenotypeGenerated is None:
            self._fenotypeGenerated=self._fun(self._classSamples+self._outerSamples,
                self._classSamplesVal+self._outerSamplesVal)

        return self._fenotypeGenerated

        
    