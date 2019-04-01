"""
Created on 1. 4. 2019

Module containing chromosome/individual for genetic algorithm.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from inspect import getmembers, isfunction
from .evo_data_set import EvoDataSet
import numpy as np
import random
from . import functions
from typing import Callable, List, Any

class Individual(object):
    """
    Representation of one individual in population.
    An individual is classifier.
    
    Each individual is composed by series of genes that are grouped together into a groups of genes that represents
    function for each class.
    """
    
    def __init__(self, dataSet:EvoDataSet):
        """
        Initialization of individual.
        
        :param dataSet: Data set that will be used for evolution and evaluation of solution.
        :type dataSet: EvoDataSet
        """
        self._score=None
        self._dataSet=dataSet
        self._funGenes=[]
        
    @classmethod
    def createInit(cls, dataSet:EvoDataSet, maxMutations):
        """
        Creates individual for initial population.
        
        :param dataSet: Data set that will be used for evolution and evaluation of solution.
        :type dataSet: EvoDataSet
        :param maxMutations: Maximal number of mutations used in init.
            Is used for determination of maximal number of sample slots for FunGenes.
        :type maxMutations: int
        :return: The new individual.
        :rtype: Individual
        """
        
        theNewSelf=cls(dataSet)
        
        #let's add fun genes for each class
        for c in dataSet.classes:
            indices=[]
            outerIndices=[]
            for i in dataSet.trainIndices:
                if dataSet.labels[i]==c:
                    indices.append(i)
                else:
                    outerIndices.append(i)
        
            theNewSelf._funGenes.append(FunGenes.createInit(dataSet,indices,outerIndices,maxMutations))
            
    @property
    def score(self):
        """
        Fitness of that individual.
        
        :return: Returns accuracy on given test set.
        :rtype: float
        """
        if self._score is None:
            self._score=0
            #ok we do not have it yet so lets calc it
            for sampleInd in self._dataSet.testIndices:
                if self.predict(self._dataSet.data[sampleInd])==self._dataSet.labels[sampleInd]:
                    self._score+=1
        
            if self._dataSet.testIndices.shape[0]>0:
                self._score/=self._dataSet.testIndices.shape[0]
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
            
    @classmethod      
    def crossover(cls, individuals):
        """
        Performs uniform crossover on given individuals.
        
        :param individuals: Individuals for crossover.
        :type individuals: List[Individual]
        :return: Sibling of its parents.
        :rtype: Individual
        """
        
        theNewSib=cls(individuals[0]._dataSet)
        
        for fi in range(len(individuals._funGenes)):
            #for each fun gene
            theNewSib._funGenes.append(FunGenes.crossover([i._funGenes[fi] for i in individuals]))
            
        return theNewSib

    
    def mutate(self, m:int):
        """
        Performs mutation.
        
        :param m: Maximum number of mutations that can be performed on
            that individual.
        :type m: int
        """
        
        #randomly selects number of mutations
        actM=random.randint(0,m)
        
        for fG in random.shuffle(self._funGenes):
            #we are iterating f. genes in random order
            actM-=fG.mutate(actM)
            if actM<=0:
                #maximal number of mutations exceeded
                break

    
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
    
    def __init__(self, dataSet:EvoDataSet, classDataIndices, outerIndices, function:Callable[[List[np.array],List[float]],float]=None):
        """
        Initializes genes.
        
        :param dataSet: Data set that will be used for evolution and evaluation of solution.
        :type dataSet: EvoDataSet
        :param classDataIndices: Indices of current class samples.
        :type classDataIndices: List[int]
        :param outerIndices: Indices of others classes samples.
        :type outerIndices: List[int]
        :param function: Function that should be used for estimating likelihood. None means
            that you just want select randomly one of FUNCTIONS
        :type function: Callable[[List[np.array],List[float]],float]
        """
        self._dataSet=dataSet
        self._classDataIndices=classDataIndices
        self._outerIndices=outerIndices
        
        self._fun=random.choice(self.FUNCTIONS) if function is None else function

        
        self._classSamples=[]
        self._classSamplesVal=[]
        
        self._outerSamples=[]
        self._outerSamplesVal=[]

        self._fenotypeGenerated=None

    @classmethod
    def createInit(cls, dataSet:EvoDataSet, classDataIndices,outerIndices, maxSlots):
        """
        Creates individual for initial population.
        
        :param dataSet: Data set that will be used for evolution and evaluation of solution.
        :type dataSet: EvoDataSet
        :param classDataIndices: Indices of current class samples.
        :type classDataIndices: List[int]
        :param outerIndices: Indices of others classes samples.
        :type outerIndices: List[int]
        :param maxSlots: Maximum number of initial class and outer slots. Minimal is one slot.
            So at least 2 slots will be always created (one for each).
        :type maxSlots: int
        :return: The genes.
        :rtype: FunGenes
        """
        
        self=cls(dataSet,classDataIndices,outerIndices)
        
        #individuals in initial population haves
        #at least one class sample and one outer sample
        
        if maxSlots<2:
            #at least one
            numberOfClassSlots=1
            numberOfOuterSlots=1
        else:
            numberOfClassSlots=random.randint(1,maxSlots)
            numberOfOuterSlots=random.randint(1,maxSlots)
        
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
        actSel=self._dataSet.data[indicies[sel]].todense()
        if any(True for x in t if np.allclose(actSel,x)):
            #not unique sample
            #find next that is free
            tmpSel=(sel+1)%indicies.shape[0]
            while tmpSel!=sel:
                actSel=self._dataSet.data[indicies[tmpSel]].todense()
                if not any(True for x in t if np.allclose(actSel,x)):
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

        return sum(1 for i in self._classDataIndices if np.allclose(self._dataSet.data[i].todense(),sample))
    
    @classmethod      
    def crossover(cls, funGenes):
        """
        Performs uniform crossover on given FunGenes.
        
        :param funGenes: FunGenes for crossover.
        :type funGenes: List[FunGenes]
        """
        

        #let's create brand new and choose parent for function 
        newSib=cls(funGenes[0]._dataSet,funGenes[0]._classDataIndices, funGenes[0]._outerIndices,
                   random.choice(funGenes)._fun)
        
        #let's do the crossover for class samples
        #Results length will be between min(len(x_classSamples)) and max (len(x_classSamples))
        maxLength=max( len(f._classSamples) for f in funGenes)
        for i in range(maxLength):
            parentForGene=random.choice(funGenes)
            try:
                if any(True for x in newSib._classSamples if np.allclose(parentForGene._classSamples[i],x)):
                    #is already in sibling, skip it
                    #TODO: maybe it will be better solution to try different parent
                    continue
                    
                newSib._classSamples.append(parentForGene._classSamples[i])
                newSib._classSamplesVal.append(parentForGene._classSamplesVal[i])
            except IndexError:
                #parent with shorter part of chromosome was chosen
                #just skip it
                pass
            
        #now the crossover for outer samples
        maxLength=max( len(f._outerSamples) for f in funGenes)
        for i in range(maxLength):
            parentForGene=random.choice(funGenes)
            try:
                if any(True for x in newSib._outerSamples if np.allclose(parentForGene._outerSamples[i],x)):
                    #is already in sibling, skip it
                    #TODO: maybe it will be better solution to try different parent
                    continue
                    
                newSib._outerSamples.append(parentForGene._outerSamples[i])
                newSib._outerSamplesVal.append(parentForGene._outerSamplesVal[i])
            except IndexError:
                #parent with shorter part of chromosome was chosen
                #just skip it
                pass
            
        return newSib
    
    @classmethod
    def _slotsCrossover(cls, classSamples, *funGenes):
        """
        Performs uniform crossover over slots..
        
        :param classSamples: True means crossover over class slots. False means over outer slots.
        :type classSamples: bool
        :param funGenes: FunGenes for crossover.
        :type funGenes: FunGenes
        """
    
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

        
    
        
    
        
