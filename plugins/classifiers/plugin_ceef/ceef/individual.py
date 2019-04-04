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
from typing import Callable, List
from scipy.sparse import csr_matrix
import time

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
            indices=np.array(indices)
            outerIndices=np.array(outerIndices)

            theNewSelf._funGenes.append(FunGenes.createInit(dataSet,indices,outerIndices,maxMutations))

        return theNewSelf
    
    @property
    def score(self):
        """
        Fitness of that individual.
        
        :return: Returns accuracy on given test set.
        :rtype: float
        """
        if self._score is None:
            print("CALC score enter")
            self._score=0
            #ok, we do not have it yet so lets calc it
            predicted=self.predict(self._dataSet.testData)
            
            self._score=np.sum(predicted == self._dataSet.testLabels)
        
            if self._dataSet.testData.shape[0]>0:
                self._score/=self._dataSet.testData.shape[0]
            
            print("CALC score leave")
        return self._score
    
    def predict(self, samples):
        """
        Predicts class for given samples according to actual chromosome.
        
        :param samples: The data samples.
        :type samples: np.array
        :return: Predicted classes for samples.
        :rtype: np.array
        """
        predicted=np.empty(samples.shape[0], self._dataSet.classes.dtype)
        
        funVals=np.empty((len(self._funGenes),samples.shape[0]))

        for i, fg in enumerate(self._funGenes):
            funVals[i] = fg.fenotype(samples)
            
        funVals=funVals.T   #for better iteration
        
        for i, row in enumerate(funVals):
            predicted[i]=self._dataSet.classes[np.argmax(row)]

        return predicted
            
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
        
        for fi in range(len(individuals[0]._funGenes)):
            #for each fun gene
            theNewSib._funGenes.append(FunGenes.crossover([i._funGenes[fi] for i in individuals]))
            
        return theNewSib

    
    def mutate(self, m:int):
        """
        Performs mutation.
        
        Also invalidates some internal members.
        
        :param m: Maximum number of mutations that can be performed on
            that individual.
        :type m: int
        """
        
        self._score=None
        
        #randomly selects number of mutations
        actM=random.randint(0,m)

        order=[o for o in range(len(self._funGenes))]
        random.shuffle(order)
        
        for o in order:
            #we are iterating f. genes in random order
            fG=self._funGenes[o]
            
            actM-=fG.mutate(actM)
            if actM<=0:
                #maximal number of mutations exceeded
                break

    
class FunGenes(object):
    """
    Genes of likely function of on class.
    
    Each FunGenes is composed from n data samples and compound function.
    """
    
    FUNCTIONS=[o for _,o in getmembers(functions) if isfunction(o)]
    """Functions that are useful for estimating 
    likelihood that given vector is in a class."""
    
    class CanNotFillSlotException(Exception):
        """
        This exception raises when there is no free sample that can fill empty slot.
        """
        def __init__(self):
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
        #no need to store values of outer samples because be default all have zero

        self._fenotypeGenerated=None

    @classmethod
    def createInit(cls, dataSet:EvoDataSet, classDataIndices,outerIndices, maxSlots):
        """
        Creates funGenes for initial population.
        
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

        slots=self._classSamples if classSamples else self._outerSamples

        for _ in range(n):
            #select one
            actSel=self._selectUniqueSample(slots, classSamples)
            slots.append(actSel)
            if classSamples:
                self._classSamplesVal.append(1)
                    

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
        actSel=self._dataSet.data[indicies[sel]].todense().A1   #.A1 matrix to vector conversion
        if any(True for x in t if np.allclose(actSel,x)):
            #not unique sample
            #find next that is free
            tmpSel=(sel+1)%indicies.shape[0]
            while tmpSel!=sel:
                actSel=self._dataSet.data[indicies[tmpSel]].todense().A1   #.A1 matrix to vector conversion
                if not any(True for x in t if np.allclose(actSel,x)):
                    #unique
                    break
                tmpSel=(tmpSel+1)%indicies.shape[0]
                
            if tmpSel==sel:
                #couldn't get unique
                raise self.CanNotFillSlotException()

        return actSel
        
    '''
    TODO: DELETE
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
 
        
        sampleSparse=csr_matrix(sample)

        res=0
        for i in self._classDataIndices:
            act=self._dataSet.data[i]
            if sampleSparse.nnz==act.nnz:
                res+=1

        return res
    '''
    
    
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
            except IndexError:
                #parent with shorter part of chromosome was chosen
                #just skip it
                pass
            
        return newSib
    
    '''
    TODO:REMOVE
    @classmethod
    def _slotsCrossover(cls, classSamples, *funGenes):
        """
        Performs uniform crossover over slots..
        
        :param classSamples: True means crossover over class slots. False means over outer slots.
        :type classSamples: bool
        :param funGenes: FunGenes for crossover.
        :type funGenes: FunGenes
        """
        pass
        '''
    
    def mutate(self, m:int):
        """
        Performs mutation. (randomly selects one mutation or none)
        
        Also invalidates some internal members.
        
        :param m: Maximum number of mutations that can be performed on
            that genes.
        :type m: int
        :return: Number of mutations actually performed.
        :rtype: int
        """
        self._fenotypeGenerated=None
        
        mutK=[
            self._mutateClassSamples,
            self._mutateOuterSamples,
            self._mutateClassSamplesSlots,
            self._mutateOuterSamplesSlots,
            self._mutateFunction
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
                if classSamples:
                    self._classSamplesVal[sel]=1
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

        maxMut=min(self._mutations,len(slots)-1)
        if maxMut==0:
            #ok nevermind maybe later
            return
        
        mut=random.randint(1,maxMut)
        
        
        self._mutations-=mut
        
        if len(slots)==1 or random.randint(0,1)==1:
            #add
            try:
                self._addNewSlots(mut, classSamples)
            except self.CanNotFillSlotException as e:
                #not enought new data
                return
            
        else:
            #remove
            for _ in range(mut):
                i=random.randrange(len(slots))
                del slots[i]
                if classSamples:
                    del self._classSamplesVal[i]
  
    def _mutateFunction(self):
        """
        Changes actual function.
        """
        self._mutations-=1
        self._fun=random.choice(self.FUNCTIONS)
        #TODO: not all functions are suitable
        #LinearNDInterpolator  cipy.spatial.qhull.QhullError: QH6214 qhull input error: not enough points(3) to construct initial simplex (need 6)
        
    
    @property
    def fenotype(self):
        """
        Fenotype of that individual.
        """

        if self._fenotypeGenerated is None:
            self._fenotypeGenerated=self._fun(np.array(self._classSamples),np.array(self._outerSamples),
                np.array(self._classSamplesVal))

        return self._fenotypeGenerated

        
    
        
    
        
