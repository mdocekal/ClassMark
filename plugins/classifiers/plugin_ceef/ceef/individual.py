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
import copy
from . import functions
from typing import Callable, List

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

        self._scoresSum=0
        self._numOfScoresInSum=0
        self._evaluate=True #True means that this individual should be evaluated
        self._dataSet=dataSet
        self._funGenes=[]
        
    @classmethod
    def createInit(cls, dataSet:EvoDataSet, maxMutations, maxStartSlots):
        """
        Creates individual for initial population.

        :param dataSet: Data set that will be used for evolution and evaluation of solution.
        :type dataSet: EvoDataSet
        :param maxMutations: Maximal number of mutations used in init.
            Is used for determination of maximal number of sample slots for FunGenes.
        :type maxMutations: int
        :param maxStartSlots: Maximum number of slots for start. (minimal is always 1)
        :type maxStartSlots: int
        :return: The new individual.
        :rtype: Individual
        """
        
        theNewSelf=cls(dataSet)
        
        #let's add fun genes for each class
        for c in dataSet.classes:

            indices=np.where(dataSet.labels==c)[0]
            outerIndices=np.where(dataSet.labels!=c)[0]

            theNewSelf._funGenes.append(FunGenes.createInit(dataSet,indices,outerIndices,maxMutations))

        return theNewSelf
    
    def __copy__(self):
        """
        Makes copy of itself.
        
        :return: Copy of this.
        :rtype: Individual
        """
        c=type(self)(self._dataSet)
        c._funGenes=[copy.copy(f) for f in self._funGenes]
        
        return c
        
    def shouldEvaluate(self):
        """
        Set's flag that this individual should be evaluated (again).
        """
        self._evaluate=True
    
    @property
    def score(self):
        """
        Fitness of that individual.
        
        It is AVG score of all evaluations.
        Sets evaluate flag to false. Because this individual becomes evaluated one.
        
        :return: Returns AVG accuracy on all tests used for evaluation.
        :rtype: float
        """

        if self._evaluate:
            self._evaluate=False

            #ok, we do not have it yet so lets calc it
            predicted=self.classify(self._dataSet.testData)
            
            score=np.sum(predicted == self._dataSet.testLabels)
        
            if self._dataSet.testData.shape[0]>0:
                score/=self._dataSet.testData.shape[0]
                
            self._scoresSum+=score
            self._numOfScoresInSum+=1

        return self._scoresSum/self._numOfScoresInSum
    
    def classify(self, samples):
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
    def crossover(cls, firstParent, secondParent):
        """
        Performs uniform crossover on given individuals.
        
        :param firstParent: Individual for crossover.
        :type firstParent: Individual
        :param secondParent: Individual for crossover.
        :type secondParent: Individual
        :return: Children of given parents. 
        :rtype: Tuple[Individual,Individual]
        """
        
        theNewSib=cls(firstParent._dataSet)
        theNewSib2=cls(firstParent._dataSet)
        parents=[firstParent,secondParent]
        for fi in range(len(parents[0]._funGenes)):
            #for each fun gene
            childOne,childTwo=FunGenes.crossover(
                firstParent._funGenes[fi],secondParent._funGenes[fi])
            
            theNewSib._funGenes.append(childOne)
            theNewSib2._funGenes.append(childTwo)
            
        return (theNewSib,theNewSib2)

    
    def mutate(self, m:int):
        """
        Performs mutation.
        
        Also invalidates some internal members.
        
        :param m: Maximum number of mutations that can be performed on
            that individual.
        :type m: int
        """

        #randomly selects number of mutations
        actM=random.randint(0,m)
        if actM >0:
            self._score=None

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
        :type classDataIndices: np.array indices
        :param outerIndices: Indices of others classes samples.
        :type outerIndices: np.array indices
        :param function: Function that should be used for estimating likelihood. None means
            that you just want select randomly one of FUNCTIONS
        :type function: Callable[[List[np.array],List[float]],float]
        """
        self._dataSet=dataSet
        self._classDataIndices=classDataIndices
        self._outerIndices=outerIndices
        
        self._fun=random.choice(self.FUNCTIONS) if function is None else function

        
        self._samplesSlots=[]
        self._samplesSlotsVal=[]

        self._fenotypeGenerated=None

    def __copy__(self):
        """
        Makes copy of itself.
        
        :return: Copy of this.
        :rtype: Individual
        """
        c=type(self)(self._dataSet,self._classDataIndices,self._outerIndices,self._fun)
        c._samplesSlots=self._samplesSlots[:]
        c._samplesSlotsVal=self._samplesSlotsVal[:]
        return c
    
    @classmethod
    def createInit(cls, dataSet:EvoDataSet, classDataIndices,outerIndices, maxSlots):
        """
        Creates funGenes for initial population.
        
        :param dataSet: Data set that will be used for evolution and evaluation of solution.
        :type dataSet: EvoDataSet
        :param classDataIndices: Indices of current class samples.
        :type classDataIndices: np.array indices
        :param outerIndices: Indices of others classes samples.
        :type outerIndices: np.array indices
        :param maxSlots: Maximum number of initial slots for class and outer samples. Minimal is two slots.
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
            numberOfClassSlots=random.randint(1,maxSlots-1)
            numberOfOuterSlots=random.randint(1,maxSlots-numberOfClassSlots)

        
        try:
            self._addNewSlots(numberOfClassSlots, True)
        except self.CanNotFillSlotException as e:
            #if we have at least one than it is ok
            if len(self._samplesSlots)==0:
                raise e

        try:
            self._addNewSlots(numberOfOuterSlots, False)
        except self.CanNotFillSlotException as e:
            #if we have at least two than it is ok
            if len(self._samplesSlots)<2:
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

        for _ in range(n):
            #select one
            self._samplesSlots.append(self._selectUniqueSample(classSamples))
            
            self._samplesSlotsVal.append(1 if classSamples else -1)

    def _selectUniqueSample(self, classSamples):
        """
        Selects unique sample.

        :param classSamples: True means select from class samples. False means outer samples.
        :type classSamples: bool
        :raise CanNotFillSlotException: This exception raises when there is no free sample that can fill empty slot.
        :return: selected sample
        :rtype: np.array
        """
        
        indicies=self._classDataIndices if classSamples else self._outerIndices
        sel=random.randrange(indicies.shape[0])
        actSel=self._dataSet.data[indicies[sel]].todense().A1   #.A1 matrix to vector conversion
        if self._sampleAllreadyIn(self._samplesSlots,actSel):
            #not unique sample
            #find next that is free
            tmpSel=(sel+1)%indicies.shape[0]
            while tmpSel!=sel:
                actSel=self._dataSet.data[indicies[tmpSel]].todense().A1   #.A1 matrix to vector conversion
                if not self._sampleAllreadyIn(self._samplesSlots,actSel):
                    #unique
                    break
                tmpSel=(tmpSel+1)%indicies.shape[0]
                
            if tmpSel==sel:
                #couldn't get unique
                raise self.CanNotFillSlotException()

        return actSel
        
    @staticmethod
    def _sampleAllreadyIn(samplesSlots, sample):
        """
        Checks if sample is already in some slot.
        
        :param samplesSlots: Searcher in that slots.
        :type samplesSlots: np.array
        :param sample: The samples that should be checked.
        :type sample: np.array
        """
        
        return any(True for x in samplesSlots if np.allclose(sample,x))
        
        
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
    def crossover(cls, firstParent, secondParent):
        """
        Performs uniform crossover on given FunGenes.
        
        :param firstParent: FunGenes for crossover.
        :type firstParent: FunGenes
        :param secondParent: FunGenes for crossover.
        :type secondParent: FunGenes
        :return: Children of given parents. 
        :rtype: Tuple[FunGenes,FunGenes]
        """

        parents=[firstParent, secondParent]
        parSel=random.randint(0,1)
        
        
        #let's create brand new and choose parent for function 
        newSib=cls(parents[0]._dataSet,parents[0]._classDataIndices, parents[0]._outerIndices,
                   parents[parSel]._fun)
        
        newSib2=cls(parents[0]._dataSet,parents[0]._classDataIndices, parents[0]._outerIndices,
                   parents[(parSel+1)%2]._fun)
        
        #let's do the crossover for samples slots
        #Results length will be between min(len(x_samplesSlots)) and max (len(x_samplesSlots))
        maxLength=max( len(f._samplesSlots) for f in parents)
        for i in range(maxLength):
            parSel=random.randint(0,1)
            
            for s in [newSib, newSib2]:
                try:
                    actSel=parents[parSel]._samplesSlots[i]
                    if not cls._sampleAllreadyIn(s._samplesSlots,actSel):
                        #actual sample is not in this sibling already
                        s._samplesSlots.append(actSel)
                        s._samplesSlotsVal.append(parents[parSel]._samplesSlotsVal[i])
                except IndexError:
                    #parent with shorter part of chromosome was chosen
                    #just skip it
                    pass
                parSel=(parSel+1)%2
                
        return (newSib,newSib2)
    

    
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
            self._mutateSample,
            self._mutateSlots,
            self._mutateSampleValue,
            self._mutateFunction
            ]
        
        #reload the mutations counter
        self._mutations=random.randint(0,m)
        
        numPer=self._mutations
        while self._mutations>0:
            #select one of mutation kind
            random.choice(mutK)()
        
        return numPer
    
    
    def _mutateSample(self):
        """
        Mutates random sample.
        """
        #we select the maximum according to number of slots
        #because we are trying to minimize chance, that one slot will be mutated multiple times.
        maxMut=len(self._samplesSlots) if self._mutations>len(self._samplesSlots) else self._mutations

        
        mut=random.randint(1,maxMut)
        for _ in range(mut):
            try:
                classSamples=bool(random.getrandbits(1))    #randomly chooses class sample or outer
                sel=random.randrange(len(self._samplesSlots))
                changeTo=self._selectUniqueSample(classSamples)
                self._mutations-=1
                
                #overwrite
                self._samplesSlots[sel]=changeTo
                self._samplesSlotsVal[sel]=1 if classSamples else -1

            except self.CanNotFillSlotException:
                #ok we are out of data
                break
    
    def _mutateSlots(self):
        """
        Mutates (adds/removes) samples slots.
        """
        if len(self._samplesSlots)==1 or random.randint(0,1)==1:
            mut=random.randint(1,self._mutations)
            self._mutations-=mut
            #add
            try:
                self._addNewSlots(mut, bool(random.getrandbits(1))) #randomly selects outer or class sample
            except self.CanNotFillSlotException as e:
                #not enought new data
                return
            
        else:
            maxMut=min(self._mutations,len(self._samplesSlots)-1)
            mut=random.randint(1,maxMut)
            self._mutations-=mut
            #remove
            for _ in range(mut):
                i=random.randrange(len(self._samplesSlots))
                del self._samplesSlots[i]
                del self._samplesSlotsVal[i]

            
    def _mutateSampleValue(self):
        """
        Mutates value of random sample.

        """
        mut=random.randint(1,self._mutations)
        for _ in range(mut):
            self._mutations-=1
            sel=random.randrange(len(self._samplesSlotsVal))
            
            change=random.uniform(-1,1)
            
            self._samplesSlotsVal[sel]=self._samplesSlotsVal[sel]+change
            
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
            self._fenotypeGenerated=self._fun(np.array(self._samplesSlots),
                                              np.array(self._samplesSlotsVal))

        return self._fenotypeGenerated

        
    
        
    
        
