"""
Created on 18. 3. 2019
CEEF classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import Classifier, PluginAttribute,PluginAttributeIntChecker,\
    PluginAttributeFloatChecker
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin,\
    MinMaxScalerPlugin,StandardScalerPlugin, RobustScalerPlugin

from scipy.sparse import csr_matrix, spmatrix
from .evo_data_set import EvoDataSet
from .individual import Individual
from typing import List
from .selection import Selector, Rulete, Rank
import random
import copy
import numpy as np




class CEEF(Classifier):
    """
    Classification by evolutionary estimated functions (or CEEF) is classification method that uses
    something like probability density functions, one for each class, to classify input data.
    """
    SELECTION_METHODS={"RANK":Rank(), "RULETE":Rulete()}
    
    def __init__(self, normalizer:BaseNormalizer=None, generations:int=100, stopAccuracy:float=None, \
                 population:int=10, selectionMethod:Selector="RANK", randomSeed:int=None, maxCrossovers:int=1, maxMutations:int=5, \
                 maxStartSlots=2,
                 crossoverProb:float=0.75, testSetSize:float=1, changeTestSet:bool=False, logGenFitness:bool=True):
        """
        Classifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer
        :param generations: Maximum number of generations.
        :type generations: int
        :param stopAccuracy: Stop evolution when accuracy reaches concrete value.
        :type stopAccuracy: None | float
        :param population: Population size.
        :type population: int
        :param selectionMethod: Selection method for evolution.
        :type selectionMethod: Selector
        :param randomSeed: If not None than fixed seed is used.
        :type randomSeed: int
        :param maxCrossovers: Maximum number of crossovers when creating generation.
        :type maxCrossovers: int
        :param maxMutations: Maximum number of changed genes in one mutation.
        :type maxMutations: int
        :param maxStartSlots: Maximum number of slots for start. (minimal is always 1)
        :type maxStartSlots: int
        :param crossoverProb: Probability of crossover between two selected individuals.
            If random says no crossover than one of parent is randomly chosen and its chromosome is used.
        :type crossoverProb: float
        :param testSetSize: Size of test set, that is used for fitness score calculation.
        :type testSetSize: float
        :param changeTestSet: Change test set for every generation.
        :type changeTestSet: bool
        :param logGenFitness: Log generation fitness.
        :type logGenFitness: bool
        """


        self._normalizer=PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                         [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin])
        self._normalizer.value=normalizer
        
        self._generations=PluginAttribute("Number of generations", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=0))
        self._generations.value=generations
        
        self._stopAccuracy=PluginAttribute("Stop accuracy", PluginAttribute.PluginAttributeType.VALUE, 
                                           PluginAttributeFloatChecker(minV=0.0, maxV=1.0,couldBeNone=True))
        self._stopAccuracy.value=stopAccuracy
        
        self._population=PluginAttribute("Population size", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._population.value=population
    
        
        self._selectionMethod=PluginAttribute("Selection method", PluginAttribute.PluginAttributeType.SELECTABLE, None,
                                              list(self.SELECTION_METHODS.keys()))
        self._selectionMethod.value=selectionMethod

        
        self._randomSeed=PluginAttribute("Random seed", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(couldBeNone=True))
        self._randomSeed.value=randomSeed
        
        self._maxCrossovers=PluginAttribute("Max crossovers in generation", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._maxCrossovers.value=maxCrossovers
        
        self._maxMutations=PluginAttribute("Max mutations", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=0))
        self._maxMutations.value=maxMutations
        
        self._maxStartSlots=PluginAttribute("Max start slots", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=2))
        self._maxStartSlots.value=maxStartSlots
        
        self._crossoverProb=PluginAttribute("Crossover probability", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeFloatChecker(minV=0.0, maxV=1.0))
        self._crossoverProb.value=crossoverProb
        
        self._testSetSize=PluginAttribute("Test set size", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeFloatChecker(minV=0.0, maxV=1.0))
        self._testSetSize.value=testSetSize
        
        self._changeTestSet=PluginAttribute("Change test set for each generation", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._changeTestSet.value=changeTestSet
        
        
        self._logGenFitness=PluginAttribute("Log generation fitness", PluginAttribute.PluginAttributeType.CHECKABLE, bool)
        self._logGenFitness.value=logGenFitness
        
        self._evolvedCls=None    #there the evolved classifier will be stored
        
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
        if self._normalizer.value is not None:
            data=self._normalizer.value.fitTransform(data)
            if not isinstance(data, spmatrix):
                #some normalizers may return np array
                #but we need sparse
                data=csr_matrix(data)
        
        self._evolvedCls=None
        
        #let's create train and test set
        dataset=EvoDataSet(data, labels,self._testSetSize.value,self._randomSeed.value,self._generations.value)
        self.trainedClasses=dataset.classes
        
        #create initial population
        population=[]
        
        saveExp=None
        for _ in range(self._population.value):
            try:
                population.append(Individual.createInit(dataset, self._maxStartSlots.value))
            except Exception as e:
                #probably to few samples
                #let's skip it maybe another will be generated with better numbers
                saveExp=e
        
        if saveExp is not None and len(population)<1:
            raise saveExp
        #evaluate
        
        #set the best as the result
        self._evolvedCls=max(population, key=lambda i: i.score)
        
        generations=1
        while (self._stopAccuracy.value is None or self._stopAccuracy.value>self._evolvedCls.score) and \
               self._generations.value>=generations:
            if self._logGenFitness.value:
                self._logger.log("{}/{} | Actual score: {}".format(generations, self._generations.value, self._evolvedCls.score))
            

            #get new population
            #    performs steps:
            #        selection
            #        crossover
            #        mutation
            #        replacement
            self._generateNewPopulation(population)
            
            #evaluate
            if self._changeTestSet.value:
                #ok new test set requested
                dataset.changeTestSet()
                self._evolvedCls.shouldEvaluate()   #because of the new test set
                for i in population: i.shouldEvaluate()
                
            actBest=max(population, key=lambda i: i.score)
            
            if actBest.score>=self._evolvedCls.score:
                #new best one
                self._evolvedCls=actBest
            
            generations+=1
                
    def _generateNewPopulation(self, population:List[Individual]):
        """
        Generates new population from actual population.
        Whole new population is created and the actual best one is added (elitism).
        
        :param population: Actual population. Manipulates in place.
        :type population: List[Individual]
        """

        selMethod=self.SELECTION_METHODS[self._selectionMethod.value]

        #selection
        parents=selMethod(population,2)   #we are selecting two parents an their positions in population
        
        for _ in range(random.randint(1,self._maxCrossovers.value)):
            #crossover
            if random.uniform(0,1)<=self._crossoverProb.value:
                #the crossover is requested
                child1,child2=Individual.crossover(parents[0][1],parents[1][1])
            else:
                #ok no crossover
                child1=copy.copy(parents[0][1])
                child2=copy.copy(parents[1][1])
            
            #mutate
            child1.mutate(self._maxMutations.value)
            child2.mutate(self._maxMutations.value)
            
            
            #select two best out of children and parents
            bestOfThem=sorted([child1,child2,parents[0][1],parents[1][1]], key=lambda i: i.score, reverse=True)[:2]
            population[parents[0][0]]=bestOfThem[0]  #instead of first parent (could be inserted first parent again)
            population[parents[1][0]]=bestOfThem[1]  #instead of second parent (could be inserted second parent again)

        
    def classify(self, data):
        if self._normalizer.value is not None:
            data=self._normalizer.value.transform(data)
            
            if not isinstance(data, spmatrix):
                #some normalizers may return np array
                #but we need sparse
                data=csr_matrix(data)
        try:
            return self._evolvedCls.classify(data.A)
        except MemoryError:
            #ok lets try the parts
            predicted=np.empty(data.shape[0], dtype=self.trainedClasses.dtype)
            memEr=True
            partSize=data.shape[0]
            while memEr:
                memEr=False
                try:
                    partSize=int(partSize/2)
                    for offset in range(0, data.shape[0], partSize):
                        predicted[offset:offset+partSize]=self._evolvedCls.classify(data[offset:offset+partSize,:].A)
                except MemoryError:
                    memEr=True
    
            return predicted
    
