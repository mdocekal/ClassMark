"""
Created on 1. 4. 2019
This module contains functors for parent selection in evolution algorithm.

This module could be used for auto importing in a way:
     FUNCTIONS=[o for o in Selector.__subclasses__()]

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""


from abc import ABC, abstractmethod
from typing import List, Any
import random
from .individual import Individual

class Selector(ABC):
    """
    Base functor for all evolution selectors.
    """
    
    @abstractmethod
    def __call__(self, population:List[Individual], n):
        """
        For every selection algorithm we define common interface that requires
        individuals and their scores.
        Every selection algorithms selects n individuals from population.
        
        :param population: Individuals for selection.
        :type population: List[Individual]
        :param n: Number of selected individuals. One individual could be selected multiple times.
        :type n: int
        :param theirScore: Score of each individual.
        :type theirScore: List[float]
        :return: Selected individuals.
        :rtype: List[Individual]
        """
        pass
        
class Rulete(Selector):
    """
    Rulete selector
    """
    def __call__(self, population:List[Individual],n):
        s=sum(i.score for i in population)
        res=[]
        while n>0:
            #select the value
            sel=random.uniform(0, s)
            
            c=0
            for individual in population:
                c+= individual.score
                
                if sel<=c:
                    res.append(individual)
                    n-=1
                    break
        
        return res

class Rank(Selector):
    """
    Rank selection.
    """
    
    def __call__(self, population:List[Individual],n):
        s=sorted(population, key=lambda i: i.score)
        
        #count the sum of rank sequence
        rankSum=len(population)*(1+len(population))/2
        
        res=[]
        while n>0:
            #select the rank
            sel=random.uniform(0, rankSum)
            
            #Example:
            #    Ranks: 1 2 3 4
            #    Intervals: 0-1 1-3 3-6 6-10
            
            c=0
            for rank,i in enumerate(s):
                c+= rank+1
                
                if sel<=c:
                    res.append(i)
                    n-=1
                    break
        
        return res
    