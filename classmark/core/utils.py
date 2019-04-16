"""
Created on 9. 3. 2019
Usefull utils

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from scipy.sparse import spmatrix
from functools import wraps


def getAllSubclasses(cls):
    """
    Searches all subclasses of given class.
    
    :param cls: The base class.
    :type cls: class
    """
    
    stack = [cls]
    sub=[]
    while len(stack):
        base = stack.pop()
        for child in base.__subclasses__():
            if child not in sub:
                sub.append(child)
                stack.append(child)
                
    return sub

def sparseMatVariance(mat):
    """
    Calculates variance for given spmatrix.
    :param mat: The matrix.
    :type mat: spmatrix
    """
    
    return mat.power(2).mean() - mat.mean()**2


class Observable(object):
    """
    Implementation of observer like design pattern.
    
    Example Usage:
    
        class A(Observable):
            def __init__(self):
                super().__init__()
                
            @Observable._event("STARTS")
            def startsTheEngine(self):
                ...
                
            @Observable._event("END", True)    #true means that all arguments will be passed to observer
            def endTheEngine(self, data):
                ...
    
        a=A()
        a.registerObserver("STARTS", observerCallbackMethod)
    """
    
    @staticmethod
    def _event(tag, passArguments=False):
        """
        Use this decorator to mark methods that could be observed.
        """
        def tags_decorator(f):
            @wraps(f)
            def funcWrapper(o, *arg, **karg):
                f(o, *arg, **karg)
                if passArguments:
                    o._Observable__notify(tag, *arg, **karg)
                else:
                    o._Observable__notify(tag)
            return funcWrapper
        
        return tags_decorator
    
    def __init__(self):
        self.__observers = {}
        
    def clearObservers(self):
        """
        Clears all observers.
        """
        self.__observers={}
        
    def registerObserver(self, eventTag, observer):
        """
        Register new observer for observable method (_event).
        
        :param eventTag: The tag that is passed as parameter for _event decorator.
        :type eventTag: str
        :param observer: Method that should be called
        :type observer: Callable
        """
        
        s = self.__observers.setdefault(eventTag, set())
        s.add(observer)
        
    def unregisterObserver(self, eventTag, observer):
        """
        Unregister observer for observable method (_event).
        
        :param eventTag: The tag that is passed as parameter for _event decorator.
        :type eventTag: str
        :param observer: Method that should no longer be called
        :type observer: Callable
        """
        
        try:
            self.__observers[eventTag].remove(observer)
        except KeyError:
            pass
        
        
    def __notify(self, eventTag, *arg, **kw):
        """
        Notify all obervers for given method.
        
        :param eventTag: The tag that is passed as parameter for _event decorator.
        :type eventTag: str
        """
        try:
            for o in self.__observers[eventTag]:
                o(*arg, **kw)
        except KeyError:
            pass
    