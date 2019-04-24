"""
Created on 19. 4. 2019
Module for logger class.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .utils import Observable, Singleton

class Logger(Observable, metaclass=Singleton):
    """
    This singleton class is useful for broadcasting logs to observes that registers to
    log method.
    
    If you want to log something just call Logger().log("something") and all 
    observers, registered with Logger().registerObserver("LOG", observerCallbackMethod) method,
    will be called.
    
    """
    
    @Observable._event("LOG", True)
    def log(self, txt:str):
        """
        Make a log.
        
        :param txt: Text of the log.
        :type txt: str
        """
        pass
    
    