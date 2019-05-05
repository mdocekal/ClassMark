"""
Created on 19. 3. 2019
This module contains functions that are useful for estimating likelihood that given vector is in a class.

This module could be used for auto importing in a way:
     FUNCTIONS=[o for o in getmembers(functions) if isfunction(o[1])]

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from scipy.spatial import cKDTree
import numpy as np

def fNearest(samples, samplesVals):
    """
    Linear interpolation according to nearest neighbour.
    
    :param samples: Coords for interpolation.
    :type samples: np.array
    :param samplesVals: Values on class coords.
    :type samplesVals: np.array
    """

    fnAll=cKDTree(samples)
    
    def res(p):
        #check the nearest
        _, IA = fnAll.query(p,1)
        return samplesVals[IA]
    return res
    
def fNearest2x2FromEachClass(samples, samplesVals):
    """
    Finds two nearest from class and two from outer samples and performs weighted average of their values.
    As weight is used distance. If distance from some data sample is zero than it's
    value is returned. Beware that if there is multiple samples with zero distance
    than zero value(outer) haves priority. 
    
    Outer class have values that are equal or smaller than 0 (by default) and actual class must have values greater than zero.

    :param samples: Coords for interpolation.
    :type samples: np.array
    :param samplesVals: Values on class coords.
    :type samplesVals: np.array
    """
    
    cInd=np.where(samplesVals>0)
    classData=samples[cInd]
    classVals=samplesVals[cInd]
    haveClassData=classData.shape[0]>0
    
    oInd=np.where(samplesVals<=0)
    outerData=samples[oInd]
    outerVals=samplesVals[oInd]
    haveOuterData=outerData.shape[0]>0

    #nearest 2x2 (from each class) interpolate

    if haveClassData:
        fnClass=cKDTree(classData)
        fnClassMaxNeigh=1 if classData.shape[0]<2 else 2
    
    if haveOuterData:
        fnOuter=cKDTree(outerData)
        fnOuterMaxNeigh=1 if outerData.shape[0]<2 else 2
    
    def res(p):
        #check the nearest

        if haveClassData:
            dC, iC=fnClass.query(p,fnClassMaxNeigh)
            if fnClassMaxNeigh==1:
                #we need col vectors
                dC=dC[:, np.newaxis]
                iC=iC[:, np.newaxis]
        if haveOuterData:
            dO, oC=fnOuter.query(p,fnOuterMaxNeigh)
            if fnOuterMaxNeigh==1:
                #we need col vectors
                dO=dO[:, np.newaxis]
                oC=oC[:, np.newaxis]
        if haveClassData and haveOuterData:
            values=np.hstack((classVals[iC],outerVals[oC]))
            del iC
            del oC
            distances=np.hstack((dC,dO))

        elif haveClassData:
            values=classVals[iC]
            del iC
            distances=dC
        else:
            #only outer remains
            values=outerVals[oC]
            del oC
            distances=dO

        with np.errstate(divide='ignore',invalid='ignore'):
            #we want to detect zero distance values
            #this values will show as inf in 1/distances and nans in avg
            distances=1./distances
            
            avg=np.average(values, axis=1, weights=distances)
            
            #find problems, if exists
            problems=np.where(np.isnan(avg))
            if problems[0].shape[0]>0:
                problemsCols=(problems[0],np.array(np.argmax(np.isinf(distances[problems]),axis=1)))    #we are interested in the first only
                
                #change the nans with values of the problematic points
                avg[problems]=values[problemsCols]
            
            return avg
        
    return res

def fNearest2x2FromEachClass2AtAll(samples, samplesVals):
    """
    Finds two nearest from each class(outer and act. class), two at all and performs weighted average of their values.
    As weight is used distance. If distance from some data sample is zero than it's
    value is returned.
    
    Outer class have values that are equal or smaller than 0 (by default) and actual class must have values greater than zero.

    :param samples: Coords for interpolation.
    :type samples: np.array
    :param samplesVals: Values on class coords.
    :type samplesVals: np.array
    """

    cInd=np.where(samplesVals>0)
    classData=samples[cInd]
    classVals=samplesVals[cInd]
    haveClassData=classData.shape[0]>0
        
    
    oInd=np.where(samplesVals<=0)
    outerData=samples[oInd]
    outerVals=samplesVals[oInd]
    haveOuterData=outerData.shape[0]>0
  
    
    fnAll=cKDTree(samples)
    fnAllMaxNeigh=1 if samplesVals.shape[0]<2 else 2
    
    if haveClassData:
        fnClass=cKDTree(classData)
        fnClassMaxNeigh=1 if classData.shape[0]<2 else 2
    
    if haveOuterData:
        fnOuter=cKDTree(outerData)
        fnOuterMaxNeigh=1 if outerData.shape[0]<2 else 2
    
    def res(p):
        #check the nearest

        DA, IA = fnAll.query(p,fnAllMaxNeigh)
        if fnAllMaxNeigh==1:
            #we need col vectors
            DA=DA[:, np.newaxis]
            IA=IA[:, np.newaxis]
            
        if haveClassData:
            dC, iC=fnClass.query(p,fnClassMaxNeigh)
            if fnClassMaxNeigh==1:
                #we need col vectors
                dC=dC[:, np.newaxis]
                iC=iC[:, np.newaxis]
            
        if haveOuterData:
            dO, oC=fnOuter.query(p,fnOuterMaxNeigh)
            if fnOuterMaxNeigh==1:
                #we need col vectors
                dO=dO[:, np.newaxis]
                oC=oC[:, np.newaxis]
            
        #compile data we have
        if haveClassData and haveOuterData:
            values=np.hstack((samplesVals[IA],classVals[iC],outerVals[oC]))
            del iC
            del oC
            distances=np.hstack((DA,dC,dO))
        elif haveClassData:
            values=np.hstack((samplesVals[IA],classVals[iC]))
            del iC
            distances=np.hstack((DA,dC))
        else:
            #we have just outer not class
            values=np.hstack((samplesVals[IA],outerVals[oC]))
            del oC
            distances=np.hstack((DA,dO))
        
        del IA
        
        
        with np.errstate(divide='ignore',invalid='ignore'):
            #we want to detect zero distance values
            #this values will show as inf in 1/distances and nans in avg
            distances=1./distances
            avg=np.average(values, axis=1, weights=distances)
            
            #find problems, if exists
            problems=np.where(np.isnan(avg))
            if problems[0].shape[0]>0:
                problemsCols=(problems[0],np.array(np.argmax(np.isinf(distances[problems]),axis=1)))    #we are interested in the first only
                #change the nans with values of the problematic points
                avg[problems]=values[problemsCols]
            
            return avg

    return res
