"""
Created on 19. 3. 2019
This module contains functions that are useful for estimating likelihood that given vector is in a class.

This module could be used for auto importing in a way:
     FUNCTIONS=[o for o in getmembers(functions) if isfunction(o[1])]

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import cKDTree
import math
import numpy as np

'''
def fLinInterExtWithConst(data, vals, c=0):
    """
    Performs linear interpolation inside convex hull that is defined be
    provided data. Outside convex hull it is doing "extrapolation" with constant
    (every value outside is considered to be c).
    
    :param data: Coords for interpolation.
    :type data: List[np.array]
    :param vals: Values on given coords.
    :type vals: List[float]
    :param c: Constant for extrapolation.
    :type c: float
    """

    return LinearNDInterpolator(data, vals, fill_value=c )

def fLinInterExtNearest(data, vals):
    """
    Performs linear interpolation inside convex hull that is defined be
    provided data. Outside convex hull it is doing extrapolation by finding nearest
    known value.
    
    :param data: Coords for interpolation.
    :type data: List[np.array]
    :param vals: Values on given coords.
    :type vals: List[float]
    """

    f = LinearNDInterpolator(data, vals)
    fn=NearestNDInterpolator(data, vals)

    def res(p):
        v=f(p)
        if math.isnan(v):
            """
            TODO:
            Traceback (most recent call last):
  File "/home/windionleaf/Development/python/ClassMark/classmark/core/experiment.py", line 385, in run
    for step, (predicted, realLabels) in enumerate(self._experiment.evaluationMethod.run(c, data, labels, extMap)):
  File "/home/windionleaf/Development/python/ClassMark/classmark/core/validation.py", line 62, in run
    classifier.train(trainFeatures, trainLabels)
  File "/home/windionleaf/Development/python/ClassMark/plugins/classifiers/plugin_ceef/ceef/ceef.py", line 124, in train
    self._evolvedCls=max(population, key=lambda i: i.score)
  File "/home/windionleaf/Development/python/ClassMark/plugins/classifiers/plugin_ceef/ceef/ceef.py", line 124, in <lambda>
    self._evolvedCls=max(population, key=lambda i: i.score)
  File "/home/windionleaf/Development/python/ClassMark/plugins/classifiers/plugin_ceef/ceef/individual.py", line 78, in score
    if self.predict(self._dataSet.data[sampleInd].todense().A1)==self._dataSet.labels[sampleInd]:
  File "/home/windionleaf/Development/python/ClassMark/plugins/classifiers/plugin_ceef/ceef/individual.py", line 96, in predict
    predicted[i] = fg.fenotype(sample)
  File "/home/windionleaf/Development/python/ClassMark/plugins/classifiers/plugin_ceef/ceef/functions.py", line 50, in res
    return fn(p)
  File "/home/windionleaf/.local/lib/python3.6/site-packages/scipy/interpolate/ndgriddata.py", line 81, in __call__
    return self.values[i]
TypeError: only integer scalar arrays can be converted to a scalar index

            """
            return fn(p)
        else:
            return v
    return res
'''
def fNearest2x2FromEachClass(classData, outerData, classVals):
    """
    Finds two nearest from each class and performs weighted average of their values.
    As weight is used distance. If distance from some data sample is zero than it's
    value is returned. Beware that if there is multiple samples with zero distance
    than zero value(outer) haves priority. 
    
    Outer class have values of 0 (by default) and actual class must have non zero values.
    
    
    :param classData: Class coords for interpolation.
    :type classData: np.array
    :param outerData: Outer coords for interpolation.
    :type outerData: np.array
    :param classVals: Values on class coords.
    :type classVals: np.array
    """
    #nearest 2x2 (from each class) interpolate

    fnClass=cKDTree(classData)
    fnClassMaxNeigh=1 if classData.shape[0]<2 else 2
    
    fnOuter=cKDTree(outerData)
    fnOuterMaxNeigh=1 if outerData.shape[0]<2 else 2
    
    def res(p):
        #check the nearest

        dC, iC=fnClass.query(p,fnClassMaxNeigh)
        dO, _=fnOuter.query(p,fnOuterMaxNeigh)

        values=np.hstack((classVals[iC],np.zeros_like(dO)))
        del iC
        distances=np.hstack((dC,dO))

        with np.errstate(divide='ignore',invalid='ignore'):
            #we want to detect zero distance values
            #this values will show as inf in 1/distances and nans in avg
            distances=1./distances
            avg=np.average(values, axis=1, weights=distances)
            
            #find problems, if exists
            problems=np.where(np.isnan(avg))
            if problems[0].shape[0]>0:
                problemCols=np.where(np.isinf(distances[problems]))
                
                #change the nans with values of the problematic points
                avg[problems]=values[problems][problemCols]
            
            return avg
        
    return res

'''
def fLinInterExtNearest2x2(data, vals):
    """
    Performs linear interpolation inside convex hull that is defined be
    provided data. Outside convex hull it is doing extrapolation with
    fNearest2x2FromEachClass function.
    
    Outer class must have values of 0 and actual class must have non zero values.
    
    :param data: Coords for interpolation.
    :type data: List[np.array]
    :param vals: Values on given coords.
    :type vals: List[float]
    """

    f = LinearNDInterpolator(data, vals)
    fN= fNearest2x2FromEachClass(data, vals)
    def res(p):
        v=f(p)
        if math.isnan(v):
            return fN(p)
        else:
            return v
    return res

'''
def fNearest2x2FromEachClass2AtAll(classData, outerData, classVals):
    """
    Finds two nearest from each class, two at all and performs weighted average of their values.
    As weight is used distance. If distance from some data sample is zero than it's
    value is returned.
    
    Outer class must have values of 0 and actual class must have non zero values.
    
    :param classData: Class coords for interpolation.
    :type classData: np.array
    :param outerData: Outer coords for interpolation.
    :type outerData: np.array
    :param classVals: Values on class coords.
    :type classVals: np.array
    """


  
    fnAll=cKDTree(np.vstack((classData,outerData)))
    valsAll=np.hstack((classVals,np.zeros(outerData.shape[0])))
    fnAllMaxNeigh=1 if valsAll.shape[0]<2 else 2
    
    fnClass=cKDTree(classData)
    fnClassMaxNeigh=1 if classData.shape[0]<2 else 2
    
    fnOuter=cKDTree(outerData)
    fnOuterMaxNeigh=1 if outerData.shape[0]<2 else 2
    
    def res(p):
        #check the nearest

        DA, IA = fnAll.query(p,fnAllMaxNeigh)
        dC, iC=fnClass.query(p,fnClassMaxNeigh)
        dO, _=fnOuter.query(p,fnOuterMaxNeigh)
        
        values=np.hstack((valsAll[IA],classVals[iC],np.zeros_like(dO)))
        del iC
        del IA
        distances=np.hstack((DA,dC,dO))
        
        with np.errstate(divide='ignore',invalid='ignore'):
            #we want to detect zero distance values
            #this values will show as inf in 1/distances and nans in avg
            distances=1./distances
            avg=np.average(values, axis=1, weights=distances)
            
            #find problems, if exists
            problems=np.where(np.isnan(avg))
            if problems[0].shape[0]>0:
                problemCols=np.where(np.isinf(distances[problems]))
                
                #change the nans with values of the problematic points
                avg[problems]=values[problems][problemCols]
            
            return avg

    return res
'''
def fLinInterExtNearest2x2FromEachClass2AtAll(data, vals):
    """
    Performs linear interpolation inside convex hull that is defined be
    provided data. Outside convex hull it is doing extrapolation with
    fNearest2x2FromEachClass2AtAll function.
    
    Outer class must have values of 0 and actual class must have non zero values.
    
    :param data: Coords for interpolation.
    :type data: List[np.array]
    :param vals: Values on given coords.
    :type vals: List[float]
    """

    f = LinearNDInterpolator(data, vals)
    fN= fNearest2x2FromEachClass2AtAll(data, vals)
    def res(p):
        v=f(p)
        if math.isnan(v):
            return fN(p)
        else:
            return v
    return res
'''