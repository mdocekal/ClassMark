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

def fNearest2x2FromEachClass(data, vals):
    """
    Finds two nearest from each class and performs weighted average of their values.
    As weight is used distance. If distance from some data sample is zero than it's
    value is returned. Beware that if there is multiple samples with zero distance
    than zero value(outer) haves priority. 
    
    Outer class must have values of 0 and actual class must have non zero values.
    
    
    :param data: Coords for interpolation.
    :type data: List[np.array]
    :param vals: Values on given coords.
    :type vals: List[float]
    """
    #nearest 2x2 (from each class) interpolate
    classData=[]
    classDataInd=[]
    outerData=[]
    outerDataInd=[]
    for i, x in enumerate(data):
        if vals[i]!=0:
            classData.append(x)
            classDataInd.append(i)
        else:
            outerData.append(x)
            outerDataInd.append(i)


    fnClass=cKDTree(classData)
    del classData
    
    fnOuter=cKDTree(outerData)
    del outerData
    
    def res(p):
        #check the nearest

        dC, iC=fnClass.query(p,2)
        dO, _=fnOuter.query(p,2)

        valuesC=[vals[classDataInd[i]] for i in iC]

        numerator=0
        denominator=0
        for d in dO:
            if d==0:
                return 0
            denominator+=1/d
        for i, d in enumerate(dC):
            if d==0:
                return valuesC[i]
            x=1/d
            numerator+=valuesC[i]*x
            denominator+=x

        return numerator/denominator

    return res


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


def fNearest2x2FromEachClass2AtAll(data, vals):
    """
    Finds two nearest from each class, two at all and performs weighted average of their values.
    As weight is used distance. If distance from some data sample is zero than it's
    value is returned.
    
    Outer class must have values of 0 and actual class must have non zero values.
    
    :param data: Coords for interpolation.
    :type data: List[np.array]
    :param vals: Values on given coords.
    :type vals: List[float]
    """


    classData=[]
    classDataInd=[]
    outerData=[]
    outerDataInd=[]
    for i, x in enumerate(data):
        if vals[i]!=0:
            classData.append(x)
            classDataInd.append(i)
        else:
            outerData.append(x)
            outerDataInd.append(i)
            
    fnAll=cKDTree(data)
    
    fnClass=cKDTree(classData)
    del classData
    
    fnOuter=cKDTree(outerData)
    del outerData
    
    

    def res(p):
        #check the nearest
        numerator=0
        denominator=0
        DA, IA = fnAll.query(p,2)
        DA=DA.tolist()
        IA=IA.tolist()

        dC, iC=fnClass.query(p,2)
        for d,i in zip(dC, iC):
            DA.append(d)
            IA.append(classDataInd[i])

        dO, ic=fnOuter.query(p,2)
        for d,i in zip(dO, ic):
            DA.append(d)
            IA.append(outerDataInd[i])

        for dA, iA in zip(DA, IA):
            if dA==0:
                return vals[iA]
            x=1/dA
            numerator+=x*vals[iA]
            denominator+=x

        return numerator/denominator

    return res

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