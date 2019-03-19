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
    :type data:
    :param vals: Values on given coords.
    :type vals:
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
    :type data:
    :param vals: Values on given coords.
    :type vals:
    """

    f = LinearNDInterpolator(data, vals)
    fn=NearestNDInterpolator(data, vals)

    def res(x,y):
        v=f(x,y)
        if math.isnan(v):
            return fn(x,y)
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
    :type data:
    :param vals: Values on given coords.
    :type vals:
    """
    #nearest 2x2 (from each class) interpolate
    classData=[x for i, x in enumerate(data) if vals[i]!=0]
    outerData=[x for i, x in enumerate(data) if vals[i]==0]

    fnClass=cKDTree(classData)
    fnOuter=cKDTree(outerData)

    def res(x,y):
        #check the nearest
        p=[x,y]
        dC, iC=fnClass.query(p,2)
        dO, _=fnOuter.query(p,2)

        valuesC=[vals[data.index(classData[i])] for i in iC]

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
    :type data:
    :param vals: Values on given coords.
    :type vals:
    """

    f = LinearNDInterpolator(data, vals)
    fN= fNearest2x2FromEachClass(data, vals)
    def res(x,y):
        v=f(x,y)
        if math.isnan(v):
            return fN(x,y)
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
    :type data:
    :param vals: Values on given coords.
    :type vals:
    """

    classData=[x for i, x in enumerate(data) if vals[i]!=0]
    outerData=[x for i, x in enumerate(data) if vals[i]==0]
    fnAll=cKDTree(data)
    fnClass=cKDTree(classData)
    fnOuter=cKDTree(outerData)

    def res(x,y):
        #check the nearest
        p=[x,y]
        numerator=0
        denominator=0
        DA, IA = fnAll.query(p,2)
        DA=list(DA)
        IA=list(IA)

        dC, iC=fnClass.query(p,2)
        for d,i in zip(dC, iC):
            DA.append(d)
            IA.append(data.index(classData[i]))

        dO, ic=fnOuter.query(p,2)
        for d,i in zip(dO, ic):
            DA.append(d)
            IA.append(data.index(outerData[i]))

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
    :type data:
    :param vals: Values on given coords.
    :type vals:
    """

    f = LinearNDInterpolator(data, vals)
    fN= fNearest2x2FromEachClass2AtAll(data, vals)
    def res(x,y):
        v=f(x,y)
        if math.isnan(v):
            return fN(x,y)
        else:
            return v
    return res