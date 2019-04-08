"""
Created on 9. 3. 2019
Usefull utils

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from scipy.sparse import spmatrix

def sparseMatVariance(mat):
    """
    Calculates variance for given spmatrix.
    :param mat: The matrix.
    :type mat: spmatrix
    """
    
    return mat.power(2).mean() - mat.mean()**2