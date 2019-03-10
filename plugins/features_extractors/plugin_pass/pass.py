"""
Created on 4. 3. 2019
Feature extractor plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor

from scipy.sparse import csr_matrix

class Pass(FeatureExtractor):
    """
    Pass feature extractor plugin for ClassMark.
    This extractor just returns the input data in scipy.sparse matrix (float) format.

    """
    
    @staticmethod
    def getName():
        return "Pass"
    
    @staticmethod
    def getNameAbbreviation():
        return "Pass"
 
    @staticmethod
    def getInfo():
        return ""
    
    def fit(self, data, labels=None):
        """
        This function is there only for compatibility with FeatureExtractor.
        There is really no need to call it, but if you do so, than this is just empty
        operation.
        
        :param data: It does not matter.
        :type data: It does not matter.
        :param labels: It does not matter.
        :type labels: It does not matter.
        """
        pass

    def extract(self, data):
        """
        This extractor just returns the input data in scipy.sparse matrix (float) format.
        
        :param data: Original data for features extraction.
        :type data: ArrayLike
        :return: Converted data.
        :rtype: scipy.sparse.spmatrix
        """

        return csr_matrix(data, dtype=float)

    def fitAndExtract(self, data, labels=None):
        return self.extract(data)

        