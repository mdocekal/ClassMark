"""
Created on 15. 4. 2019
Histogram of Oriented Gradients plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from classmark.core.plugins import FeatureExtractor, PluginAttribute, PluginAttributeIntChecker
from skimage.feature import hog
from scipy.sparse import csr_matrix

class HOG(FeatureExtractor):
    """
    HOG feature extractor plugin for ClassMark.
    """
    def __init__(self, orientationsBins:int=9, pixelsPerCellHorizontal:int=8, \
                 pixelsPerCellVertical:int=8, cellsPerBlockHorizontal:int=3, \
                 cellsPerBlockVertical:int=3, blockNorm:str="L2-Hys"):
        """
        Feature extractor initialization.
        
        :param orientationsBins: Number of orientation bins.
        :type orientationsBins: int
        :param pixelsPerCellHorizontal: Width (in pixels) of a cell.
        :type pixelsPerCellHorizontal: int
        :param pixelsPerCellVertical: Height (in pixels) of a cell.
        :type pixelsPerCellVertical: int
        :param cellsPerBlockHorizontal: Number of cells in each block (horizontal).
        :type cellsPerBlockHorizontal: int
        :param cellsPerBlockVertical: Number of cells in each block (vertical).
        :type cellsPerBlockVertical: int
        :param blockNorm: Block normalization method.
        :type blockNorm: str
        """
        
        self._orientationsBins=PluginAttribute("Number of orientation bins", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._orientationsBins.value=orientationsBins
        
        self._pixelsPerCellHorizontal=PluginAttribute("Width of a cell [px]", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._pixelsPerCellHorizontal.value=pixelsPerCellHorizontal
        
        self._pixelsPerCellVertical=PluginAttribute("Height of a cell [px]", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._pixelsPerCellVertical.value=pixelsPerCellVertical
        
        self._cellsPerBlockHorizontal=PluginAttribute("Number of cells in each block (horizontal)", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._cellsPerBlockHorizontal.value=cellsPerBlockHorizontal
        
        self._cellsPerBlockVertical=PluginAttribute("Number of cells in each block (vertical)", PluginAttribute.PluginAttributeType.VALUE, PluginAttributeIntChecker(minV=1))
        self._cellsPerBlockVertical.value=cellsPerBlockVertical
        
        self._blockNorm=PluginAttribute("Normalization", PluginAttribute.PluginAttributeType.SELECTABLE, None,
                                            ["L1","L1-sqrt","L2","L2-Hys"])
        self._blockNorm.value=blockNorm
        
    
    @staticmethod
    def getName():
        return "Histogram of Oriented Gradients"
    
    @staticmethod
    def getNameAbbreviation():
        return "HOG"
 
    @staticmethod
    def getInfo():
        return ""
    
    @classmethod
    def expDataType(cls):
        """
        Expected data type for extraction.
        Overwrite this method if you want to use different data type.
        """
        return cls.DataTypes.IMAGE
    
    def fit(self, data, labels=None):
        """
        This extractor is stateless so this operation is empty.
        """
        pass
    
    def extract(self, data):
        res=[]

        for lReader in data:
            data=lReader.getRGB()
            res.append( hog(data, 
                            orientations=self._orientationsBins.value, 
                            pixels_per_cell=(self._pixelsPerCellHorizontal.value, self._pixelsPerCellVertical.value), 
                            cells_per_block=(self._cellsPerBlockHorizontal.value, self._cellsPerBlockVertical.value), 
                            block_norm=self._blockNorm.value, 
                            feature_vector=True, multichannel=len(data.shape)>2))
        return csr_matrix(res)
    
    def fitAndExtract(self, data, labels=None):
        return self.extract(data)