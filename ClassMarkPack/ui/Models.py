"""
Created on 28. 2. 2019
This module contains models.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from PySide2.QtCore import QAbstractTableModel, Qt
from ..data.DataSet import DataSet

class TableDataAttributesModel(QAbstractTableModel):
    """
    Model for tableview that is for showing dataset attributes.

    """
    
    NUM_COL=5
    """Number of columns in table."""
    
    COLL_ATTRIBUTE_NAME=0
    """Index of attribute name column."""
    
    COLL_USE=1
    """Index of use column."""
    
    COLL_PATH=2
    """Index of path column."""
    
    COLL_LABEL=3
    """Index of label column."""
    
    COLL_FEATURE_EXTRACTION=4
    """Index of feature extraction method column."""
    
    
    def __init__(self, parent, dataSet:DataSet):
        """
        Initialization of model.
        
        :param parent: Parent widget.
        :type parent: Widget
        :param dataSet: Dataset which attributes you want to show.
        :type dataSet: DataSet | None
        """
        QAbstractTableModel.__init__(self, parent)
        self._dataSet = dataSet
        self._init()
        
        
    def _init(self):
        """
        Data initialization of the model.
        """
        attrNum=0
        if self._dataSet is not None:
            attrNum=len(self._dataSet.attributes)  
        #flag that determining if given attribute should be used
        self._use=[True]* attrNum
            
        #flag that determining if given attribute value is path to file that should be read
        #and its content should be used instead of the path
        self._path=[False]*attrNum
            
        #name of attribute which should be used as label
        self._label=None 
            
        #feature extractor which is assigned to an attribute
        self._featureExt=[None]*attrNum
 
    @property
    def dataSet(self):
        """
        Assigned data set.
        """
        return self._dataSet
    
    @dataSet.setter
    def dataSet(self, dataSet:DataSet):
        """
        Assign new data set.
        
        :param dataSet: New data set that should be now used.
        :type dataSet:DataSet
        """
        self._dataSet=dataSet
        self._init()
        self.beginResetModel()
        
    def rowCount(self, parent):
        try:
            return len(self._dataSet.attributes)
        except AttributeError:
            #probably no assigned data set
            return 0
    
    def columnCount(self, parent):
        return self.NUM_COL
    
    def flags(self, index):
        """
        Determine flag for column on given index.
        
        :param index: Index containing row and col.
        :type index: QModelIndex
        :return: Flag for indexed cell.
        :rtype: PySide2.QtCore.Qt.ItemFlags
        """
        f= Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() in {self.COLL_USE, self.COLL_PATH}:
            f|=Qt.ItemIsUserCheckable
        
        if index.column() == self.COLL_LABEL:
            f|=Qt.ItemIsEditable
        return f
    
    def setData(self, index, value, role=Qt.EditRole):
        """
        Set new data on given index.
        
        :param index: Index of the cell.
        :type index: QModelIndex
        :param value: New value.
        :type value: object
        :param role: Cell role.
        :type role: int
        """

        if not index.isValid():
            return False
        
        if role == Qt.CheckStateRole and index.column() in {self.COLL_USE, self.COLL_PATH}:
            #checkbox change
            writeTo=self._use if index.column()==self.COLL_USE else self._path
            
            if value ==Qt.Checked:
                writeTo[index.row()]=True
            else:
                writeTo[index.row()]=False
        
        elif role==Qt.EditRole and index.column() == self.COLL_LABEL:
            #radio button
            self._label=index.row()
        
        self.dataChanged.emit(index, index)
        return True
    
    def data(self, index, role):
        """
        Getter for content of the table.
        
        :param index: Index containing row and col.
        :type index: QModelIndex
        :param role: Cell role.
        :type role: int
        :return: Data for indexed cell.
        :rtype: object
        """

        if not index.isValid():
            return None
        
        if role == Qt.DisplayRole or role==Qt.EditRole:
            if index.column()==self.COLL_ATTRIBUTE_NAME:
                #attribute name
                return self._dataSet.attributes[index.row()]
            
            if index.column()==self.COLL_LABEL:
                #Is on that index selected label?
                return index.row()==self._label
        
        if role ==Qt.CheckStateRole:
            if index.column()==self.COLL_USE:
                #use column
                return Qt.Checked if self._use[index.row()] else Qt.Unchecked
            
            if index.column()==self.COLL_PATH:
                #path column
                return Qt.Checked if self._path[index.row()] else Qt.Unchecked

        return None
        
    
    def headerData(self, section, orientation, role):
        """
        Data for header cell.
        
        :param section: Header column.
        :type section: PySide2.QtCore.int
        :param orientation: Table orientation.
        :type orientation: PySide2.QtCore.Qt.Orientation
        :param role: Role of section.
        :type role: PySide2.QtCore.int
        :return: Data for indexed header cell.
        :rtype: object
        """
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            #we have horizontal header only.
            try:
                return self._HEADERS[section]
            except AttributeError:
                """Name of columns in table. Initialization is performed on demand."""
                self._HEADERS=[self.tr("attributes"),self.tr("use"),self.tr("path"),self.tr("label"),self.tr("feature extraction")]
                return self._HEADERS[section]
        
        return None
