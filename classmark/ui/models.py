"""
Created on 28. 2. 2019
This module contains models.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from PySide2.QtCore import QAbstractTableModel, Qt
from ..core.experiment import Experiment
from typing import Callable

class TableDataAttributesModel(QAbstractTableModel):
    """
    Model for tableview that is for showing dataset attributes.

    """
    
    NUM_COL=6
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
    
    COLL_FEATURE_EXTRACTION_PROPERITES=5
    """Index of feature extraction method properties column."""
    
    
    def __init__(self, parent, experiment:Experiment, showExtractorAttr:Callable[[int],None]=None):
        """
        Initialization of model.
        
        :param parent: Parent widget.
        :type parent: Widget
        :param experiment: Experiment which attributes you want to show.
        :type experiment: Experiment
        :param showExtractorAttr: This method will be called, with parameter containing row number, when extractor is changed.
        :type showExtractorAttr: Callable[[int],None]
        """
        QAbstractTableModel.__init__(self, parent)
        self._experiment = experiment
        self._showExtractorAttr=showExtractorAttr

    @property
    def experiment(self):
        """
        Assigned experiment.
        """
        return self._experiment
    
    @experiment.setter
    def experiment(self, experiment:Experiment):
        """
        Assign new experiment.
        
        :param experiment: New experiment that should be now used.
        :type experiment: Experiment
        """
        self._experiment=experiment
        self.beginResetModel()
        
    def rowCount(self, parent=None):
        try:
            return len(self._experiment.dataset.attributes)
        except AttributeError:
            #probably no assigned data set
            return 0
    
    def columnCount(self, parent=None):
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
        
        #attribute name on current row
        attributeName=self._experiment.dataset.attributes[index.row()]
        
        if role == Qt.CheckStateRole and index.column() in {self.COLL_USE, self.COLL_PATH}:
            #checkbox change
            changeSetting=Experiment.AttributeSettings.USE if index.column()==self.COLL_USE else Experiment.AttributeSettings.PATH
            
            if value ==Qt.Checked:
                self._experiment.setAttributeSetting(attributeName, changeSetting, True)
            else:
                self._experiment.setAttributeSetting(attributeName, changeSetting, False)
        
        elif role==Qt.EditRole:
            if index.column() == self.COLL_LABEL:
                #radio button
                self._experiment.label=attributeName
            elif index.column() == self.COLL_FEATURE_EXTRACTION:
                if self._experiment.getAttributeSetting(attributeName, 
                        Experiment.AttributeSettings.FEATURE_EXTRACTOR).getName()!=value:
                    #we are interested only if there is a change
                    self._experiment.setAttributeSetting(attributeName, 
                        Experiment.AttributeSettings.FEATURE_EXTRACTOR, self._experiment.featuresExt[value]())
                    
                    #Emit attributes click event, because we want to show to user actual feature extractor
                    #attributes.
                    if self._showExtractorAttr is not None:
                        self._showExtractorAttr(index.row())
                
                
        
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
        
        #attribute name on current row
        attributeName=self._experiment.dataset.attributes[index.row()]

        if role == Qt.DisplayRole or role==Qt.EditRole:
            if index.column()==self.COLL_ATTRIBUTE_NAME:
                #attribute name
                return attributeName
            
            if index.column()==self.COLL_LABEL:
                #Is on that index selected label?
                return attributeName==self._experiment.label
            
            if index.column()==self.COLL_FEATURE_EXTRACTION:

                return self._experiment.getAttributeSetting(attributeName, 
                                                            Experiment.AttributeSettings.FEATURE_EXTRACTOR).getName()
        
        if role ==Qt.CheckStateRole:
            if index.column()==self.COLL_USE:
                #use column
                return Qt.Checked if self._experiment.getAttributeSetting(attributeName, Experiment.AttributeSettings.USE) else Qt.Unchecked
            
            if index.column()==self.COLL_PATH:
                #path column
                return Qt.Checked if  self._experiment.getAttributeSetting(attributeName, Experiment.AttributeSettings.PATH) else Qt.Unchecked

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
                self._HEADERS=[self.tr("attributes"),self.tr("use"),self.tr("path"),self.tr("label"),self.tr("features extraction"), ""]
                return self._HEADERS[section]
        
        return None
