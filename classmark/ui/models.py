"""
Created on 28. 2. 2019
This module contains models.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from PySide2.QtCore import QAbstractTableModel, Qt
from ..core.experiment import Experiment
from ..core.results import Results
from ..core.validation import Validator
from typing import Callable

class TableSummarizationResultsModel(QAbstractTableModel):
    """
    Model for tableview that is for showing summarized statistics of experiment.
    """
    
    NUM_COL=13
    """Number of columns in table."""
    
    COLL_CLASSIFIER_NAME=0
    """Index of classifier name column."""
    
    COLL_ACCURACY=1
    """Index of column with average accuracy."""
    
    COLL_MICRO_AVG_F1_SCORE=2
    """Index of column with micro avg F1 score."""
    
    COLL_MICRO_AVG_PRECISION_SCORE=3
    """Index of column with micro avg precision score."""
    
    COLL_MICRO_AVG_RECALL_SCORE=4
    """Index of column with micro avg recall score."""
    
    COLL_MACRO_AVG_F1_SCORE=5
    """Index of column with macro avg F1 score."""
    
    COLL_MACRO_AVG_PRECISION_SCORE=6
    """Index of column with macro avg precision score."""
    
    COLL_MACRO_AVG_RECALL_SCORE=7
    """Index of column with macro avg recall score."""
    
    COLL_WEIGHTED_AVG_F1_SCORE=8
    """Index of column with weighted avg F1 score."""
    
    COLL_WEIGHTED_AVG_PRECISION_SCORE=9
    """Index of column with weighted avg precision score."""
    
    COLL_WEIGHTED_AVG_RECALL_SCORE=10
    """Index of column with weighted avg recall score."""
    
    COLL_TRAIN_TIME_SCORE=11
    """Index of column with average train time."""
    
    COLL_TEST_TIME_SCORE=12
    """Index of column with average test time."""
    
    def __init__(self, parent, results:Results):
        """
        Initialization of model.
        
        :param parent: Parent widget.
        :type parent: Widget
        :param results: Experiment results.
        :type results: Results
        """
        QAbstractTableModel.__init__(self, parent)
        self._results = results
        
    
    @property
    def results(self):
        """
        Assigned results.
        """
        return self._results
    
    @results.setter
    def results(self, results:Results):
        """
        Assign new results.
        
        :param results: New results that should be now used.
        :type results: Results
        """
        self._results=results
        self.beginResetModel()
    
    def rowCount(self, parent=None):
        try:
            return len(self._results.classifiers)
        except (AttributeError, TypeError):
            #probably no assigned data stats
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
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable
    
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

        #classifier on current row
        classifier=self._results.classifiers[index.row()]

        if role == Qt.DisplayRole:
            if index.column()==self.COLL_CLASSIFIER_NAME:
                return str(classifier.getName())
            
            if index.column()==self.COLL_ACCURACY:
                return str(self._results.scores[classifier][Results.ScoreType.ACCURACY])
            
            
            
            if index.column()==self.COLL_MICRO_AVG_F1_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MICRO_F1])
            
            if index.column()==self.COLL_MICRO_AVG_PRECISION_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MICRO_PRECISION])
            
            if index.column()==self.COLL_MICRO_AVG_RECALL_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MICRO_RECALL])
            
            
            
            if index.column()==self.COLL_MACRO_AVG_F1_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MACRO_F1])
            
            if index.column()==self.COLL_MACRO_AVG_PRECISION_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MACRO_PRECISION])
            
            if index.column()==self.COLL_MACRO_AVG_RECALL_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MACRO_RECALL])
            
            
            
            if index.column()==self.COLL_WEIGHTED_AVG_F1_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.WEIGHTED_F1])
            
            if index.column()==self.COLL_WEIGHTED_AVG_PRECISION_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.WEIGHTED_PRECISION])
            
            if index.column()==self.COLL_WEIGHTED_AVG_RECALL_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.WEIGHTED_RECALL])
            
            
            
            if index.column()==self.COLL_TRAIN_TIME_SCORE:
                return str(self._results.times[classifier][Validator.TimeDuration.TRAINING_PROC])
            
            if index.column()==self.COLL_TEST_TIME_SCORE:
                return str(self._results.times[classifier][Validator.TimeDuration.TEST_PROC])


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
                self._HEADERS=[self.tr("classifier"),self.tr("accuracy"),
                               self.tr("micro avg F1"),self.tr("micro avg precision"),self.tr("micro avg recall"),
                               self.tr("macro avg F1"),self.tr("macro avg precision"),self.tr("macro avg recall"),
                               self.tr("weighted avg F1"),self.tr("weighted avg precision"),self.tr("weighted avg recall"),
                               self.tr("train time"),self.tr("test time")]
                return self._HEADERS[section]
        
        return None
    

class TableAttributesStatsModel(QAbstractTableModel):
    """
    Model for tableview that is for showing statistics of attributes.
    """
    
    NUM_COL=3
    """Number of columns in table."""
    
    COLL_ATTRIBUTE_NAME=0
    """Index of attribute name column."""
    
    COLL_NUM_OF_FEATURES=1
    """Index of column with number of features for attribute."""
    
    COLL_FEATURES_VARIANCE=2
    """Index of column with features variance."""

    
    def __init__(self, parent, experiment:Experiment):
        """
        Initialization of model.
        
        :param parent: Parent widget.
        :type parent: Widget
        :param experiment: Experiment which attributes you want to show.
        :type experiment: Experiment
        """
        QAbstractTableModel.__init__(self, parent)
        self._experiment = experiment
        
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
            return len(self._experiment.dataStats.attributes)
        except (AttributeError, TypeError):
            #probably no assigned data stats
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
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable
 

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

        #class name on current row
        attributeName=self._experiment.dataStats.attributes[index.row()]

        if role == Qt.DisplayRole:
            if index.column()==self.COLL_ATTRIBUTE_NAME:
                return str(attributeName)
            
            if index.column()==self.COLL_NUM_OF_FEATURES:
                return str(self.experiment.dataStats.attributesFeatures[attributeName])
            
            if index.column()==self.COLL_FEATURES_VARIANCE:
                return str(self.experiment.dataStats.attributesAVGFeatureVariance[attributeName])

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
                self._HEADERS=[self.tr("attribute"),self.tr("number of features"),self.tr("average features variance")]
                return self._HEADERS[section]
        
        return None

class TableClassStatsModel(QAbstractTableModel):
    """
    Model for tableview that is for showing statistics of classes.

    """
    
    NUM_COL=4
    """Number of columns in table."""
    
    COLL_CLASS_NAME=0
    """Index of class name column."""
    
    COLL_SAMPLES=1
    """Index of column with number of samples in that class."""
    
    COLL_SAMPLES_ORIG=2
    """Index of column with original number of samples in that class."""
    
    COLL_USE=3
    """Index of use column."""
    
    
    def __init__(self, parent, experiment:Experiment):
        """
        Initialization of model.
        
        :param parent: Parent widget.
        :type parent: Widget
        :param experiment: Experiment which attributes you want to show.
        :type experiment: Experiment
        """
        QAbstractTableModel.__init__(self, parent)
        self._experiment = experiment

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
            return len(self._experiment.origDataStats.classes)
        except (AttributeError, TypeError):
            #probably no assigned data stats
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
        
        if index.column() in {self.COLL_USE}:
            f|=Qt.ItemIsUserCheckable
            
        if index.column() in {self.COLL_SAMPLES}:
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
        #class name on current row
        className=self._experiment.origDataStats.classes[index.row()]
        
        if role == Qt.CheckStateRole and index.column() in {self.COLL_USE}:
            #checkbox change

            if value ==Qt.Checked:
                self._experiment.dataStats.activateClass(className)
            else:
                self._experiment.dataStats.deactivateClass(className)
            
            self.dataChanged.emit(index, self.COLL_SAMPLES)
        
        elif role==Qt.EditRole:
            if index.column() == self.COLL_SAMPLES:
                if self._experiment.dataStats.isActive(className):
                    #new number of samples
                    try:
                        iVal=int(value)
                    except ValueError:
                        pass
                    else:
                        origVal=self._experiment.origDataStats.classSamples[className]
                        
                        setVal=iVal if iVal<=origVal else origVal
                        setVal=1 if setVal<1 else setVal
                        self._experiment.dataStats.changeSamplesInClass(className,setVal)
                
        
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
        
        #class name on current row
        className=self._experiment.origDataStats.classes[index.row()]

        if role == Qt.DisplayRole or role==Qt.EditRole:
            if index.column()==self.COLL_CLASS_NAME:
                #class name
                return str(className)
            
            if index.column()==self.COLL_SAMPLES:
                if self._experiment.dataStats.isActive(className):
                    return str(self._experiment.dataStats.classSamples[className])
                else:
                    return 0
            
            if index.column()==self.COLL_SAMPLES_ORIG:
                return str(self._experiment.origDataStats.classSamples[className])
            
        if role ==Qt.CheckStateRole:
            if index.column()==self.COLL_USE:
                #use column
                return Qt.Checked if self._experiment.dataStats.isActive(className) else Qt.Unchecked
            
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
                self._HEADERS=[self.tr("class"),self.tr("selected samples"),self.tr("original samples"),
                               self.tr("use")]
                return self._HEADERS[section]
        
        return None
    
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
