"""
Created on 28. 2. 2019
This module contains models.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from PySide2.QtCore import QAbstractTableModel, QAbstractListModel, Qt
from PySide2.QtGui import QBrush, QColor
from ..core.experiment import Experiment, LastUsedExperiments
from ..core.results import Results
from ..core.validation import Validator
from ..core.plugins import Classifier
from typing import Callable

import csv


def saveTableModelAsCsv(model: QAbstractTableModel, path: str):
    """
    Saves table model on given path as csv.
    
    Takes just horizontal header.
    
    :param model: Model you want to save.
    :type model: QAbstractTableModel
    :param path: Path where model should be saved.
    :type path: str
    """
    with open(path, 'w') as f:
        writer = csv.writer(f)

        # header
        writer.writerow([model.headerData(i, Qt.Horizontal, Qt.DisplayRole) for i in range(model.columnCount())])

        for row in range(model.rowCount()):
            writer.writerow(
                [model.data(model.index(row, column), Qt.DisplayRole) for column in range(model.columnCount())])


class ListLastExperiments(QAbstractListModel):
    """
    Model for last experiments.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # when the list changes
        LastUsedExperiments().registerObserver("CHANGE", self.endResetModel)

    def rowCount(self, parent=None):
        return len(LastUsedExperiments().list)

    def data(self, index, role=Qt.DisplayRole):
        """
        Getter for content of the list.
        
        :param index: Index.
        :type index: QModelIndex
        :param role: Cell role.
        :type role: int
        :return: Data for index.
        :rtype: object
        """
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return str(LastUsedExperiments().list[index.row()])
        return None

    def flags(self, index):
        """
        Determine flag for column on given index.
        
        :param index: Index containing row and col.
        :type index: QModelIndex
        :return: Flag for indexed cell.
        :rtype: PySide2.QtCore.Qt.ItemFlags
        """
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable


class TableResultsModel(QAbstractTableModel):
    """
    Model for tableview that is for showing results of experiment.
    """

    NUM_COL = 4
    """Number of columns in table without important features."""

    COLL_ROW_INDEX = 0
    """Index of row data index column."""

    COLL_VAL_STEP = 1
    """Index of validation step index column."""

    COLL_REAL_LABEL = 2
    """Index of real label column."""

    COLL_PREDICTED_LABEL = 3
    """Index of predicted label column."""

    def __init__(self, parent, results: Results, classifier: Classifier):
        """
        Initialization of model.
        
        :param parent: Parent widget.
        :type parent: Widget
        :param results: Experiment results.
        :type results: Results
        :param classifier: Confusion matrix will be shown for this classifier.
        :type classifier: Classifier
        """
        QAbstractTableModel.__init__(self, parent)
        self._results = results
        self._classifier = classifier
        self._numOfResults = sum(s.numOfPredictedLabels for s in self.results.steps)

    @property
    def results(self):
        """
        Assigned results.
        """
        return self._results

    @results.setter
    def results(self, results: Results):
        """
        Assign new results.
        
        :param results: New results that should be now used.
        :type results: Results
        """
        self.beginResetModel()
        self._results = results

        self._numOfResults = sum(s.numOfPredictedLabels for s in self.results.steps)
        self.endResetModel()

    def rowCount(self, parent=None):
        return self._numOfResults

    def columnCount(self, parent=None):
        topFeatures = self._results.steps[0].importantFeaturesForCls(self._classifier)
        return self.NUM_COL + (0 if topFeatures is None else topFeatures.shape[1])

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

        if role == Qt.BackgroundRole or role == Qt.DisplayRole:
            # in which validation step this row is?
            step = 0
            stepSum = 0

            # we need to transform row index to index in current step
            useIndex = index.row()

            for s in self.results.steps:
                stepSum += s.numOfPredictedLabels
                if index.row() < stepSum:
                    break

                useIndex -= s.numOfPredictedLabels
                step += 1

            stepNum = step + 1
            step = self._results.steps[step]
            if role == Qt.DisplayRole:

                if useIndex < 0: useIndex = index.row()

                if index.column() < self.NUM_COL:
                    if index.column() == self.COLL_ROW_INDEX:
                        return str(step.testIndicesForCls(self._classifier)[useIndex] + 1)

                    if index.column() == self.COLL_VAL_STEP:
                        return stepNum

                    if index.column() == self.COLL_REAL_LABEL:
                        return str(self.results.encoder.inverse_transform([step.labels[useIndex]])[0])

                    if index.column() == self.COLL_PREDICTED_LABEL:
                        return str(
                            self.results.encoder.inverse_transform([step.predictionsForCls(self._classifier)[useIndex]])[0])
                else:
                    # columns for top features
                    return f"{step.importantFeaturesForCls(self._classifier)[useIndex][index.column()-self.NUM_COL]} | " \
                           f"{step.importanceOfFeaturesForCls(self._classifier)[useIndex][index.column()-self.NUM_COL]}"

            if role == Qt.BackgroundRole:
                if step.labels[useIndex] != step.predictionsForCls(self._classifier)[useIndex]:
                    return QBrush(QColor(255, 0, 0, 80));

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
            # we have horizontal header only.
            try:
                return self._HEADERS[section]
            except AttributeError:
                """Name of columns in table. Initialization is performed on demand."""
                self._HEADERS = [self.tr("data row index"), self.tr("validation step"), self.tr("real label"),
                                 self.tr("predicted label")]

                for topK in range(1, (self.columnCount() - self.NUM_COL)+1):
                    # all above self.NUM_COL are columns for top-k features
                    self._HEADERS.append(f"{topK}. top feature")
                return self._HEADERS[section]

        return None


class TableConfusionMatrixModel(QAbstractTableModel):
    """
    Model for tableview that is for showing confusion matrix of experiment.
    """

    def __init__(self, parent, results: Results, classifier: Classifier):
        """
        Initialization of model.
        
        :param parent: Parent widget.
        :type parent: Widget
        :param results: Experiment results.
        :type results: Results
        :param classifier: Confusion matrix will be shown for this classifier.
        :type classifier: Classifier
        """
        QAbstractTableModel.__init__(self, parent)
        self._results = results
        self._classifier = classifier

    @property
    def results(self):
        """
        Assigned results.
        """
        return self._results

    @results.setter
    def results(self, results: Results):
        """
        Assign new results.
        
        :param results: New results that should be now used.
        :type results: Results
        """
        self.beginResetModel()
        self._results = results
        self.endResetModel()

    def rowCount(self, parent=None):
        try:
            return self._results.confusionMatrix(self._classifier).shape[0]
        except (AttributeError, TypeError):
            # probably no assigned data stats
            return 0

    def columnCount(self, parent=None):
        return self._results.confusionMatrix(self._classifier).shape[1]

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

        if role == Qt.DisplayRole:
            return str(self._results.confusionMatrix(self._classifier)[index.row()][index.column()])

        return None

    def headerData(self, section, orientation, role):
        """
        Data for header cell.
        
        :param section: Header column. (row for vertical header)
        :type section: PySide2.QtCore.int
        :param orientation: Table orientation.
        :type orientation: PySide2.QtCore.Qt.Orientation
        :param role: Role of section.
        :type role: PySide2.QtCore.int
        :return: Data for indexed header cell.
        :rtype: object
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                labelRealPred = "Predicted: "
            else:
                labelRealPred = "Real: "
            return labelRealPred + self._results.encoder.inverse_transform([section])[
                0]  # we can use section directly because encoder encodes labels from zero

        return None


class TableSummarizationResultsModel(QAbstractTableModel):
    """
    Model for tableview that is for showing summarized statistics of experiment.
    """

    NUM_COL = 16
    """Number of columns in table."""

    COLL_CLASSIFIER_NAME = 0
    """Index of classifier name column."""

    COLL_CLASSIFIER_ABBER_NAME = 1
    """Index of classifier name abbreviation column."""

    COLL_ACCURACY = 2
    """Index of column with average accuracy."""

    COLL_MICRO_AVG_F1_SCORE = 3
    """Index of column with micro avg F1 score."""

    COLL_MICRO_AVG_PRECISION_SCORE = 4
    """Index of column with micro avg precision score."""

    COLL_MICRO_AVG_RECALL_SCORE = 5
    """Index of column with micro avg recall score."""

    COLL_MACRO_AVG_F1_SCORE = 6
    """Index of column with macro avg F1 score."""

    COLL_MACRO_AVG_PRECISION_SCORE = 7
    """Index of column with macro avg precision score."""

    COLL_MACRO_AVG_RECALL_SCORE = 8
    """Index of column with macro avg recall score."""

    COLL_WEIGHTED_AVG_F1_SCORE = 9
    """Index of column with weighted avg F1 score."""

    COLL_WEIGHTED_AVG_PRECISION_SCORE = 10
    """Index of column with weighted avg precision score."""

    COLL_WEIGHTED_AVG_RECALL_SCORE = 11
    """Index of column with weighted avg recall score."""

    COLL_TRAIN_TIME_SCORE = 12
    """Index of column with average train time."""

    COLL_TEST_TIME_SCORE = 13
    """Index of column with average test time."""

    COLL_TRAIN_PROCESS_TIME_SCORE = 14
    """Index of column with average train process time."""

    COLL_TEST_PROCESS_TIME_SCORE = 15
    """Index of column with average test process time."""

    def __init__(self, parent, results: Results):
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
    def results(self, results: Results):
        """
        Assign new results.
        
        :param results: New results that should be now used.
        :type results: Results
        """

        self.beginResetModel()
        self._results = results
        self.endResetModel()

    def rowCount(self, parent=None):
        try:
            return len(self._results.classifiers)
        except (AttributeError, TypeError):
            # probably no assigned data stats
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

        # classifier on current row
        classifier = self._results.classifiers[index.row()]

        if role == Qt.DisplayRole:
            if index.column() == self.COLL_CLASSIFIER_NAME:
                return str(classifier.getName())

            if index.column() == self.COLL_CLASSIFIER_ABBER_NAME:
                return str(classifier.getNameAbbreviation())

            if index.column() == self.COLL_ACCURACY:
                return str(self._results.scores[classifier][Results.ScoreType.ACCURACY])

            if index.column() == self.COLL_MICRO_AVG_F1_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MICRO_F1])

            if index.column() == self.COLL_MICRO_AVG_PRECISION_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MICRO_PRECISION])

            if index.column() == self.COLL_MICRO_AVG_RECALL_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MICRO_RECALL])

            if index.column() == self.COLL_MACRO_AVG_F1_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MACRO_F1])

            if index.column() == self.COLL_MACRO_AVG_PRECISION_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MACRO_PRECISION])

            if index.column() == self.COLL_MACRO_AVG_RECALL_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.MACRO_RECALL])

            if index.column() == self.COLL_WEIGHTED_AVG_F1_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.WEIGHTED_F1])

            if index.column() == self.COLL_WEIGHTED_AVG_PRECISION_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.WEIGHTED_PRECISION])

            if index.column() == self.COLL_WEIGHTED_AVG_RECALL_SCORE:
                return str(self._results.scores[classifier][Results.ScoreType.WEIGHTED_RECALL])

            if index.column() == self.COLL_TRAIN_TIME_SCORE:
                return str(self._results.times[classifier][Validator.TimeDuration.TRAINING])

            if index.column() == self.COLL_TEST_TIME_SCORE:
                return str(self._results.times[classifier][Validator.TimeDuration.TEST])

            if index.column() == self.COLL_TRAIN_PROCESS_TIME_SCORE:
                return str(self._results.times[classifier][Validator.TimeDuration.TRAINING_PROC])

            if index.column() == self.COLL_TEST_PROCESS_TIME_SCORE:
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
            # we have horizontal header only.
            try:
                return self._HEADERS[section]
            except AttributeError:
                """Name of columns in table. Initialization is performed on demand."""
                self._HEADERS = [self.tr("classifier"), self.tr("classifier abbrevation"), self.tr("accuracy"),
                                 self.tr("micro avg F1"), self.tr("micro avg precision"), self.tr("micro avg recall"),
                                 self.tr("macro avg F1"), self.tr("macro avg precision"), self.tr("macro avg recall"),
                                 self.tr("weighted avg F1"), self.tr("weighted avg precision"),
                                 self.tr("weighted avg recall"),
                                 self.tr("avg train time"), self.tr("avg test time"), self.tr("avg train process time"),
                                 self.tr("avg test process time")]
                return self._HEADERS[section]

        return None


class TableAttributesStatsModel(QAbstractTableModel):
    """
    Model for tableview that is for showing statistics of attributes.
    """

    NUM_COL = 3
    """Number of columns in table."""

    COLL_ATTRIBUTE_NAME = 0
    """Index of attribute name column."""

    COLL_NUM_OF_FEATURES = 1
    """Index of column with number of features for attribute."""

    COLL_FEATURES_SD = 2
    """Index of column with features standard deviation."""

    def __init__(self, parent, experiment: Experiment):
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
    def experiment(self, experiment: Experiment):
        """
        Assign new experiment.
        
        :param experiment: New experiment that should be now used.
        :type experiment: Experiment
        """
        self.beginResetModel()
        self._experiment = experiment
        self.endResetModel()

    def rowCount(self, parent=None):
        try:
            return len(self._experiment.dataStats.attributes)
        except (AttributeError, TypeError):
            # probably no assigned data stats
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

        # class name on current row
        attributeName = self._experiment.dataStats.attributes[index.row()]

        if role == Qt.DisplayRole:
            if index.column() == self.COLL_ATTRIBUTE_NAME:
                return str(attributeName)

            if index.column() == self.COLL_NUM_OF_FEATURES:
                return str(self.experiment.dataStats.attributesFeatures[attributeName])

            if index.column() == self.COLL_FEATURES_SD:
                return str(self.experiment.dataStats.attributesAVGFeatureSD[attributeName])

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
            # we have horizontal header only.
            try:
                return self._HEADERS[section]
            except AttributeError:
                """Name of columns in table. Initialization is performed on demand."""
                self._HEADERS = [self.tr("attribute"), self.tr("number of features"),
                                 self.tr("average features standard deviation")]
                return self._HEADERS[section]

        return None


class TableClassStatsModel(QAbstractTableModel):
    """
    Model for tableview that is for showing statistics of classes.

    """

    NUM_COL = 4
    """Number of columns in table."""

    COLL_CLASS_NAME = 0
    """Index of class name column."""

    COLL_SAMPLES = 1
    """Index of column with number of samples in that class."""

    COLL_SAMPLES_ORIG = 2
    """Index of column with original number of samples in that class."""

    COLL_USE = 3
    """Index of use column."""

    def __init__(self, parent, experiment: Experiment):
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
    def experiment(self, experiment: Experiment):
        """
        Assign new experiment.
        
        :param experiment: New experiment that should be now used.
        :type experiment: Experiment
        """

        self.beginResetModel()
        self._experiment = experiment
        self.endResetModel()

    def rowCount(self, parent=None):
        try:
            return len(self._experiment.origDataStats.classes)
        except (AttributeError, TypeError):
            # probably no assigned data stats
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
        f = Qt.ItemIsEnabled | Qt.ItemIsSelectable

        if index.column() in {self.COLL_USE}:
            f |= Qt.ItemIsUserCheckable

        if index.column() in {self.COLL_SAMPLES}:
            f |= Qt.ItemIsEditable

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
        # class name on current row
        className = self._experiment.origDataStats.classes[index.row()]

        if role == Qt.CheckStateRole and index.column() in {self.COLL_USE}:
            # checkbox change

            if value == Qt.Checked:
                self._experiment.dataStats.activateClass(className)
            else:
                self._experiment.dataStats.deactivateClass(className)

            self.dataChanged.emit(index, self.COLL_SAMPLES)

        elif role == Qt.EditRole:
            if index.column() == self.COLL_SAMPLES:
                if self._experiment.dataStats.isActive(className):
                    # new number of samples
                    try:
                        iVal = int(value)
                    except ValueError:
                        pass
                    else:
                        origVal = self._experiment.origDataStats.classSamples[className]

                        setVal = iVal if iVal <= origVal else origVal
                        setVal = 1 if setVal < 1 else setVal
                        self._experiment.dataStats.changeSamplesInClass(className, setVal)

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

        # class name on current row
        className = self._experiment.origDataStats.classes[index.row()]

        if role == Qt.DisplayRole or role == Qt.EditRole:
            if index.column() == self.COLL_CLASS_NAME:
                # class name
                return str(className)

            if index.column() == self.COLL_SAMPLES:
                if self._experiment.dataStats.isActive(className):
                    return str(self._experiment.dataStats.classSamples[className])
                else:
                    return 0

            if index.column() == self.COLL_SAMPLES_ORIG:
                return str(self._experiment.origDataStats.classSamples[className])

        if role == Qt.CheckStateRole:
            if index.column() == self.COLL_USE:
                # use column
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
            # we have horizontal header only.
            try:
                return self._HEADERS[section]
            except AttributeError:
                """Name of columns in table. Initialization is performed on demand."""
                self._HEADERS = [self.tr("class"), self.tr("selected samples"), self.tr("original samples"),
                                 self.tr("use")]
                return self._HEADERS[section]

        return None


class TableDataAttributesModel(QAbstractTableModel):
    """
    Model for tableview that is for showing dataset attributes.

    """

    NUM_COL = 6
    """Number of columns in table."""

    COLL_ATTRIBUTE_NAME = 0
    """Index of attribute name column."""

    COLL_USE = 1
    """Index of use column."""

    COLL_PATH = 2
    """Index of path column."""

    COLL_LABEL = 3
    """Index of label column."""

    COLL_FEATURE_EXTRACTION = 4
    """Index of feature extraction method column."""

    COLL_FEATURE_EXTRACTION_PROPERITES = 5
    """Index of feature extraction method properties column."""

    def __init__(self, parent, experiment: Experiment, showExtractorAttr: Callable[[int], None] = None):
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
        self._showExtractorAttr = showExtractorAttr

    @property
    def experiment(self):
        """
        Assigned experiment.
        """
        return self._experiment

    @experiment.setter
    def experiment(self, experiment: Experiment):
        """
        Assign new experiment.
        
        :param experiment: New experiment that should be now used.
        :type experiment: Experiment
        """

        self.beginResetModel()
        self._experiment = experiment
        self.endResetModel()

    def rowCount(self, parent=None):
        try:
            return len(self._experiment.dataset.attributes)
        except AttributeError:
            # probably no assigned data set
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
        f = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() in {self.COLL_USE, self.COLL_PATH}:
            f |= Qt.ItemIsUserCheckable

        if index.column() == self.COLL_LABEL:
            f |= Qt.ItemIsEditable
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

        # attribute name on current row
        attributeName = self._experiment.dataset.attributes[index.row()]

        if role == Qt.CheckStateRole and index.column() in {self.COLL_USE, self.COLL_PATH}:
            # checkbox change
            changeSetting = Experiment.AttributeSettings.USE if index.column() == self.COLL_USE else Experiment.AttributeSettings.PATH

            if value == Qt.Checked:
                self._experiment.setAttributeSetting(attributeName, changeSetting, True)
            else:
                self._experiment.setAttributeSetting(attributeName, changeSetting, False)

        elif role == Qt.EditRole:
            if index.column() == self.COLL_LABEL:
                # radio button
                self._experiment.setAttributeSetting(attributeName, Experiment.AttributeSettings.LABEL, True)
            elif index.column() == self.COLL_FEATURE_EXTRACTION:
                if self._experiment.getAttributeSetting(attributeName,
                                                        Experiment.AttributeSettings.FEATURE_EXTRACTOR).getName() != value:
                    # we are interested only if there is a change
                    self._experiment.setAttributeSetting(attributeName,
                                                         Experiment.AttributeSettings.FEATURE_EXTRACTOR,
                                                         self._experiment.featuresExt[value]())

                    # Emit attributes click event, because we want to show to user actual feature extractor
                    # attributes.
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

        # attribute name on current row
        attributeName = self._experiment.dataset.attributes[index.row()]

        if role == Qt.DisplayRole or role == Qt.EditRole:
            if index.column() == self.COLL_ATTRIBUTE_NAME:
                # attribute name
                return attributeName

            if index.column() == self.COLL_LABEL:
                # Is on that index selected label?
                return attributeName == self._experiment.label

            if index.column() == self.COLL_FEATURE_EXTRACTION:
                return self._experiment.getAttributeSetting(attributeName,
                                                            Experiment.AttributeSettings.FEATURE_EXTRACTOR).getName()

        if role == Qt.CheckStateRole:
            if index.column() == self.COLL_USE:
                # use column
                return Qt.Checked if self._experiment.getAttributeSetting(attributeName,
                                                                          Experiment.AttributeSettings.USE) else Qt.Unchecked

            if index.column() == self.COLL_PATH:
                # path column
                return Qt.Checked if self._experiment.getAttributeSetting(attributeName,
                                                                          Experiment.AttributeSettings.PATH) else Qt.Unchecked

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
            # we have horizontal header only.
            try:
                return self._HEADERS[section]
            except AttributeError:
                """Name of columns in table. Initialization is performed on demand."""
                self._HEADERS = [self.tr("attributes"), self.tr("use"), self.tr("path"), self.tr("label"),
                                 self.tr("features extraction"), ""]
                return self._HEADERS[section]

        return None
