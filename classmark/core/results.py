"""
Created on 10. 3. 2019
Module for experiment results.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""

from .plugins import Classifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from typing import List, Any, Optional
from enum import Enum, auto
import numpy as np
import copy


class Results(object):
    """
    Storage of experiment results.
    """

    class ScoreType(Enum):
        """
        Average scores.
        """

        ACCURACY = auto()
        """Average accuracy among all steps."""

        MACRO_PRECISION = auto()
        """Average macro precision among all steps."""

        MACRO_RECALL = auto()
        """Average macro recall among all steps."""

        MACRO_F1 = auto()
        """Average macro F1 among all steps."""

        MICRO_PRECISION = auto()
        """Average micro precision among all steps."""

        MICRO_RECALL = auto()
        """Average micro recall among all steps."""

        MICRO_F1 = auto()
        """Average micro F1 among all steps."""

        WEIGHTED_PRECISION = auto()
        """Average weighted precision among all steps."""

        WEIGHTED_RECALL = auto()
        """Average weighted recall among all steps."""

        WEIGHTED_F1 = auto()
        """Average weighted F1 among all steps."""

    class ValidationStep(object):
        """
        Results for 
        """

        def __init__(self):
            self._predictions = {}
            self._importantFeatures = {}
            self._importanceOfFeatures = {}
            self._testIndices = {}
            self._times = {}
            self._stats = {}
            self.labels = None  # True labels for test data in that validation step.

        @property
        def numOfPredictedLabels(self):
            """
            Num of predicted labels.
            """
            try:
                return len(next(iter(self._predictions.values())))
            except:
                return 0

        def addResults(self, classifier: Classifier, predictions: np.array, importantFeatures: Optional[np.array],
                       importanceOfFeatures: Optional[np.array], testIndices: np.array, times, stats):
            """
            Add experiment results for classifier.

            :param classifier: The classifier.
            :type classifier: Classifier
            :param predictions: Classifier's predictions of labels.
            :type predictions:np.array
            :param importantFeatures: A classifier may provide top important features (their names) that were most
                important for a prediction.
                Ordered from most important to least important.
            :type importantFeatures: Optional[np.array]
            :param importanceOfFeatures: Importance of a feature for given prediction.
                Ordered from most important to least important.
            :type importanceOfFeatures: Optional[np.array]
            :param testIndices: Indices of data that were use for prediction.
            :type testIndices: np.array
            :param times: dict of times
            :type times: Dict[Validator.TimeDuration,float]
            :param stats: samples stats
            :type stats: Dict[Validator.SamplesStats,float]
            """
            self._predictions[classifier.stub()] = predictions
            self._importantFeatures[classifier.stub()] = importantFeatures
            self._importanceOfFeatures[classifier.stub()] = importanceOfFeatures
            self._testIndices[classifier.stub()] = testIndices
            self._times[classifier.stub()] = times
            self._stats[classifier.stub()] = stats

        def testIndicesForCls(self, classifier: Classifier):
            """
            Get test indices for classifier.
            
            :param classifier: The classifier
            :type classifier: Classifier
            :return: Indices of test samples in use dataset.
            :rtype: np.array
            """

            return self._testIndices[classifier.stub()]

        def predictionsForCls(self, classifier: Classifier):
            """
            Get predictions for a classifier.
            
            :param classifier: The classifier
            :type classifier: Classifier
            :return: Predicted labels for this step.
            :rtype: np.array
            """

            return self._predictions[classifier.stub()]

        def importantFeaturesForCls(self, classifier: Classifier):
            """
            Get important features for a classifier.

            :param classifier: The classifier
            :type classifier: Classifier
            :return: Top important features for each prediction.
                Ordered from most important to least important.
            :rtype: np.array
            """

            return self._importantFeatures[classifier.stub()]

        def importanceOfFeaturesForCls(self, classifier: Classifier):
            """
            Get importance of important features for a classifier.

            :param classifier: The classifier
            :type classifier: Classifier
            :return: Importance of top features for each prediction.
                Ordered from most important to least important.
            :rtype: np.array
            """

            return self._importanceOfFeatures[classifier.stub()]

        def timesForCls(self, classifier: Classifier):
            """
            Get times for classifier.
            
            :param classifier: The classifier
            :type classifier: Classifier
            :return: Times for classifier.
            :rtype: Dict[Validator.TimeDuration,float]
            """

            return self._times[classifier.stub()]

        def statsForCls(self, classifier: Classifier):
            """
            Get stats like number of samples and features.
            
            :param classifier: The classifier.
            :type classifier: Classifier
            :return: samples stats
            :rtype: Dict[Validator.SamplesStats,float]
            """

            return self._stats[classifier.stub()]

        def confusionMatrix(self, c: Classifier):
            """
            Get confusion matrix for given classifier.

            :param c: The classifier
            :type c: Classifier
            :return: Confusion matrix of given classifier.
            :rtype: np.array
            """
            return confusion_matrix(self.labels, self._predictions[c])

        def score(self):
            """
            Calculates score for this step.
            
            :return: Score.
                Dict key is classifier stub.
            :rtype: Dict[PluginStub,Dict[ScoreType,float]]
            """

            res = {}
            for c, predictions in self._predictions.items():
                macroPrecision, macroRecall, macroF1score, _ = precision_recall_fscore_support(self.labels, predictions,
                                                                                               average='macro')
                microPrecision, microRecall, microF1score, _ = precision_recall_fscore_support(self.labels, predictions,
                                                                                               average='micro')
                weightedPrecision, weightedRecall, weightedF1score, _ = precision_recall_fscore_support(self.labels,
                                                                                                        predictions,
                                                                                                        average='weighted')
                res[c] = {
                    Results.ScoreType.ACCURACY: accuracy_score(self.labels, predictions),
                    Results.ScoreType.MACRO_PRECISION: macroPrecision,
                    Results.ScoreType.MACRO_RECALL: macroRecall,
                    Results.ScoreType.MACRO_F1: macroF1score,
                    Results.ScoreType.MICRO_PRECISION: microPrecision,
                    Results.ScoreType.MICRO_RECALL: microRecall,
                    Results.ScoreType.MICRO_F1: microF1score,
                    Results.ScoreType.WEIGHTED_PRECISION: weightedPrecision,
                    Results.ScoreType.WEIGHTED_RECALL: weightedRecall,
                    Results.ScoreType.WEIGHTED_F1: weightedF1score
                }

            return res

    def __init__(self, numOfSteps: int, classifiers: List[Classifier], labelEncoder: LabelEncoder):
        """
        Initialization of results.
        
        :param numOfSteps: Total number of steps to complete validation process for one classifier.
        :type numOfSteps: int
        :param classifiers: Classifiers that will be tested. This parameter is mainly used to determinate classifier order.
        :type classifiers: List[Classifier]
        :param labelcEncoder: We assume that providet labels are in encoded form. This encoder
            serves for getting real labels.
        :type labelcEncoder:LabelEncoder
        """

        self.steps = [self.ValidationStep() for _ in range(numOfSteps)]
        self._classifiers = [c.stub() for c in classifiers]
        self.encoder = labelEncoder
        self._finalize = False
        self._finalScore = None
        self._finalConfMat = {}
        self._finalTimes = None
        self._log = ""  # whole log for experiment

    @property
    def log(self):
        """
        whole log for experiment
        """
        return self._log

    @log.setter
    def log(self, l: str):
        """
        Setter for experiment log.
        
        :param l: The new log.
        :type l: str
        """
        self._log = l

    def finalize(self):
        """
        Mark this result as complete.
        When you mark the result than repeated calling of score (and other metrics) method will be faster, because
        the score will be computed only for the first time and next time the saved one will be used.
        """
        self._finalize = True

    @property
    def classifiers(self):
        """
        Tested classifiers plugin stubs.
        """
        return self._classifiers

    @property
    def times(self):
        """
        Average times for all classifiers among all steps.
        
        :return: Average times.
        :rtype: Dict[PluginStub,Dict[Validator.TimeDuration,float]] | None
        """

        if len(self.steps) == 0:
            return None

        if self._finalTimes is not None:
            # use cached
            return self._finalTimes

        stepIter = iter(self.steps)
        try:
            # get the sum for avg
            res = copy.copy(next(stepIter)._times)
            while True:
                for c, cVals in next(stepIter)._times.items():
                    for t, v in cVals.items():
                        res[c][t] += v
        except StopIteration:
            # ok we have all steps
            # let's calc the avg

            for c, cVals in res.items():
                for t, v in cVals.items():
                    res[c][t] = v / len(self.steps)

        if self._finalize:
            # these are final results so we can cache the avg times
            self._finalTimes = res

        return res

    def confusionMatrix(self, c: Classifier):
        """
        Get confusion matrix for given classifier.
        
        The matrix is sum of all confusion matrices in all cross validation steps.
        
        :param c: The classifier
        :type c: Classifier
        :return: Confusion matrix of given classifier.
        :rtype: np.array
        """
        if len(self.steps) == 0:
            return None

        if c in self._finalConfMat:
            # use cached
            return self._finalConfMat[c]

        # If we have two few samples
        # there could be steps without some labels.
        # So we must have on mind this.

        resConfMat = np.zeros([len(self.encoder.classes_), len(self.encoder.classes_)], dtype=int)

        for step in self.steps:
            actMat = step.confusionMatrix(c)

            if actMat.shape[0] != resConfMat.shape[0]:
                # a label is missing
                usL = np.unique(
                    np.concatenate([step.labels, step.predictionsForCls(c)]))  # returns unique sorted labels
                # our labels are just number that are zero or greater

                for row in range(actMat.shape[0]):
                    for col in range(actMat.shape[1]):
                        # mapping example:
                        #    all labels: 0 1 2 3 4
                        #    real labels in act step: 1 3 4
                        #    predicted labels in act step:  1 2 4
                        #    all unique labels in act step:  1 2 3 4
                        #
                        #    row=0 -> 1
                        #    col=1 -> 2
                        resConfMat[usL[row]][usL[col]] += actMat[row][col]
            else:
                # all labels
                resConfMat += actMat

        if self._finalize:
            self._finalConfMat[c] = resConfMat

        return resConfMat

    @property
    def scores(self):
        """
        Average scores for all classifiers among all steps.
        
        :return: Score.
            Dict key is classifier plugin stub.
        :rtype: Dict[PluginStub,Dict[ScoreType,float]] | None
        """
        if len(self.steps) == 0:
            return None

        if self._finalScore is not None:
            # use cached
            return self._finalScore

        res = {}
        # let's calc sum, it will be used for avg calculation, over all scores
        for s in self.steps:
            for c, scores in s.score().items():
                for scoreType, scoreVal in scores.items():
                    try:
                        res[c][scoreType] += scoreVal  # calc sum
                    except KeyError:
                        if c not in res:
                            res[c] = {}
                        res[c][scoreType] = scoreVal

        # calc avg from sum and number of steps
        for scores in res.values():
            for scoreType, scoreVal in scores.items():
                scores[scoreType] = scoreVal / len(self.steps)

        if self._finalize:
            # these are final results so we can cache the scores
            self._finalScore = res
        return res
