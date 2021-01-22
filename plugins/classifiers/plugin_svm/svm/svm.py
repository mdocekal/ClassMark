"""
Created on 4. 3. 2019
SVM classifier plugin for ClassMark.

:author:     Martin Dočekal
:contact:    xdocek09@stud.fit.vubtr.cz
"""
from functools import partial

from classmark.core.plugins import Classifier, PluginAttribute, PluginAttributeIntChecker
from classmark.core.preprocessing import BaseNormalizer, NormalizerPlugin, \
    MinMaxScalerPlugin, StandardScalerPlugin, RobustScalerPlugin
from sklearn.svm import LinearSVC, SVC
from typing import List, Tuple
import numpy as np


class SVM(Classifier):
    """
    SVM classifier plugin for ClassMark.
    """

    def __init__(self, normalizer: BaseNormalizer = None,
                 kernel: str = "linear", showImpFeatures: int = 0):
        """
        Classifier initialization.
        
        :param normalizer: Normalizer used for input data. If none than normalization/scalling is omitted.
        :type normalizer: None | BaseNormalizer
        :param kernel: Kernel type that should be used.
        :type kernel: str
        :param showImpFeatures: Sow top-k important features for predictions.
            0 deactivates this.
        :type showImpFeatures: int
        """

        normalizer = NormalizerPlugin()

        self._normalizer = PluginAttribute("Normalize", PluginAttribute.PluginAttributeType.SELECTABLE_PLUGIN, None,
                                           [None, NormalizerPlugin, MinMaxScalerPlugin, StandardScalerPlugin,
                                            RobustScalerPlugin])
        self._normalizer.value = normalizer

        self._kernel = PluginAttribute("Kernel", PluginAttribute.PluginAttributeType.SELECTABLE, str,
                                       ["linear", "poly", "rbf", "sigmoid"])
        self._kernel.value = kernel

        self._showImpFeatures = PluginAttribute("Top-k features (linear only)",
                                                PluginAttribute.PluginAttributeType.VALUE,
                                                PluginAttributeIntChecker(minV=0))

        # let's make sure that top-k features will be used only with linear kernel

        self._showImpFeatures.value = showImpFeatures

    @staticmethod
    def getName():
        return "Support Vector Machines"

    @staticmethod
    def getNameAbbreviation():
        return "SVM"

    @staticmethod
    def getInfo():
        return ""

    def train(self, data, labels):
        if self._normalizer.value is not None:
            data = self._normalizer.value.fitTransform(data)

        # The documentation says:
        #    Prefer dual=False when n_samples > n_features.
        if self._kernel.value == "linear":
            # this should be faster
            self._cls = LinearSVC(dual=data.shape[0] <= data.shape[1])
        else:
            self._cls = SVC(kernel=self._kernel.value)

        self._cls.fit(data, labels)

    def classify(self, data):
        if self._normalizer.value is not None:
            data = self._normalizer.value.transform(data)
        return self._cls.predict(data)

    def classifyShowTopFeatures(self, data, featuresNames: np.array):
        """
        Classify label on provided data and provides top important features that were used for decision.

        Number of most important features determines each classifier itself.
        Consider adding a user editable attribute for it.

        :param data: Data for classification.
        :type data: scipy.sparse matrix
        :param featuresNames: Name for each feature in an input data vector that was passed to the model during training.
        :type featuresNames: np.array
        :return: Predicted labels, array of features names with array of importance scores. Both arrays
        (names, importance) are in descending order according to importance
        :rtype: Tuple[ArrayLike, ArrayLike, ArrayLike]
        """
        predictions = self.classify(data)
        predFeaturesNames = []
        predFeaturesImportance = []

        usedClasses = self._cls.classes_.tolist()
        if len(usedClasses) == 2:
            # binnary classifiers must be handled differently

            # The decision function is for SVM:
            #   sgn(w^t * x + b)
            # Let's get importance score for each sample and feature
            impScores = data.multiply(self._cls.coef_.ravel()).tocsr()

            for p, imp in zip(predictions, impScores):
                imp = imp.toarray().ravel()

                if p != usedClasses[1]:
                    # negative prediction sgn(w^t * x + b) = -1
                    # because importance should be increasing as the feature is more important we should flip
                    # signs
                    imp = -imp

                impSortedIndices = np.argsort(imp)[::-1][:self._showImpFeatures.value]

                predFeaturesNames.append(featuresNames[impSortedIndices])
                predFeaturesImportance.append(imp[impSortedIndices])

        else:
            # we have for each class pair a classifier
            impScores = [data.multiply(classCoef).tocsr() for classCoef in self._cls.coef_]

            # Predicted class p is not direct index even though labels are 0 .. n_classes -1 and classes_ are sorted
            # with np.unique, because in some iteration step there may be a missing class so we need mapping.

            class2Index = {c: i for i, c in enumerate(self._cls.classes_)}

            for i, p in enumerate(predictions):
                imp = impScores[class2Index[p]][i].toarray().ravel()
                impSortedIndices = np.argsort(imp)[::-1][:self._showImpFeatures.value]
                predFeaturesNames.append(featuresNames[impSortedIndices])
                predFeaturesImportance.append(imp[impSortedIndices])

        return predictions, np.array(predFeaturesNames), np.array(predFeaturesImportance)

    def featureImportanceShouldBeShown(self) -> bool:
        return self._kernel.value == "linear" and self._showImpFeatures.value > 0
