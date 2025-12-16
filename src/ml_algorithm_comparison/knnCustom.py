"""
knnCustom.py

Custom implementation of the k-Nearest Neighbors (KNN) algorithm.
"""

import numpy as np
from collections import Counter


class KNNCustom:
    """
    Custom K-Nearest Neighbors classifier.

    Parameters
    ----------
    kNeighbors : int
        Number of nearest neighbors used for voting.
    """

    def __init__(self, kNeighbors=3):
        self.kNeighbors = kNeighbors

    def fit(self, trainData, trainLabels):
        """
        Fit the model using training data.

        Parameters
        ----------
        trainData : array-like
            Feature vectors of training samples.
        trainLabels : array-like
            Class labels for training samples.
        """
        self.trainData = np.array(trainData)
        self.trainLabels = np.array(trainLabels)

    def _euclideanDistance(self, a, b):
        """
        Compute Euclidean distance between two points.

        Parameters
        ----------
        a : array-like
            First point.
        b : array-like
            Second point.

        Returns
        -------
        float
            Euclidean distance.
        """
        return np.sqrt(np.sum((a - b) ** 2))

    def predict(self, testData):
        """
        Predict class labels for test data.

        Parameters
        ----------
        testData : array-like
            Feature vectors to classify.

        Returns
        -------
        numpy.ndarray
            Predicted class labels.
        """
        predictions = []

        for point in testData:
            distances = [
                self._euclideanDistance(point, trainPoint)
                for trainPoint in self.trainData
            ]
            nearest = np.argsort(distances)[:self.kNeighbors]
            labels = self.trainLabels[nearest]
            predictions.append(Counter(labels).most_common(1)[0][0])

        return np.array(predictions)
