"""
evaluation.py

This module compares the performance of the custom KNN implementation with
scikit-learn's KNN classifier using accuracy as the evaluation metric.
"""

from sklearn.metrics import accuracy_score
from .knnCustom import KNNCustom
from .knnSklearn import KNNSklearn

def compareKnn(trainData, trainLabels, testData, testLabels, kNeighbors=3):
    """
    Compare custom KNN and scikit-learn KNN models.

    Parameters
    ----------
    trainData : array-like
        Training feature vectors.
    trainLabels : array-like
        Training class labels.
    testData : array-like
        Test feature vectors.
    testLabels : array-like
        True labels for test samples.
    kNeighbors : int
        Number of neighbors used in KNN.

    Returns
    -------
    dict
        Dictionary containing accuracy scores for both models.
    """
    custom = KNNCustom(kNeighbors)
    sklearn = KNNSklearn(kNeighbors)

    custom.fit(trainData, trainLabels)
    sklearn.fit(trainData, trainLabels)

    return {
        "customAccuracy": accuracy_score(testLabels, custom.predict(testData)),
        "sklearnAccuracy": accuracy_score(testLabels, sklearn.predict(testData))
    }
