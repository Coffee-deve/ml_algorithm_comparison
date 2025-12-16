"""
knnSklearn.py

This module provides a wrapper around scikit-learn's KNeighborsClassifier.
It is used to compare the custom KNN implementation with a standard library
implementation.
"""
from sklearn.neighbors import KNeighborsClassifier

"""
 Wrapper class for scikit-learn's KNeighborsClassifier.
 """

class KNNSklearn:
    def __init__(self, kNeighbors=3):
        self.model = KNeighborsClassifier(n_neighbors=kNeighbors)

    def fit(self, trainData, trainLabels):
        self.model.fit(trainData, trainLabels)
    """
    Fit the scikit-learn KNN model using training data.

    Parameters
    ----------
    trainData : array-like
        Training feature vectors.
    trainLabels : array-like
        Training class labels.
    """
    def predict(self, testData):
        return self.model.predict(testData)
    """
    Predict class labels using the trained scikit-learn model.

    Parameters
    ----------
    testData : array-like
        Feature vectors for test samples.

    Returns
    -------
    numpy.ndarray
        Predicted class labels.
    """