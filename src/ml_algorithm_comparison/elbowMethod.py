"""
elbowMethod.py

This module implements the Elbow Method for selecting the optimal number
of neighbors (k) in the KNN algorithm by analyzing error rates.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from .knnCustom import KNNCustom

def plotElbowMethod(trainData, trainLabels, testData, testLabels, maxK=10):
    """
    Plot error rate versus number of neighbors (k).

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
    maxK : int
        Maximum number of neighbors to test.
    """
    kValues = range(1, maxK + 1)
    errorRates = []

    for k in kValues:
        model = KNNCustom(kNeighbors=k)
        model.fit(trainData, trainLabels)

        predictions = model.predict(testData)
        accuracy = accuracy_score(testLabels, predictions)
        errorRate = 1 - accuracy

        errorRates.append(errorRate)

    # Plot
    plt.figure()
    plt.plot(kValues, errorRates, marker="o")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Error Rate")
    plt.title("Elbow Method for K-NN")
    plt.grid(True)
    plt.show()
