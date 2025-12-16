"""
visualization.py

This module provides visualization utilities for plotting training data,
test data, and illustrating the behavior of the KNN algorithm.
"""

import matplotlib.pyplot as plt
import numpy as np

colors = {0: "blue", 1: "deeppink",2: "green", 3: "purple"}
markers = {0: "o", 1: "*",2: "^", 3: "s"}

def plotDataPoints(trainData, trainLabels, testData=None, testLabels=None):
    """
    Plot training data points and optional test points.

    Parameters
    ----------
    trainData : array-like
        Training feature vectors.
    trainLabels : array-like
        Class labels for training data.
    testData : array-like, optional
        Test points to visualize.
    """
    trainData = np.array(trainData)
    trainLabels = np.array(trainLabels)


    for label in np.unique(trainLabels):
        points = trainData[trainLabels == label]
        plt.scatter(
            points[:, 0],
            points[:, 1],
            color=colors[label],
            marker=markers[label],
            label=f"Train class {label}",
        )

    if testData is not None:
        testData = np.array(testData)
        plt.scatter(
            testData[:, 0],
            testData[:, 1],
            color="black",
            marker="X",
            s=100,
            label="Test points"
        )

    plt.title("K-NN Data Points Visualization")
    plt.legend()
    plt.grid(True)
    plt.show()
