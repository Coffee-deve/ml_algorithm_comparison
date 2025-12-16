import numpy as np

from ml_algorithm_comparison.knnCustom import KNNCustom
from ml_algorithm_comparison.visualization import plotDataPoints
from ml_algorithm_comparison.elbowMethod import plotElbowMethod


def testKnnWithVisualizationAndElbow():
    trainData = np.array([
        [1, 1], [1.5, 1.8], [2, 2], [2.2, 2.4], [2.8, 2.6],
        [3, 3], [3.2, 3.1], [3.5, 3.3], [4, 4], [4.2, 4.1],
        [4.5, 4.2], [5, 5], [5.2, 5.1], [5.5, 5.6], [4.6, 4.7],
        [6, 6], [6.2, 6.3], [6.5, 6.6], [7, 7], [7.2, 7.1]
    ])

    trainLabels = np.array([
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ])

    testData = np.array([
        [2.1, 2.1],
        [2.5, 2.8],
        [4.5, 4.0],
        [4.5, 3.0],
        [4.2, 4.5],
        [6.5, 7.2],
        [4.5, 4.5]
    ])

    testLabels = np.array([0, 0, 1, 0, 1, 1, 1])

    model = KNNCustom(kNeighbors=3)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)

    # ---- Visualization ----
    plotDataPoints(trainData, trainLabels, testData)

    # ---- Elbow method ----
    plotElbowMethod(trainData, trainLabels, testData, testLabels, maxK=15)

    assert len(predictions) == len(testData)
