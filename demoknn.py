"""
demoknn.py

Demonstration script for the KNN project.
This script visualizes the dataset, runs the KNN algorithm,
and displays results such as predictions and the elbow method plot.
"""

import numpy as np

from ml_algorithm_comparison.knnCustom import KNNCustom
from ml_algorithm_comparison.visualization import plotDataPoints
from ml_algorithm_comparison.elbowMethod import plotElbowMethod

np.random.seed(67)

class0 = np.random.uniform(low=0, high=3.3, size=(20, 2))
class1 = np.random.uniform(low=3, high=6, size=(20, 2))
class2 = np.random.uniform(low=5.5, high=8, size=(20, 2))

class3_x = np.random.uniform(low=6, high=8, size=(20, 1))
class3_y = np.random.uniform(low=1, high=3, size=(20, 1))
class3 = np.hstack((class3_x, class3_y))

trainData = np.vstack((class0, class1, class2, class3))

trainLabels = np.array(
    [0] * 20 +
    [1] * 20 +
    [2] * 20 +
    [3] * 20
)

testData = np.array([
    [0.3,2.1],
    [3.0,3.0],
    [3.0, 4.1],
    [4.6, 4.4],
    [7.8, 7.4],
    [5.5, 7.5],
    [7.6, 1.6]

])

testLabels = np.array([0,0, 1, 1, 2, 2, 3])

model = KNNCustom(
    kNeighbors = 3,
    distanceMetric = "euclidean"  # "euclidean " or "manhattan" or "minkowski"
)

model.fit(trainData, trainLabels)
predictions = model.predict(testData)

print("Predictions:", predictions)

# ===== VISUALIZATION =====
plotDataPoints(trainData, trainLabels, testData)

# ===== ELBOW METHOD =====
plotElbowMethod(
    trainData,
    trainLabels,
    testData,
    testLabels,
    maxK=10
)