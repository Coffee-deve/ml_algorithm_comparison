import numpy as np
from collections import Counter


class KNNCustom:
    def __init__(self, kNeighbors=3, distanceMetric="euclidean", minkowskiP=2):
        """
        Custom K-Nearest Neighbors classifier.

        Parameters
        ----------
        kNeighbors : int
            Number of nearest neighbors.
        distanceMetric : str
            Distance metric to use: 'euclidean', 'manhattan', or 'minkowski'.
        minkowskiP : int
            Power parameter for Minkowski distance.
        """
        self.kNeighbors = kNeighbors
        self.distanceMetric = distanceMetric
        self.minkowskiP = minkowskiP

    def fit(self, trainData, trainLabels):
        self.trainData = np.array(trainData)
        self.trainLabels = np.array(trainLabels)

    def _distance(self, a, b):
        if self.distanceMetric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2))

        elif self.distanceMetric == "manhattan":
            return np.sum(np.abs(a - b))

        elif self.distanceMetric == "minkowski":
            return np.sum(np.abs(a - b) ** self.minkowskiP) ** (1 / self.minkowskiP)

        else:
            raise ValueError(
                "Invalid distanceMetric. Use 'euclidean', 'manhattan', or 'minkowski'."
            )

    def predict(self, testData):
        predictions = []

        for point in testData:
            distances = [
                self._distance(point, trainPoint)
                for trainPoint in self.trainData
            ]

            nearestIndices = np.argsort(distances)[:self.kNeighbors]
            nearestLabels = self.trainLabels[nearestIndices]

            predictions.append(
                Counter(nearestLabels).most_common(1)[0][0]
            )

        return np.array(predictions)
