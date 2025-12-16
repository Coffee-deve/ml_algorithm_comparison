import numpy as np
import pytest
from ml_algorithm_comparison.knnCustom import KNNCustom

class TestKNNCustomDistance:
  def test_euclidean_distance(self):
    knn = KNNCustom(distanceMetric="euclidean")
    a = np.array([0, 0])
    b = np.array([3, 4])
    assert knn._distance(a, b) == 5.0

  def test_euclidean_distance_same_points(self):
    knn = KNNCustom(distanceMetric="euclidean")
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    assert knn._distance(a, b) == 0.0

  def test_manhattan_distance(self):
    knn = KNNCustom(distanceMetric="manhattan")
    a = np.array([0, 0])
    b = np.array([3, 4])
    assert knn._distance(a, b) == 7

  def test_manhattan_distance_same_points(self):
    knn = KNNCustom(distanceMetric="manhattan")
    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3])
    assert knn._distance(a, b) == 0

  def test_minkowski_distance_p2(self):
    knn = KNNCustom(distanceMetric="minkowski", minkowskiP=2)
    a = np.array([0, 0])
    b = np.array([3, 4])
    assert np.isclose(knn._distance(a, b), 5.0)

  def test_minkowski_distance_p1(self):
    knn = KNNCustom(distanceMetric="minkowski", minkowskiP=1)
    a = np.array([0, 0])
    b = np.array([3, 4])
    assert np.isclose(knn._distance(a, b), 7.0)

  def test_invalid_distance_metric(self):
    knn = KNNCustom(distanceMetric="invalid")
    a = np.array([0, 0])
    b = np.array([1, 1])
    with pytest.raises(ValueError, match="Invalid distanceMetric"):
      knn._distance(a, b)
      
class TestKNNCustomPrediction:
  def test_prediction(self):
    trainData = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
    trainLabels = np.array([0, 0, 1, 1])
    testData = np.array([[1.5, 2.5], [5.5, 6.5]])

    knn = KNNCustom(kNeighbors=1)
    knn.fit(trainData, trainLabels)
    predictions = knn.predict(testData)

    assert np.array_equal(predictions, np.array([0, 1]))

  def test_prediction_with_ties(self):
    trainData = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    trainLabels = np.array([0, 0, 1, 1])
    testData = np.array([[1.5, 1.5]])

    knn = KNNCustom(kNeighbors=4)
    knn.fit(trainData, trainLabels)
    predictions = knn.predict(testData)

    # In case of a tie, the first encountered class is chosen
    assert predictions[0] in [0, 1]