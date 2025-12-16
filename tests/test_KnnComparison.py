import numpy as np
from ml_algorithm_comparison.evaluation import compareKnn

def testComparisonWorks():
    X_train = np.array([[1,2],[2,3],[3,4],[6,7]])
    y_train = np.array([0,0,1,1])
    X_test = np.array([[1.5,2.5],[5.5,6.5]])
    y_test = np.array([0,1])

    result = compareKnn(X_train, y_train, X_test, y_test, kNeighbors=1)
    assert 0 <= result["customAccuracy"] <= 1
    assert 0 <= result["sklearnAccuracy"] <= 1
