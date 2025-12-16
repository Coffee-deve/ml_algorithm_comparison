knn-algorithm-project

A from-scratch implementation of the k-Nearest Neighbors (K-NN) algorithm with support for multiple distance metrics, visualization, and comparison against the reference implementation available in scikit-learn.

This project was developed as part of the Programming in Python Language course and demonstrates a complete machine-learning workflow including algorithm implementation, evaluation, testing, and documentation.

Project Features

The project includes:

A custom K-NN implementation written from scratch (no sklearn logic reused)

Support for multiple distance metrics:

Euclidean distance

Manhattan distance

Minkowski distance

Multi-class classification (four classes: 0, 1, 2, 3)

Direct comparison with scikit-learn’s KNeighborsClassifier

Elbow Method visualization for selecting the optimal number of neighbors

2D data visualization of training and test points

Unit tests validating correctness and stability

Fully documented code using Python docstrings

Project Structure

The project follows the src/ layout recommended by PyScaffold:

ml_algorithm_comparison/
├── src/ml_algorithm_comparison/
│   ├── __init__.py
│   ├── knnCustom.py          # Custom K-NN implementation
│   ├── knnSklearn.py         # scikit-learn wrapper
│   ├── evaluation.py         # Accuracy comparison logic
│   ├── visualization.py     # Data and result plots
│   └── elbowMethod.py        # Elbow method implementation
│
├── tests/
│   ├── testKnnComparison.py
│   └── testKnnUnit.py
│
├── docs/
│   ├── knnTheory.rst
│   ├── implementation.rst
│   └── comparison.rst
│
├── demoknn.py                # Runnable demo script
├── README.rst
├── CHANGELOG.rst
└── pyproject.toml

Algorithm Overview

The k-Nearest Neighbors algorithm is a non-parametric, instance-based learning method.
For a given test point:

The distance to all training points is computed

The k closest neighbors are selected

The predicted class is determined by majority voting

This project allows experimenting with different distance metrics and values of k, and visually inspecting their influence on classification performance.

Usage Example

Basic usage of the custom K-NN model:

import numpy as np
from ml_algorithm_comparison.knnCustom import KNNCustom

trainData = np.array([[1, 1], [2, 2], [6, 6]])
trainLabels = np.array([0, 0, 1])

model = KNNCustom(kNeighbors=3, distanceMetric="euclidean")
model.fit(trainData, trainLabels)

predictions = model.predict([[1.5, 1.7]])

Comparison with scikit-learn

The project includes a direct comparison between:

Custom K-NN implementation

sklearn.neighbors.KNeighborsClassifier

Comparison metrics include:

Classification accuracy

Prediction consistency

Behavior for different k values

Results are generated using identical datasets to ensure fairness.

Visualization

The project provides visual outputs to help understand K-NN behavior:

Scatter plots of training and test points

Class-colored data visualization

Elbow Method plot showing error rate vs. number of neighbors

These plots help explain:

how class boundaries form

how the choice of k affects model performance

Testing

The tests/ directory contains unit tests and integration tests that:

Validate distance calculations

Ensure predictions are valid

Verify comparison results are within expected bounds

Tests can be run using:

pytest


or with tox:

tox

Documentation

The docs/ directory contains:

Algorithm theory explanation

Detailed code walkthrough

Comparison analysis with scikit-learn

All source files include proper Python docstrings suitable for automatic documentation generation.

Summary

This project demonstrates:

A correct and transparent implementation of the K-NN algorithm

Proper software engineering practices

Clear separation between algorithm, evaluation, visualization, and testing

Compliance with academic requirements for documentation and reproducibility
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
