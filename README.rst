.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/knn-project.svg?branch=main
        :alt: Build Status
        :target: https://cirrus-ci.com/github/<USER>/knn-project
    .. image:: https://readthedocs.org/projects/knn-project/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://knn-project.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/pypi/v/knn-project.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/knn-project/

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

============
knn-project
============

A **from-scratch implementation of the k-Nearest Neighbors (K-NN) algorithm**
developed for educational purposes and comparison with the implementation
available in **scikit-learn**.

This project was developed as part of the *Programming in Python Language*
course at **AGH University of Krakow**.

It includes:

- a custom implementation of the K-NN classification algorithm
- support for multiple distance metrics (Euclidean, Manhattan, Minkowski)
- comparison with ``sklearn.neighbors.KNeighborsClassifier``
- visualization of data points and decision behavior
- Elbow Method visualization for selecting the optimal number of neighbors ``k``
- automated unit tests validating correctness of the implementation

Project Structure
=================

The project follows the ``src/`` layout recommended by PyScaffold:

::

    knn-project/
    ├── src/ml_algorithm_comparison/
    │   ├── __init__.py
    │   ├── knnCustom.py
    │   ├── knnSklearn.py
    │   ├── evaluation.py
    │   ├── visualization.py
    │   └── elbowMethod.py
    ├── tests/
    │   └── testKnnComparison.py
    ├── docs/
    ├── CHANGELOG.rst
    ├── README.rst
    └── pyproject.toml

Algorithm Description
=====================

The **k-Nearest Neighbors (K-NN)** algorithm is a non-parametric, instance-based
learning method used for classification.

For each test point:

1. The distance to all training points is computed
2. The ``k`` nearest neighbors are selected
3. The predicted class is chosen by majority voting

The project supports multiple distance metrics:

- **Euclidean distance**
- **Manhattan distance**
- **Minkowski distance**

Comparison with scikit-learn
============================

The project provides a direct comparison between:

- the custom ``KNNCustom`` implementation
- ``sklearn.neighbors.KNeighborsClassifier``

Both implementations are evaluated on the same datasets using identical
parameters. Accuracy scores are reported and compared to validate correctness.

Usage Example
=============

Basic usage of the custom K-NN model::

    import numpy as np
    from ml_algorithm_comparison.knnCustom import KNNCustom

    trainData = np.array([[1, 1], [2, 2], [3, 3], [6, 6]])
    trainLabels = np.array([0, 0, 1, 1])

    model = KNNCustom(kNeighbors=3)
    model.fit(trainData, trainLabels)

    predictions = model.predict([[2.5, 2.5]])

Visualization
=============

The project includes visualization utilities that allow:

- plotting training and test data points in 2D space
- visual inspection of class distributions
- graphical comparison of clustering behavior

Additionally, the **Elbow Method** is implemented to help select an optimal
value for ``k`` by plotting the error rate as a function of ``k``.

Testing
=======

Automated tests are provided using **pytest**.  
Tests verify that:

- the custom K-NN implementation runs correctly
- predictions have valid output
- accuracy values are within valid ranges
- the comparison with scikit-learn executes successfully

Tests can be run using::

    pytest

Note
====

This project was generated using **PyScaffold** and follows recommended best
practices for Python project structure, testing, and documentation.

For more information about PyScaffold see https://pyscaffold.org/.

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
