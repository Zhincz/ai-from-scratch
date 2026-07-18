import numpy as np
import pytest
from warmup import normalize, euclidean, onehot, softmax, accuracy


def test_normalize():
    x = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    out = normalize(x)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(out.std(axis=0), 1, atol=1e-8)


def test_euclidean():
    assert euclidean(np.array([0, 0]), np.array([3, 4])) == pytest.approx(5.0)


def test_onehot():
    out = onehot(np.array([0, 2, 1]), 3)
    assert np.array_equal(out, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))


def test_softmax():
    out = softmax(np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out[0, 2] > out[0, 0]


def test_accuracy():
    y = np.array([1, 0, 1, 1])
    p = np.array([1, 0, 0, 1])
    assert accuracy(y, p) == pytest.approx(0.75)
