import numpy as np
from logistic_regression import sigmoid, LogisticRegression
from utils.data import make_blobs2, train_test_split


def test_sigmoid_range():
    z = np.array([-100.0, 0.0, 100.0])
    s = sigmoid(z)
    assert s[0] < 1e-3 and abs(s[1] - 0.5) < 1e-8 and s[2] > 1 - 1e-3


def test_separates_blobs():
    X, y = make_blobs2(n=300, seed=2)
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    model = LogisticRegression(epochs=800).fit(Xtr, ytr)
    acc = (model.predict(Xte) == yte).mean()
    assert acc > 0.9


def test_loss_decreases():
    X, y = make_blobs2(seed=4)
    model = LogisticRegression().fit(X, y)
    assert model.history[-1] < model.history[0]
