import numpy as np
from linear_regression import LinearRegression, mse
from utils.data import make_linear, train_test_split


def test_recovers_true_params():
    # dữ liệu quanh y = 2x + 1, model phải học ra gần đúng
    X, y = make_linear(n=300, slope=2.0, bias=1.0, noise=0.1, seed=3)
    model = LinearRegression(lr=0.1, epochs=500).fit(X, y)
    assert abs(model.w[0] - 2.0) < 0.15
    assert abs(model.b - 1.0) < 0.15


def test_loss_decreases():
    X, y = make_linear(seed=5)
    model = LinearRegression().fit(X, y)
    assert model.history[-1] < model.history[0]


def test_generalizes():
    X, y = make_linear(n=200, seed=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y)
    model = LinearRegression(epochs=400).fit(Xtr, ytr)
    assert mse(yte, model.predict(Xte)) < 0.5
