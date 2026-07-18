import numpy as np
from sklearn.datasets import make_moons
from mlp import MLP


def _data(seed=0):
    # hai vầng trăng lồng nhau: KHÔNG tách được bằng đường thẳng -> cần lớp ẩn
    X, y = make_moons(n_samples=400, noise=0.15, random_state=seed)
    return X.astype(float), y.astype(float)


def test_forward_shape():
    X, y = _data()
    m = MLP(in_dim=2, hidden=8, epochs=1)
    p = m.forward(X)
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))


def test_learns_nonlinear():
    X, y = _data(seed=1)
    m = MLP(in_dim=2, hidden=16, lr=0.2, epochs=2000, seed=1).fit(X, y)
    acc = (m.predict(X) == y).mean()
    assert acc > 0.9


def test_loss_decreases():
    X, y = _data(seed=2)
    m = MLP(in_dim=2, hidden=16, epochs=500).fit(X, y)
    assert m.history[-1] < m.history[0]
