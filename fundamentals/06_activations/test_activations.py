import numpy as np
import pytest
import activations as A


def numeric_grad(fn, x, eps=1e-6):
    """Đạo hàm số để đối chiếu với đạo hàm mình tự viết."""
    return (fn(x + eps) - fn(x - eps)) / (2 * eps)


# ---- sigmoid (mẫu, phải pass sẵn) ----
def test_sigmoid_values():
    assert A.sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_grad_matches_numeric():
    x = np.array([-2.0, -0.5, 0.3, 1.7])
    assert np.allclose(A.sigmoid_grad(x), numeric_grad(A.sigmoid, x), atol=1e-5)


# ---- tanh ----
def test_tanh():
    x = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(A.tanh(x), np.tanh(x))
    assert np.allclose(A.tanh_grad(x), numeric_grad(A.tanh, x), atol=1e-5)


# ---- relu ----
def test_relu():
    x = np.array([-3.0, -0.1, 0.0, 0.1, 3.0])
    assert np.allclose(A.relu(x), [0, 0, 0, 0.1, 3.0])
    # tránh điểm gãy x=0 khi so với numeric grad
    xg = np.array([-2.0, 1.5])
    assert np.allclose(A.relu_grad(xg), numeric_grad(A.relu, xg), atol=1e-5)


# ---- leaky relu ----
def test_leaky_relu():
    x = np.array([-2.0, 3.0])
    assert np.allclose(A.leaky_relu(x), [-0.02, 3.0])
    assert np.allclose(A.leaky_relu_grad(x), [0.01, 1.0])


# ---- softmax ----
def test_softmax():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    out = A.softmax(x)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[1], [1 / 3, 1 / 3, 1 / 3])
    # bất biến khi cộng hằng số vào cả hàng (nhờ trừ max)
    assert np.allclose(A.softmax(x), A.softmax(x + 100))
