import math
from engine import Value


def test_add_mul_grad():
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b + a          # c = a*b + a
    c.backward()
    # dc/da = b + 1 = -2 ; dc/db = a = 2
    assert math.isclose(a.grad, -2.0, abs_tol=1e-6)
    assert math.isclose(b.grad, 2.0, abs_tol=1e-6)


def test_tanh_grad():
    x = Value(0.5)
    y = x.tanh()
    y.backward()
    # dy/dx = 1 - tanh(0.5)^2
    assert math.isclose(x.grad, 1 - math.tanh(0.5) ** 2, abs_tol=1e-6)


def test_reuse_node():
    a = Value(3.0)
    b = a + a              # dùng lại a -> grad phải cộng dồn = 2
    b.backward()
    assert math.isclose(a.grad, 2.0, abs_tol=1e-6)
