"""Mini autograd cho scalar. Điền TODO cho tới khi test pass.
Cảm hứng từ micrograd của Andrej Karpathy."""
import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None      # mặc định: node lá không đẩy gì
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # TODO: cộng dồn gradient cho self và other (xem README)
            raise NotImplementedError
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO
            raise NotImplementedError
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # TODO
            raise NotImplementedError
        out._backward = _backward
        return out

    def backward(self):
        # sắp xếp topo: cha đứng trước con
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # TODO: đặt self.grad = 1.0 rồi chạy _backward ngược theo topo
        raise NotImplementedError

    # cho phép viết 2 + a, 2 * a
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
