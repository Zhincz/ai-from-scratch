"""Các hàm kích hoạt + đạo hàm của chúng, cài tay bằng NumPy.

sigmoid làm sẵn làm mẫu (cả forward lẫn đạo hàm). Các hàm còn lại điền TODO cho
tới khi test pass. Mỗi hàm đi kèm một hàm `*_grad` trả đạo hàm theo đầu vào — đây
mới là phần quan trọng, vì lúc backprop mình cần đúng cái đạo hàm này.
"""
import numpy as np


# ---- sigmoid: BÀI MẪU ----
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


# ---- tanh ----
def tanh(x):
    # TODO: dùng np.tanh
    raise NotImplementedError


def tanh_grad(x):
    # TODO: đạo hàm tanh là 1 - tanh(x)^2
    raise NotImplementedError


# ---- ReLU ----
def relu(x):
    # TODO: max(0, x), vector hoá
    raise NotImplementedError


def relu_grad(x):
    # TODO: 1 khi x > 0, còn lại 0 (tại 0 quy ước là 0)
    raise NotImplementedError


# ---- Leaky ReLU ----
def leaky_relu(x, alpha=0.01):
    # TODO: x khi x > 0, alpha*x khi x <= 0
    raise NotImplementedError


def leaky_relu_grad(x, alpha=0.01):
    # TODO
    raise NotImplementedError


# ---- softmax (theo hàng) ----
def softmax(x):
    # TODO: trừ max mỗi hàng cho ổn định, rồi exp / tổng. x shape (n, k).
    raise NotImplementedError
