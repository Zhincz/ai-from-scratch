"""Vài hàm sinh dữ liệu giả để test thuật toán khi mình chưa muốn load dataset thật."""
import numpy as np


def make_linear(n=100, slope=2.0, bias=1.0, noise=0.3, seed=0):
    """Sinh dữ liệu 1 chiều quanh đường y = slope*x + bias, thêm nhiễu gaussian."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, size=n)
    y = slope * x + bias + rng.normal(0, noise, size=n)
    return x.reshape(-1, 1), y


def make_blobs2(n=200, seed=0):
    """Hai cụm điểm 2D tách nhau, nhãn 0/1 — dùng cho phân loại nhị phân."""
    rng = np.random.default_rng(seed)
    half = n // 2
    a = rng.normal([-2, -2], 1.0, size=(half, 2))
    b = rng.normal([2, 2], 1.0, size=(n - half, 2))
    x = np.vstack([a, b])
    y = np.concatenate([np.zeros(half), np.ones(n - half)])
    # xáo trộn để nhãn không xếp theo thứ tự
    idx = rng.permutation(n)
    return x[idx], y[idx]


def train_test_split(x, y, ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    cut = int(len(x) * ratio)
    tr, te = idx[:cut], idx[cut:]
    return x[tr], x[te], y[tr], y[te]
