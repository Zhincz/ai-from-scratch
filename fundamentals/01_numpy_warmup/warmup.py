"""Bài khởi động NumPy. Điền các TODO sao cho test pass.
Cố gắng viết vector hoá (không dùng vòng for) khi có thể."""
import numpy as np


def normalize(x):
    """Chuẩn hoá về mean 0, std 1 theo từng cột. x shape (n, d)."""
    # TODO: trả về (x - mean) / std, tính theo axis=0
    raise NotImplementedError


def euclidean(a, b):
    """Khoảng cách euclid giữa hai vector 1 chiều."""
    # TODO
    raise NotImplementedError


def onehot(labels, num_classes):
    """labels shape (n,) các số nguyên 0..num_classes-1 -> ma trận (n, num_classes)."""
    # TODO: gợi ý dùng np.eye
    raise NotImplementedError


def softmax(z):
    """Softmax theo hàng cuối. Nhớ trừ max để ổn định số học."""
    # TODO
    raise NotImplementedError


def accuracy(y_true, y_pred):
    """Tỉ lệ dự đoán đúng."""
    # TODO
    raise NotImplementedError
