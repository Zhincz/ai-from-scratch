"""Logistic regression tự cài. Điền TODO cho tới khi test pass.
Tham khảo khung ở fundamentals/02_linear_regression/linear_regression.py."""
import numpy as np


def sigmoid(z):
    # TODO: 1 / (1 + e^-z). Cẩn thận overflow khi z rất âm nếu muốn (không bắt buộc).
    raise NotImplementedError


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.history = []

    def fit(self, X, y):
        # TODO:
        #  1. khởi tạo w = zeros(d), b = 0
        #  2. lặp epochs lần: p = sigmoid(X@w + b)
        #  3. grad_w = 1/n * X.T @ (p - y) ; grad_b = mean(p - y)
        #  4. cập nhật w, b theo lr; lưu BCE loss vào self.history
        raise NotImplementedError

    def predict_proba(self, X):
        # TODO
        raise NotImplementedError

    def predict(self, X):
        # TODO: trả nhãn 0/1 theo ngưỡng 0.5
        raise NotImplementedError
