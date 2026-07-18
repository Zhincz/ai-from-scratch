"""MLP 1 lớp ẩn, forward + backward tự viết. Điền TODO cho tới khi test pass."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class MLP:
    def __init__(self, in_dim, hidden=16, lr=0.1, epochs=1000, seed=0):
        rng = np.random.default_rng(seed)
        # khởi tạo nhỏ ngẫu nhiên để phá đối xứng
        self.W1 = rng.normal(0, 0.5, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.5, size=(hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr
        self.epochs = epochs
        self.history = []

    def forward(self, X):
        # TODO: tính z1, a1 (ReLU), z2, p (sigmoid).
        # Lưu z1, a1, X vào self để backward dùng lại. Trả về p shape (n,).
        raise NotImplementedError

    def backward(self, y):
        # TODO: dùng các giá trị đã lưu ở forward, tính gradient theo gợi ý
        # trong README rồi cập nhật W1,b1,W2,b2 bằng self.lr.
        raise NotImplementedError

    def fit(self, X, y):
        for _ in range(self.epochs):
            p = self.forward(X)
            eps = 1e-9
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.history.append(float(loss))
            self.backward(y)
        return self

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
