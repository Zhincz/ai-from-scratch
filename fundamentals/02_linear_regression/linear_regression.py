"""Linear regression cài tay bằng gradient descent — không dùng sklearn.

Đây là bài MẪU hoàn chỉnh: đọc để nắm cách tổ chức một bài từ đầu (model có
fit/predict, loss, gradient tính tay), rồi các bài sau mình tự làm theo khung này.

Mô hình:  y_hat = X @ w + b
Loss   :  MSE = mean((y_hat - y)^2)
Gradient (đạo hàm MSE theo w, b) tính tay bên dưới.
"""
import numpy as np


class LinearRegression:
    def __init__(self, lr=0.1, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.history = []

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.history = []

        for _ in range(self.epochs):
            y_hat = X @ self.w + self.b
            err = y_hat - y                      # (n,)

            # dMSE/dw = 2/n * X^T @ err ; dMSE/db = 2/n * sum(err)
            grad_w = (2 / n) * (X.T @ err)
            grad_b = (2 / n) * err.sum()

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            self.history.append(float(np.mean(err ** 2)))
        return self

    def predict(self, X):
        return X @ self.w + self.b


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


if __name__ == "__main__":
    # chạy thử: python linear_regression.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.data import make_linear, train_test_split

    X, y = make_linear(n=200, slope=2.0, bias=1.0, seed=1)
    Xtr, Xte, ytr, yte = train_test_split(X, y)

    model = LinearRegression(lr=0.1, epochs=300).fit(Xtr, ytr)
    print(f"học được:  w={model.w[0]:.3f}  b={model.b:.3f}  (thật: w=2, b=1)")
    print(f"MSE test:  {mse(yte, model.predict(Xte)):.4f}")
