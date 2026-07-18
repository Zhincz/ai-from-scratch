"""Helper vẽ nhanh, khỏi lặp lại matplotlib mỗi bài."""
import matplotlib.pyplot as plt


def scatter_fit(x, y, line_fn=None, title=""):
    """Vẽ điểm dữ liệu 1 chiều, nếu có line_fn thì vẽ đường dự đoán đè lên."""
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, s=15, alpha=0.6, label="data")
    if line_fn is not None:
        xs = [x.min(), x.max()]
        plt.plot(xs, [line_fn(v) for v in xs], "r-", label="fit")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return plt


def loss_curve(losses, title="loss"):
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.tight_layout()
    return plt
