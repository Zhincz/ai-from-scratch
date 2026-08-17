"""Vẽ ranh giới quyết định của cây theo độ sâu, và ba đường impurity.

    uv run python fundamentals/07_decision_tree/viz.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from decision_tree import DecisionTree, entropy, gini  # noqa: E402
from utils.data import make_blobs2, make_xor  # noqa: E402

OUT = Path(__file__).parent


def ve_ranh_gioi(ax, tree, X, y, tieu_de):
    pad = 0.5
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 200)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 200)
    gx, gy = np.meshgrid(xs, ys)
    zz = tree.predict(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    # màu nền lấy đúng hai đầu của coolwarm cho khớp màu điểm
    ax.contourf(gx, gy, zz, alpha=0.22, levels=[-0.5, 0.5, 1.5], colors=["#3B4CC0", "#B40426"])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=6, cmap="coolwarm", edgecolors="none")
    ax.set_title(tieu_de, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def hinh_ranh_gioi(X, y, ten_file, tieu_de_chung, depths=(1, 2, 3, 6)):
    fig, axes = plt.subplots(1, len(depths), figsize=(3.2 * len(depths), 3.2))
    for ax, d in zip(axes, depths):
        tree = DecisionTree(max_depth=d).fit(X, y)
        ve_ranh_gioi(ax, tree, X, y,
                     f"depth={d} · lá={tree.n_leaves()} · acc={tree.score(X, y):.3f}")
    fig.suptitle(tieu_de_chung)
    fig.tight_layout()
    fig.savefig(OUT / ten_file, dpi=130)
    print("đã lưu", ten_file)


def hinh_impurity():
    p = np.linspace(0.001, 0.999, 400)
    counts = np.column_stack([p, 1 - p])
    plt.figure(figsize=(6, 4))
    plt.plot(p, entropy(counts), label="Entropy (đỉnh 1.0)")
    plt.plot(p, gini(counts), label="Gini (đỉnh 0.5)")
    plt.plot(p, np.minimum(p, 1 - p), "--", label="Misclassification")
    plt.axvline(0.5, color="gray", lw=0.8)
    plt.xlabel("tỉ lệ lớp dương p"); plt.ylabel("impurity")
    plt.title("Ba thước đo impurity")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / "impurity.png", dpi=130)
    print("đã lưu impurity.png")


if __name__ == "__main__":
    hinh_impurity()

    X, y = make_blobs2(n=400, seed=0)
    hinh_ranh_gioi(X, y, "boundary_blobs.png",
                   "Hai cụm tách nhau: sâu thêm chỉ để bám nhiễu")

    Xx, yx = make_xor(n=600, seed=0)
    hinh_ranh_gioi(Xx, yx, "boundary_xor.png",
                   "XOR: greedy phải bò tới 6 tầng mới gò xong")
