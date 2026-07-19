"""Vẽ đường cong các activation và đạo hàm của chúng để nhìn cho trực quan.
Hàm nào chưa cài (còn TODO) thì tự bỏ qua, cài xong chạy lại là hiện thêm.

Chạy: uv run python fundamentals/06_activations/viz.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import activations as A

HERE = Path(__file__).parent

x = np.linspace(-5, 5, 400)
funcs = [
    ("sigmoid", A.sigmoid, A.sigmoid_grad),
    ("tanh", A.tanh, A.tanh_grad),
    ("relu", A.relu, A.relu_grad),
    ("leaky_relu", A.leaky_relu, A.leaky_relu_grad),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
for name, f, g in funcs:
    try:
        ax1.plot(x, f(x), label=name)
        ax2.plot(x, g(x), label=name)
    except NotImplementedError:
        print(f"bỏ qua {name} (chưa cài)")

ax1.set_title("activation f(x)")
ax2.set_title("đạo hàm f'(x)")
for ax in (ax1, ax2):
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.legend()
plt.tight_layout()
out = HERE / "activations.png"
plt.savefig(out, dpi=110)
print(f"đã lưu {out}")
