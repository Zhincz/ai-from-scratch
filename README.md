# ai-from-scratch

Sân tập tự học AI của mình. Hai phần:

- **`fundamentals/`** — cài lại các thuật toán từ số 0 bằng NumPy, không dùng
  framework, để hiểu bản chất (gradient, backprop, autograd).
- **`experiments/`** — playground thử nhanh PyTorch / sklearn / HuggingFace, mỗi
  ý tưởng một thư mục.

## Cài đặt

Dự án dùng [uv](https://docs.astral.sh/uv/). Deps đã khai trong `pyproject.toml`,
chỉ cần:

```bash
uv sync
```

## Fundamentals — lộ trình

Làm lần lượt. Bài 02 làm **sẵn hoàn chỉnh** làm mẫu; các bài còn lại là khung +
TODO để mình tự điền cho tới khi test pass.

| # | Bài | Trạng thái | Học được gì |
|---|-----|-----------|-------------|
| 01 | NumPy warmup | tự làm | vector hoá, broadcasting, softmax |
| 02 | Linear regression | ✅ mẫu | gradient descent, MSE, gradient tay |
| 03 | Logistic regression | tự làm | sigmoid, cross-entropy |
| 04 | Neural net | tự làm | forward/backward, backprop tay |
| 05 | Tiny autograd | tự làm | tự động tính đạo hàm (kiểu micrograd) |
| 06 | Activation functions | tự làm | sigmoid/tanh/relu/softmax + đạo hàm, vanishing gradient |

Chạy test một bài:

```bash
uv run pytest fundamentals/02_linear_regression -q
```

Chạy hết test:

```bash
uv run pytest -q
```

Mới đầu các bài tự-làm sẽ FAIL (do còn `NotImplementedError`) — đó là chuyện
bình thường, điền code vào là xanh dần.

## Experiments

```bash
cp -r experiments/_template experiments/2026-07-18_ten-thi-nghiem
uv run python experiments/2026-07-18_ten-thi-nghiem/run.py
```

## Notebooks

```bash
uv run jupyter lab
```

Mở `notebooks/00_sanity_check.ipynb` để kiểm tra môi trường.
