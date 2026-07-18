# 03 — Logistic Regression (tự làm)

Phân loại nhị phân. Giống bài 02 nhưng thêm hàm sigmoid và đổi loss sang
binary cross-entropy.

## Việc cần làm
Mở `logistic_regression.py`, hoàn thiện các TODO trong lớp `LogisticRegression`
cho tới khi test pass:

```bash
uv run pytest fundamentals/03_logistic_regression -q
```

## Gợi ý toán
- `p = sigmoid(X @ w + b)`, với `sigmoid(z) = 1 / (1 + e^-z)`
- Loss BCE: `-mean(y*log(p) + (1-y)*log(1-p))`
- Điều hay ho: gradient rút gọn lại **giống hệt** linear regression —
  `grad_w = 1/n * X.T @ (p - y)`, `grad_b = mean(p - y)`. Tự chứng minh xem!
- Dự đoán nhãn: `p >= 0.5`.

Dữ liệu test lấy từ `utils.data.make_blobs2` (2 cụm tách nhau, phải đạt >90% accuracy).
