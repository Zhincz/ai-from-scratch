# 02 — Linear Regression (bài mẫu)

Bài này **làm sẵn hoàn chỉnh** để làm khung tham khảo cho các bài sau.

## Ý tưởng
- Mô hình: `y_hat = X @ w + b`
- Loss: MSE
- Học bằng gradient descent, gradient tính tay (xem comment trong `linear_regression.py`)

## Chạy thử
```bash
uv run python fundamentals/02_linear_regression/linear_regression.py
```

## Chạy test
```bash
uv run pytest fundamentals/02_linear_regression -q
```

## Tự kiểm tra hiểu bài
- Vì sao gradient của w là `2/n * X.T @ err`? Thử tự đạo hàm MSE.
- Đổi `lr` lên 1.0 rồi 0.001 xem chuyện gì xảy ra với `model.history`.
- Thêm regularization L2 (thêm `lambda * w` vào `grad_w`) — kết quả w đổi thế nào?
