# 04 — Neural Net từ đầu (backprop tay)

Một mạng 2 lớp (1 lớp ẩn ReLU) phân loại nhị phân, **tự viết cả forward lẫn
backward**. Đây là bài quan trọng nhất để hiểu backpropagation thật sự làm gì.

## Kiến trúc
```
X (n, d) --W1,b1--> z1 --ReLU--> a1 --W2,b2--> z2 --sigmoid--> p
```

## Việc cần làm
Hoàn thiện `mlp.py`:
- `forward`: lưu lại các giá trị trung gian (z1, a1, ...) để dùng khi backward.
- `backward`: lan truyền ngược gradient qua sigmoid → W2 → ReLU → W1.

Chạy test:
```bash
uv run pytest fundamentals/04_neural_net -q
```

## Gợi ý backprop (chain rule)
- `dz2 = p - y`               (đạo hàm BCE qua sigmoid, gộp lại đẹp như vậy)
- `dW2 = a1.T @ dz2 / n`,  `db2 = mean(dz2)`
- `da1 = dz2 @ W2.T`
- `dz1 = da1 * (z1 > 0)`      (đạo hàm ReLU)
- `dW1 = X.T @ dz1 / n`,   `db1 = mean(dz1)`

Kiểm tra gradient bằng numerical gradient nếu nghi ngờ — test đã lo phần đó.
