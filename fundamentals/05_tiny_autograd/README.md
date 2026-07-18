# 05 — Tiny Autograd

Bài cuối và khó nhất: tự viết một **engine tự động tính đạo hàm** cho các số vô
hướng (scalar), kiểu như micrograd. Sau bài này mình sẽ hiểu vì sao PyTorch chỉ
cần `loss.backward()` là ra hết gradient.

## Ý tưởng
Mỗi giá trị là một `Value` giữ:
- `data`: số thực
- `grad`: đạo hàm của kết quả cuối theo giá trị này (khởi tạo 0)
- một hàm `_backward` biết cách đẩy grad về các "cha" tạo ra nó

Khi gọi `.backward()` trên node cuối: sắp xếp topo toàn đồ thị rồi chạy
`_backward` ngược từ cuối về đầu.

## Việc cần làm
Hoàn thiện `engine.py` (các TODO trong `__add__`, `__mul__`, `tanh`, `backward`)
tới khi test pass:

```bash
uv run pytest fundamentals/05_tiny_autograd -q
```

## Gợi ý quy tắc đạo hàm cục bộ
- `c = a + b`  → `a.grad += c.grad`, `b.grad += c.grad`
- `c = a * b`  → `a.grad += b.data * c.grad`, `b.grad += a.data * c.grad`
- `c = tanh(a)` → `a.grad += (1 - c.data**2) * c.grad`
