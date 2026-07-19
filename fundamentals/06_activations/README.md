# 06 — Activation functions (luyện tập)

Chỗ luyện riêng cho các hàm kích hoạt. Trọng tâm không chỉ là hàm forward mà là
**đạo hàm** của nó — thứ mình cần khi backprop.

`sigmoid` + `sigmoid_grad` làm sẵn làm mẫu. Điền các hàm còn lại trong
`activations.py` tới khi test pass:

```bash
uv run pytest fundamentals/06_activations -q
```

Danh sách cần làm: `tanh`, `relu`, `leaky_relu`, `softmax` (và các `*_grad`).

Test có phần đối chiếu **đạo hàm tự viết với đạo hàm số** (numerical gradient) —
nếu đạo hàm sai công thức sẽ lộ ra ngay.

## Nhìn cho trực quan
Cài xong (hoặc cài được hàm nào) thì vẽ đường cong ra ảnh:

```bash
uv run python fundamentals/06_activations/viz.py   # tạo activations.png
```

## Tự hỏi khi làm
- Vì sao sigmoid/tanh dễ "bão hoà" (đạo hàm ~0 ở hai đầu) → gây vanishing gradient?
- ReLU đạo hàm tại x<0 bằng 0 → "dying ReLU" là gì, leaky_relu chữa ra sao?
- softmax khác các hàm trên chỗ nào (nó ăn cả vector, không phải từng phần tử độc lập)?
