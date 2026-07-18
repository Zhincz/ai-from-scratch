"""Chỗ để thử nhanh một ý tưởng. Không cần chỉn chu, cứ chạy rồi ghi kết quả
vào README của thí nghiệm này.

Chạy: uv run python experiments/<ten>/run.py
"""


def main():
    # ví dụ: dùng thử PyTorch cho quen
    import torch
    x = torch.randn(3, 3, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    print("x.grad =\n", x.grad)   # phải bằng 2*x


if __name__ == "__main__":
    main()
