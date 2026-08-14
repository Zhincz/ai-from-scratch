# 07 — Decision Tree (làm sẵn)

Cây quyết định phân loại, cài theo kiểu **CART**: mỗi node hỏi đúng một câu dạng
`x[feature] <= threshold`, luôn tách nhị phân.

Bài này mình để **code hoàn chỉnh** (giống bài 02) chứ không phải khung TODO, vì
phần hay của nó nằm ở chỗ chạy thử và soi hành vi chứ không phải ở việc gõ lại.

```bash
uv run pytest fundamentals/07_decision_tree -q     # 25 test
uv run python fundamentals/07_decision_tree/decision_tree.py
uv run python fundamentals/07_decision_tree/viz.py
```

## Toán

Độ "không thuần" của một node, với $p_c$ là tỉ lệ lớp $c$:

- Gini: $G = 1 - \sum_c p_c^2$ — node thuần thì 0, nhị phân 50/50 thì 0.5
- Entropy: $H = -\sum_c p_c \log_2 p_c$ — node thuần thì 0, nhị phân 50/50 thì 1 bit

Chọn chỗ tách theo độ giảm impurity, nhớ đánh trọng số theo kích thước nhánh:

```
gain = imp(cha) - (n_trái/n * imp(trái) + n_phải/n * imp(phải))
```

Trọng số là bắt buộc. Bỏ nó đi thì cây sẽ mê những nhánh bé tí mà thuần.

## Vài chỗ đáng chú ý trong code

**Tìm ngưỡng bằng tổng tích luỹ.** Cách ngây thơ là hai vòng lặp (mỗi cột × mỗi
ngưỡng), tốn $O(n^2)$. Ở đây mình sắp xếp cột một lần rồi `cumsum` số mẫu mỗi
lớp, nên tính được impurity của **mọi** ngưỡng cùng lúc bằng phép vector. Mỗi
node còn $O(n \log n)$.

**Ngưỡng lấy trung điểm** giữa hai giá trị liền kề, không lấy thẳng giá trị. Lấy
thẳng thì ranh giới dính sát vào một điểm dữ liệu.

**Chỉ xét chỗ hai giá trị khác nhau** (`xs[:-1] != xs[1:]`). Cắt giữa hai giá trị
bằng nhau là vô nghĩa và sinh nhánh rỗng.

**Node rỗng coi như thuần** (impurity 0), không phải 1.

## Hai thứ chạy ra mới thấy

**1. Cây không ngoại suy được.** Huấn luyện với $x \in [0,10]$ rồi hỏi $x = 100$,
$x = 1000$, $x = 10000$ — cả ba trả về **cùng một nhãn**, vì đều rơi vào lá ngoài
cùng. Đây là lý do đừng dùng cây để dự báo chuỗi thời gian có xu hướng nếu chưa
khử xu hướng.

**2. XOR — chỗ thuật toán tham lam trả giá.** Chạy `decision_tree.py` sẽ thấy:

```
nhát cắt đầu greedy chọn: (0, -1.377, gain=0.0055)
max_depth=1: accuracy=0.532  lá=2
max_depth=2: accuracy=0.617  lá=4
max_depth=3: accuracy=0.948  lá=6
max_depth=4: accuracy=0.998  lá=8
max_depth=6: accuracy=1.000  lá=10
```

Cây 2 tầng (4 lá) **thừa sức** biểu diễn XOR. Bằng chứng: ép nhát cắt đầu ở
$x_1 \le 0$ thì mỗi bên chỉ cần một nhát nữa là đạt 0.993 và 0.997.

Nhưng greedy không tìm ra, vì xét riêng từng cột thì nhát cắt có nghĩa cho gain
**gần 0** — nó thua cả một nhát cắt rìa vô nghĩa có gain 0.0055 nhờ nhiễu. Phải
tới 6 tầng và 10 lá cây mới gò được về 100%, và lúc đó nó đang chắp vá chứ không
phải đã "hiểu" XOR.

Đây chính là cái giá của tham lam: cây chỉ nhìn một bước, không thấy được một
nước đi hiện tại vô lợi nhưng mở đường cho hai nước đi rất tốt ở tầng sau. Tìm
cây tối ưu toàn cục là bài toán NP-đầy đủ (Hyafil & Rivest, 1976) nên không ai
làm, và ta sống chung với hạn chế này.

## Liên quan

- Bài 03 (logistic regression) là đối chứng tốt: nó vẽ ranh giới thẳng, cây vẽ
  ranh giới bậc thang song song trục. XOR thì logistic chịu chết hẳn, cây còn gò
  được.
- Điểm yếu lớn nhất của một cây đơn là **bất ổn định**: đổi vài mẫu là cả cây
  khác. Đó là lý do người ta trồng cả rừng rồi lấy trung bình.
