"""Decision tree phân loại, cài tay theo kiểu CART.

CART luôn tách nhị phân: mỗi node hỏi đúng một câu dạng `x[feature] <= threshold`.
Chọn câu hỏi nào thì dựa vào độ giảm impurity (gini hoặc entropy).

Cách tìm ngưỡng ở đây là sắp xếp cột một lần rồi quét bằng tổng tích luỹ, nên mỗi
node tốn O(n log n) chứ không phải O(n^2) như cách lặp hai vòng.
"""
import numpy as np


def class_counts(y, n_classes):
    return np.bincount(y.astype(int), minlength=n_classes).astype(float)


# ---- Impurity ----
# Hai hàm này nhận counts shape (..., n_classes) nên dùng được cho cả một node
# lẫn cả một loạt ngưỡng cùng lúc.
def gini(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1)
    safe = np.where(total == 0, 1.0, total)[..., None]
    p = counts / safe
    out = 1.0 - np.sum(p * p, axis=-1)
    return np.where(total == 0, 0.0, out)      # node rỗng coi như thuần


def entropy(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1)
    safe = np.where(total == 0, 1.0, total)[..., None]
    p = counts / safe
    logp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    out = -np.sum(p * logp, axis=-1)
    return np.where(total == 0, 0.0, out)


IMPURITY = {"gini": gini, "entropy": entropy}


def best_split(X, y, n_classes, criterion="gini", min_samples_leaf=1):
    """Trả về (feature, threshold, gain) tốt nhất, hoặc None nếu không tách được.

    gain = impurity(cha) - trung bình có trọng số impurity của hai nhánh con.
    """
    imp = IMPURITY[criterion]
    n, d = X.shape
    parent = imp(class_counts(y, n_classes))

    # onehot để cộng dồn số mẫu mỗi lớp khi quét từ trái sang
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y.astype(int)] = 1.0

    best = None
    for j in range(d):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        cum = np.cumsum(onehot[order], axis=0)
        total = cum[-1]

        n_left = np.arange(1, n)          # cắt sau vị trí i thì nhánh trái có i+1 mẫu
        left = cum[:-1]
        right = total - left

        ok = xs[:-1] != xs[1:]            # không cắt giữa hai giá trị bằng nhau
        ok &= n_left >= min_samples_leaf
        ok &= (n - n_left) >= min_samples_leaf
        if not ok.any():
            continue

        child = (n_left * imp(left) + (n - n_left) * imp(right)) / n
        gain = np.where(ok, parent - child, -np.inf)

        k = int(np.argmax(gain))
        if best is None or gain[k] > best[2]:
            # ngưỡng lấy trung điểm hai giá trị liền kề, đừng lấy thẳng giá trị
            best = (j, (xs[k] + xs[k + 1]) / 2.0, float(gain[k]))

    if best is None or best[2] <= 0:
        return None
    return best


class Node:
    """Một node của cây. Lá thì feature = None, còn lại là node hỏi."""

    def __init__(self, counts, depth):
        self.counts = counts
        self.depth = depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.feature is None


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1,
                 min_samples_split=2):
        if criterion not in IMPURITY:
            raise ValueError("criterion phải là 'gini' hoặc 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = 0
        self.n_features = 0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.n_classes = int(y.max()) + 1
        self.n_features = X.shape[1]
        self.root = self._grow(X, y, depth=0)
        return self

    def _grow(self, X, y, depth):
        counts = class_counts(y, self.n_classes)
        node = Node(counts, depth)

        # dừng lại thành lá
        if (counts > 0).sum() <= 1:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node
        if len(y) < self.min_samples_split:
            return node

        split = best_split(X, y, self.n_classes, self.criterion, self.min_samples_leaf)
        if split is None:
            return node

        feature, threshold, _ = split
        mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def _leaf_for(self, x):
        node = self.root
        while not node.is_leaf():
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), self.n_classes))
        for i, x in enumerate(X):
            c = self._leaf_for(x).counts
            out[i] = c / c.sum()
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1).astype(int)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y).astype(int)).mean())

    # ---- mấy hàm tiện tay ----
    def n_leaves(self):
        def walk(node):
            return 1 if node.is_leaf() else walk(node.left) + walk(node.right)
        return walk(self.root)

    def depth(self):
        def walk(node):
            return 0 if node.is_leaf() else 1 + max(walk(node.left), walk(node.right))
        return walk(self.root)

    def print_tree(self, feature_names=None):
        names = feature_names or [f"x[{j}]" for j in range(self.n_features)]

        def walk(node, pad, tag):
            if node.is_leaf():
                k = int(node.counts.argmax())
                print(f"{pad}{tag}-> lớp {k}  ({int(node.counts[k])}/{int(node.counts.sum())})")
                return
            imp = IMPURITY[self.criterion](node.counts)
            print(f"{pad}{tag}{names[node.feature]} <= {node.threshold:.3f}"
                  f"   ({self.criterion}={imp:.3f}, n={int(node.counts.sum())})")
            walk(node.left, pad + "    ", "Có:    ")
            walk(node.right, pad + "    ", "Không: ")

        walk(self.root, "", "")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from utils.data import make_blobs2, make_xor, train_test_split

    X, y = make_blobs2(n=400, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, ratio=0.7, seed=0)
    tree = DecisionTree(max_depth=3, min_samples_leaf=5).fit(X_tr, y_tr)
    print(f"blobs2: train={tree.score(X_tr, y_tr):.3f}  test={tree.score(X_te, y_te):.3f}"
          f"  lá={tree.n_leaves()}  sâu={tree.depth()}\n")
    tree.print_tree(feature_names=["x", "y"])

    # XOR: chỗ thuật toán tham lam trả giá
    print("\nXOR:")
    Xx, yx = make_xor(n=600, seed=0)
    print("  nhát cắt đầu greedy chọn:", best_split(Xx, yx.astype(int), 2))
    for d in (1, 2, 3, 4, 6):
        t = DecisionTree(max_depth=d).fit(Xx, yx)
        print(f"  max_depth={d}: accuracy={t.score(Xx, yx):.3f}  lá={t.n_leaves()}")

    trai = Xx[:, 0] <= 0
    print("  nhưng nếu ÉP nhát cắt đầu ở x <= 0 rồi mỗi bên cắt thêm 1 nhát:")
    for ten, mask in (("trái", trai), ("phải", ~trai)):
        con = DecisionTree(max_depth=1).fit(Xx[mask], yx[mask])
        print(f"    {ten}: accuracy={con.score(Xx[mask], yx[mask]):.3f}")
    print("  -> cây 2 tầng giải được XOR, nhưng greedy không bao giờ tìm ra nó.")
