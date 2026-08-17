import numpy as np
import pytest
import decision_tree as DT
from utils.data import make_blobs2, make_xor, train_test_split


# ---- impurity ----
def test_gini_node_thuan_bang_0():
    assert DT.gini(np.array([10.0, 0.0])) == pytest.approx(0.0)


def test_gini_can_bang_bang_0_5():
    assert DT.gini(np.array([50.0, 50.0])) == pytest.approx(0.5)


def test_entropy_node_thuan_bang_0():
    assert DT.entropy(np.array([0.0, 7.0])) == pytest.approx(0.0)


def test_entropy_can_bang_bang_1_bit():
    assert DT.entropy(np.array([4.0, 4.0])) == pytest.approx(1.0)


def test_entropy_lech_75_25():
    # -0.75*log2(0.75) - 0.25*log2(0.25)
    assert DT.entropy(np.array([3.0, 1.0])) == pytest.approx(0.8112781, abs=1e-6)


def test_impurity_tinh_duoc_nhieu_hang_cung_luc():
    counts = np.array([[2.0, 2.0], [4.0, 0.0]])
    assert np.allclose(DT.gini(counts), [0.5, 0.0])
    assert np.allclose(DT.entropy(counts), [1.0, 0.0])


def test_node_rong_khong_chia_cho_0():
    assert DT.gini(np.array([0.0, 0.0])) == pytest.approx(0.0)
    assert DT.entropy(np.array([0.0, 0.0])) == pytest.approx(0.0)


# ---- best_split ----
def test_best_split_tim_dung_nguong():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    feature, threshold, gain = DT.best_split(X, y, n_classes=2)
    assert feature == 0
    assert threshold == pytest.approx(2.5)   # trung điểm của 2 và 3
    assert gain == pytest.approx(0.5)        # gini 0.5 -> 0


def test_best_split_chon_dung_cot_co_tin_hieu():
    rng = np.random.default_rng(0)
    y = np.array([0] * 50 + [1] * 50)
    cot_nhieu = rng.normal(size=100)
    cot_tin_hieu = y + rng.normal(0, 0.05, size=100)
    X = np.column_stack([cot_nhieu, cot_tin_hieu])
    feature, _, _ = DT.best_split(X, y, n_classes=2)
    assert feature == 1


def test_best_split_tra_none_khi_node_da_thuan():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1, 1, 1])
    assert DT.best_split(X, y, n_classes=2) is None


def test_best_split_ton_trong_min_samples_leaf():
    X = np.arange(10, dtype=float).reshape(-1, 1)
    y = np.array([0] + [1] * 9)          # chỗ cắt đẹp nhất để lại 1 mẫu bên trái
    out = DT.best_split(X, y, n_classes=2, min_samples_leaf=3)
    if out is not None:
        _, threshold, _ = out
        n_trai = int((X[:, 0] <= threshold).sum())
        assert n_trai >= 3 and len(y) - n_trai >= 3


def test_gain_khong_bao_gio_am():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = rng.integers(0, 2, size=60)
    out = DT.best_split(X, y, n_classes=2)
    if out is not None:
        assert out[2] > 0


# ---- fit / predict ----
def test_hoc_duoc_hai_cum_tach_nhau():
    X, y = make_blobs2(n=300, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, ratio=0.7, seed=0)
    tree = DT.DecisionTree(max_depth=3, min_samples_leaf=5).fit(X_tr, y_tr)
    assert tree.score(X_te, y_te) > 0.9


def test_gini_va_entropy_ra_ket_qua_tuong_duong():
    X, y = make_blobs2(n=300, seed=1)
    a = DT.DecisionTree(criterion="gini", max_depth=3).fit(X, y).score(X, y)
    b = DT.DecisionTree(criterion="entropy", max_depth=3).fit(X, y).score(X, y)
    assert abs(a - b) < 0.05


def test_predict_tra_dung_kieu_va_hinh_dang():
    X, y = make_blobs2(n=100, seed=0)
    tree = DT.DecisionTree(max_depth=2).fit(X, y)
    pred = tree.predict(X)
    assert pred.shape == (100,)
    assert pred.dtype.kind == "i"
    assert set(np.unique(pred)) <= {0, 1}


def test_predict_proba_cong_lai_bang_1():
    X, y = make_blobs2(n=100, seed=0)
    proba = DT.DecisionTree(max_depth=3).fit(X, y).predict_proba(X)
    assert proba.shape == (100, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_tha_rong_thi_hoc_thuoc_tap_huan_luyen():
    X, y = make_blobs2(n=200, seed=2)
    tree = DT.DecisionTree(max_depth=None, min_samples_leaf=1).fit(X, y)
    assert tree.score(X, y) == pytest.approx(1.0)


def test_max_depth_duoc_ton_trong():
    X, y = make_blobs2(n=300, seed=0)
    for d in (1, 2, 4):
        assert DT.DecisionTree(max_depth=d).fit(X, y).depth() <= d


def test_cay_mot_tang_co_dung_2_la():
    X, y = make_blobs2(n=200, seed=0)
    assert DT.DecisionTree(max_depth=1).fit(X, y).n_leaves() == 2


def test_min_samples_leaf_lam_cay_gon_lai():
    X, y = make_blobs2(n=400, seed=3)
    to = DT.DecisionTree(min_samples_leaf=1).fit(X, y).n_leaves()
    nho = DT.DecisionTree(min_samples_leaf=30).fit(X, y).n_leaves()
    assert nho < to


# ---- XOR: chỗ thuật toán tham lam trả giá ----
def test_xor_mot_nhat_cat_gan_nhu_vo_dung():
    X, y = make_xor(n=600, seed=0)
    assert DT.DecisionTree(max_depth=1).fit(X, y).score(X, y) < 0.65


def test_xor_tham_lam_khong_giai_duoc_o_hai_tang():
    """Cây 2 tầng thừa sức biểu diễn XOR (4 lá là đủ), nhưng greedy không tìm ra:
    nhát cắt có nghĩa ở x1 <= 0 cho gain gần 0 nên nó không được chọn."""
    X, y = make_xor(n=600, seed=0)
    assert DT.DecisionTree(max_depth=2).fit(X, y).score(X, y) < 0.75


def test_xor_ep_nhat_cat_dau_thi_hai_tang_la_du():
    """Bằng chứng cây 2 tầng tồn tại: ép cắt ở x1 <= 0, mỗi bên chỉ cần một nhát nữa."""
    X, y = make_xor(n=600, seed=0)
    trai = X[:, 0] <= 0
    for mask in (trai, ~trai):
        con = DT.DecisionTree(max_depth=1).fit(X[mask], y[mask])
        assert con.score(X[mask], y[mask]) > 0.95


def test_xor_them_tang_thi_go_lai_duoc():
    X, y = make_xor(n=600, seed=0)
    assert DT.DecisionTree(max_depth=4).fit(X, y).score(X, y) > 0.95


# ---- không ngoại suy được ----
def test_ra_ngoai_vung_huan_luyen_thi_du_doan_khong_doi():
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = (X.ravel() > 5).astype(int)
    tree = DT.DecisionTree(max_depth=3).fit(X, y)
    xa = tree.predict(np.array([[100.0], [1000.0], [10_000.0]]))
    assert len(set(xa)) == 1
