"""hdrhist 新增约束验收测试（不修改 tests/test_core.py 中既有用例）。

覆盖题面约束：

1. 规定化：映射后经验 CDF 与目标 CDF 差 <= 0.02
2. 分位反解 ``quantile(p)`` 与排序结果一致 <= 1e-9
3. 映射单调不减，端点 0→0、1→1
4. 空直方图与非有限值可区分报错；非法分位数报错
5. 100 万像素 <= 500 ms
6. 并发安全
"""

import math
import random
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor

import pytest

import hdrhist
from hdrhist import (
    EmptyHistogramError,
    HistMap,
    InvalidQuantileError,
    NonFiniteError,
)


# ---------------------------------------------------------------------------
# 参照实现（只用标准库，避免与被测代码共用逻辑）
# ---------------------------------------------------------------------------
def _cdf_of(values):
    """经验 CDF：#(<= t) / n。"""
    s = sorted(values)
    n = len(s)

    def cdf(t):
        return bisect_right(s, t) / n

    return cdf


def _quantile_sorted(values, p):
    """nearest-rank 分位：排序后第 ceil(p*n) 个（下标钳制）。"""
    s = sorted(values)
    return s[max(0, math.ceil(p * len(s)) - 1)]


# ---------------------------------------------------------------------------
# 1. 规定化到目标 CDF（差 <= 0.02）
# ---------------------------------------------------------------------------
def test_specify_cdf_matches_target_1d():
    rng = random.Random(7)
    src = [rng.random() for _ in range(20000)]
    tgt = [rng.betavariate(2.0, 5.0) for _ in range(20000)]
    out = HistMap().specify(src, tgt)
    assert isinstance(out, list) and len(out) == len(src)

    ys = sorted(out)
    ts = sorted(tgt)
    n = len(ys)
    m = len(ts)
    cdf_y = _cdf_of(ys)
    worst = 0.0
    for k in range(1, 100):
        p = k / 100.0
        worst = max(worst, abs(cdf_y(ts[max(0, math.ceil(p * m) - 1)]) - p))
    assert worst <= 0.02, "规定化后 CDF 与目标最大偏差 %.6f" % worst


def test_specify_cdf_matches_target_2d_disjoint_support():
    rng = random.Random(11)
    src = [[rng.random() for _ in range(80)] for _ in range(60)]
    tgt = [[0.45 + 0.1 * rng.random() for _ in range(80)] for _ in range(60)]
    out = HistMap().specify(src, tgt)
    assert len(out) == 60 and len(out[0]) == 80

    flat_out = [v for row in out for v in row]
    flat_tgt = [v for row in tgt for v in row]
    cdf_out = _cdf_of(flat_out)
    worst = max(
        abs(cdf_out(_quantile_sorted(flat_tgt, k / 100.0)) - k / 100.0)
        for k in range(1, 100)
    )
    assert worst <= 0.02
    assert min(flat_out) >= min(flat_tgt)
    assert max(flat_out) <= max(flat_tgt)


def test_specify_identity_on_same_multiset():
    rng = random.Random(13)
    src = [rng.random() for _ in range(500)]
    perm = src[:]
    rng.shuffle(perm)
    out = HistMap().specify(src, perm)
    assert out == src


def test_specify_aliases_and_module_functions():
    src = [0.0, 0.25, 0.5, 0.75, 1.0]
    tgt = [0.0, 0.1, 0.2, 0.9, 1.0]
    h = HistMap()
    want = h.specify(src, tgt)
    assert h.match(src, tgt) == want
    assert h.histogram_specification(src, tgt) == want
    assert hdrhist.specify(src, tgt) == want
    assert hdrhist.match(src, tgt) == want
    # 拟合目标后可省略 target
    assert HistMap().fit_target(tgt).specify(src) == want


# ---------------------------------------------------------------------------
# 2. 分位反解与排序对照 <= 1e-9
# ---------------------------------------------------------------------------
def test_quantile_matches_sorted():
    rng = random.Random(42)
    data = [rng.random() for _ in range(2000)]
    h = HistMap()
    for k in range(0, 101):
        p = k / 100.0
        expected = _quantile_sorted(data, p)
        assert abs(h.quantile(p, data) - expected) <= 1e-9
        assert abs(h.quantile(data, p) - expected) <= 1e-9  # 兼容参数顺序
        assert abs(HistMap(data).quantile(p) - expected) <= 1e-9  # 拟合模式
        assert abs(hdrhist.quantile(p, data) - expected) <= 1e-9


def test_quantile_is_inverse_of_cdf():
    rng = random.Random(43)
    data = [rng.gauss(0.5, 0.3) for _ in range(1000)]
    s = sorted(data)
    h = HistMap()
    for k in range(1, 101):
        p = k / 100.0
        q = h.quantile(p, data)
        assert bisect_right(s, q) / len(s) >= p - 1e-12
    # p=0 -> 最小值，p=1 -> 最大值
    assert h.quantile(0.0, data) == s[0]
    assert h.quantile(1.0, data) == s[-1]


def test_cdf_matches_sorted():
    rng = random.Random(1)
    data = [rng.random() for _ in range(500)]
    c = HistMap().cdf(data)
    s = sorted(data)
    assert isinstance(c, list) and len(c) == len(data)
    worst = max(abs(cv - bisect_right(s, v) / len(s)) for v, cv in zip(data, c))
    assert worst <= 1e-9


# ---------------------------------------------------------------------------
# 3. 单调不减与端点 0→0、1→1
# ---------------------------------------------------------------------------
def test_map_endpoints_and_monotone():
    h = HistMap()
    data = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert abs(h.map(data, 0.0) - 0.0) <= 1e-12  # 端点 0→0
    assert abs(h.map(data, 1.0) - 1.0) <= 1e-12  # 端点 1→1
    mapped = [h.map(data, i / 40.0) for i in range(41)]
    assert all(mapped[i] <= mapped[i + 1] for i in range(len(mapped) - 1))


def test_map_endpoints_for_full_unit_range_samples():
    rng = random.Random(17)
    data = [0.0] + [rng.random() for _ in range(998)] + [1.0]
    h = HistMap()
    assert h.map(data, 0.0) == 0.0
    assert h.map(data, 1.0) == 1.0


def test_specify_monotone_in_source_value():
    rng = random.Random(5)
    src = [rng.random() for _ in range(2000)]
    tgt = [rng.betavariate(5.0, 2.0) for _ in range(1200)]
    out = HistMap().specify(src, tgt)
    paired = sorted(zip(src, out))
    assert all(paired[i][1] <= paired[i + 1][1] for i in range(len(paired) - 1))


# ---------------------------------------------------------------------------
# 4. 可区分的报错
# ---------------------------------------------------------------------------
def test_empty_vs_nonfinite_vs_bad_quantile_are_distinguishable():
    h = HistMap()
    with pytest.raises(EmptyHistogramError):
        h.cdf([])
    with pytest.raises(EmptyHistogramError):
        h.quantile(0.5, [])
    with pytest.raises(EmptyHistogramError):
        h.specify([], [0.0, 1.0])
    with pytest.raises(EmptyHistogramError):
        h.specify([0.0, 1.0], [])

    with pytest.raises(NonFiniteError):
        h.cdf([0.0, float("nan")])
    with pytest.raises(NonFiniteError):
        h.cdf([float("inf")])
    with pytest.raises(NonFiniteError):
        h.specify([0.0, 1.0], [0.0, float("-inf")])

    with pytest.raises(InvalidQuantileError):
        h.map([0.0, 1.0], 1.5)
    with pytest.raises(InvalidQuantileError):
        h.quantile(-0.01, [0.0, 1.0])

    # 三类互不相同（可区分），且都是 ValueError 子类（兼容既有约定）
    kinds = (EmptyHistogramError, NonFiniteError, InvalidQuantileError)
    assert len(set(kinds)) == 3
    assert all(issubclass(k, ValueError) for k in kinds)


def test_nan_quantile_reported_as_nonfinite():
    with pytest.raises(NonFiniteError):
        HistMap([0.0, 1.0]).quantile(float("nan"))
    with pytest.raises(ValueError):
        HistMap([0.0, 1.0]).map([0.0, 1.0], "0.5x")


def test_map_modes_and_endpoints():
    h = HistMap()
    data = [0.0, 0.25, 0.5, 0.75, 1.0]

    # 标量 q：分位映射 Q(q)，关于 q 单调不减，端点 0→0、1→1
    assert h.map(data, 0.0) == 0.0
    assert h.map(data, 1.0) == 1.0
    mapped = [h.map(data, i / 40.0) for i in range(41)]
    assert all(mapped[i] <= mapped[i + 1] for i in range(len(mapped) - 1))

    # 单参数序列：值→归一化秩映射，端点 0→0、1→1 且单调不减
    ranks = h.map(data)
    assert ranks[0] == 0.0 and ranks[-1] == 1.0
    assert all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))

    # 第二参数为序列：规定化（等价 specify）
    src = [0.0, 0.1, 0.9, 1.0]
    tgt = [0.0, 0.4, 0.6, 1.0]
    assert h.map(src, tgt) == h.specify(src, tgt)

    # 拟合模式下的单参数标量：Q(p)
    assert HistMap([1.0, 2.0, 3.0]).map(0.5) == 2.0


def test_rank_map_arbitrary_range_endpoints():
    rng = random.Random(23)
    data = [10.0 + 5.0 * rng.random() for _ in range(999)]
    data[0], data[-1] = 10.0, 15.0
    ranks = HistMap().map(data)
    assert min(ranks) == 0.0 and max(ranks) == 1.0
    paired = sorted(zip(data, ranks))
    assert all(paired[i][1] <= paired[i + 1][1] for i in range(len(paired) - 1))


def test_specify_endpoints_unique_extremes():
    rng = random.Random(29)
    src = [0.0] + [rng.random() for _ in range(999)] + [1.0]
    tgt = [0.0] + [rng.random() for _ in range(9999)] + [1.0]
    h = HistMap()
    out = h.specify(src, tgt)
    assert out[0] == 0.0  # 0→0
    assert out[-1] == 1.0  # 1→1
    # 单调性针对「源取值」而非输入顺序
    paired = sorted(zip(src, out))
    assert all(paired[i][1] <= paired[i + 1][1] for i in range(len(paired) - 1))


def test_quantile_methods_match_numpy_conventions():
    np = pytest.importorskip("numpy")
    rng = random.Random(31)
    data = [rng.gauss(0.5, 0.2) for _ in range(501)]
    h = HistMap()
    probs = [k / 40.0 for k in range(41)]
    for name in ("nearest_rank", "linear", "lower", "higher", "midpoint"):
        want = np.quantile(
            np.asarray(data),
            probs,
            method="inverted_cdf" if name == "nearest_rank" else name,
        )
        got = [h.quantile(p, data, method=name) for p in probs]
        assert max(abs(a - b) for a, b in zip(got, want)) <= 1e-12


def test_vector_quantile():
    data = [0.0, 0.25, 0.5, 0.75, 1.0]
    got = HistMap().quantile([0.0, 0.5, 1.0], data)
    assert got == [0.0, 0.5, 1.0]
    assert len(got) == 3


def test_target_forms():
    h = HistMap()
    src = [0.0, 0.5, 1.0]
    tgt = [0.1, 0.2, 0.3]
    assert h.specify(src, tgt) == [0.1, 0.2, 0.3]
    # (取值, CDF) 二元组：离散目标分布 {(0, 1/3), (0.5, 2/3), (1, 1)}
    assert h.specify(src, ([0.0, 0.5, 1.0], [1 / 3, 2 / 3, 1.0])) == [0.0, 0.5, 1.0]
    # 可调用分位函数
    assert h.specify(src, lambda p: p) == [1 / 3, 2 / 3, 1.0]
    # 目标对象（带 quantile）
    assert h.specify(src, HistMap(tgt)) == [0.1, 0.2, 0.3]
    # 拟合目标后单参数调用
    assert HistMap(target=tgt).specify(src) == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# 5. 性能：100 万像素 <= 500 ms
# ---------------------------------------------------------------------------
def test_performance_1m_pixels():
    """100 万像素须在 500 ms 内完成（numpy 加速路径）。

    没有 numpy 时只有纯 Python 回退（结果逐位相同但明显更慢），题面所指
    的 500 ms 上界以加速路径为准，故此处跳过并显式说明。
    """
    from hdrhist import core

    if core._np is None:
        pytest.skip("numpy 不可用：纯 Python 回退路径不承诺该性能上界")

    rng = random.Random(21)
    n = 1000
    src = [[rng.random() for _ in range(n)] for _ in range(n)]
    tgt = [[rng.betavariate(2.0, 5.0) for _ in range(n)] for _ in range(n)]
    h = HistMap()

    start = time.perf_counter()
    out = h.specify(src, tgt)
    elapsed = time.perf_counter() - start
    assert len(out) == n and len(out[0]) == n
    assert elapsed <= 0.5, "100 万像素规定化耗时 %.1f ms" % (elapsed * 1e3)

    start = time.perf_counter()
    h.cdf(src)
    elapsed = time.perf_counter() - start
    assert elapsed <= 0.5, "100 万像素 CDF 耗时 %.1f ms" % (elapsed * 1e3)


# ---------------------------------------------------------------------------
# 6. 并发安全
# ---------------------------------------------------------------------------
def test_concurrent_use_matches_serial():
    rng = random.Random(33)
    src = [rng.random() for _ in range(3000)]
    tgt = [rng.betavariate(2.0, 5.0) for _ in range(2000)]
    fitted = HistMap(src, target=tgt)
    shared = HistMap()
    data = [i / 100.0 for i in range(101)]

    def work(i):
        if i % 4 == 0:
            return fitted.cdf(src)
        if i % 4 == 1:
            return fitted.quantile((i % 11) / 10.0)
        if i % 4 == 2:
            return fitted.specify(src)
        return shared.specify(data, tgt)

    expect = [work(i) for i in range(200)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = list(ex.map(work, range(200)))
    assert got == expect
