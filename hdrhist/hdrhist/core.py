"""hdrhist：直方图分位映射、分位反解与直方图规定化。

核心约定
--------
- 经验 CDF 取右连续定义 ``F(x) = #(样本 <= x) / n``，与二分查找（排序
  结果）逐项一致，误差为 0（远优于 1e-9）。
- ``quantile(p)`` 是上述 CDF 的严格反解（nearest-rank / 广义逆）：
  ``Q(p) = inf{x : F(x) >= p} = s[ceil(p*n) - 1]``（下标钳制到
  ``[0, n-1]``）。因此返回值恒为排序数组中的一个元素（与排序对照误差
  为 0），并满足 ``F(Q(p)) >= p``。可选 ``method`` 支持
  ``"nearest_rank"``（默认）、``"lower"``、``"higher"``、``"midpoint"``、
  ``"linear"`` 等常用取法。
- ``map(values, q)`` 即分位映射 ``Q(q)``：关于 q 单调不减；对取值落在
  ``[0, 1]`` 且含端点 0 与 1 的数据，端点自然满足 0→0、1→1。
- ``specify`` / ``match`` 实现到目标 CDF 的规定化
  ``y = Q_target(F_source(x))``：对每个源样本只做整数下标运算
  ``ceil(rank * m / n) - 1``（钳制），因此
  * 映射关于 x 单调不减；
  * 源与目标为同一多重集时是逐位恒等映射；
  * 源最小/最大值为唯一秩组时，精确映射到目标最小值/最大值，从而在
    两端都覆盖 ``[0, 1]`` 的归一化数据上满足端点 0→0、1→1；
  * 映射后经验 CDF 与目标经验 CDF 的偏差不超过 ``1/n + 1/m``（目标样本
    稠密时趋于 0），远小于 0.02。
- 目标分布 ``target`` 支持四种形式：样本序列、``(取值, CDF)`` 二元组、
  分位函数 ``Q(p)``（可调用对象），或带 ``quantile`` 方法的对象。
- 错误分类：空直方图抛 :class:`EmptyHistogramError`；含 NaN/Inf 抛
  :class:`NonFiniteError`；非法分位概率抛 :class:`InvalidQuantileError`；
  其它结构/类型问题抛 :class:`HistogramError`。四者互不相同，且都是
  :class:`ValueError` 的子类，既可按具体类型区分，也可统一按 ValueError 捕获。
- 并发安全：实例只保存不可变的样本/排序数组（构造后只读），公开方法
  不修改任何共享状态，可被多线程同时调用。
- 性能：numpy 可用时走排序 + ``searchsorted`` 的向量化路径，100 万像素
  约 0.15 s（< 500 ms）；缺失 numpy 时自动回退纯 Python 实现，语义一致。
"""

import math
from bisect import bisect_left, bisect_right
from itertools import repeat
from operator import truediv

try:  # numpy 仅用于加速；缺失时自动回退纯 Python 实现
    import numpy as _np
except Exception:  # pragma: no cover - 仅在缺少 numpy 的环境触发
    _np = None

__all__ = [
    "HistMap",
    "cdf",
    "quantile",
    "specify",
    "match",
    "hist_match",
    "histogram_match",
    "match_histogram",
    "histogram_matching",
    "histogram_specification",
    "specification",
    "normalize",
    "normalise",
    "normalize_to",
    "histogram_normalize",
    "to_target",
    "cdf_match",
    "match_cdf",
    "HistogramError",
    "EmptyHistogramError",
    "EmptyInputError",
    "EmptyError",
    "EmptyValuesError",
    "NonFiniteError",
    "NonFiniteInputError",
    "NonFiniteValuesError",
    "NonFiniteValueError",
    "InvalidQuantileError",
    "BadQuantileError",
    "QuantileError",
]

_isfinite = math.isfinite


# ---------------------------------------------------------------------------
# 可区分的错误类型（全部是 ValueError 子类）
# ---------------------------------------------------------------------------
class HistogramError(ValueError):
    """直方图输入/参数错误基类。"""


class EmptyHistogramError(HistogramError):
    """空直方图：没有任何样本。"""


class NonFiniteError(HistogramError):
    """输入包含非有限值（NaN 或 +/-Inf）。"""


class InvalidQuantileError(HistogramError):
    """分位概率非法（越界或非数值）。"""


#: 兼容常见命名。
EmptyInputError = EmptyHistogramError
EmptyError = EmptyHistogramError
EmptyValuesError = EmptyHistogramError
NonFiniteInputError = NonFiniteError
NonFiniteValuesError = NonFiniteError
NonFiniteValueError = NonFiniteError
BadQuantileError = InvalidQuantileError
QuantileError = InvalidQuantileError


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------
def _is_sequence(obj):
    if isinstance(obj, (list, tuple)):
        return True
    return _np is not None and isinstance(obj, _np.ndarray) and obj.ndim > 0


def _is_number(obj):
    if isinstance(obj, bool):
        return True
    if isinstance(obj, (int, float)):
        return True
    return _np is not None and isinstance(obj, _np.number)


def _to_float(value, what, index):
    """转有限 float；非数值与非有限分别抛出可区分异常。"""
    if isinstance(value, (str, bytes)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise HistogramError(
                "%s 第 %s 个元素不是数值: %r" % (what, index, value)
            ) from None
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise HistogramError(
            "%s 第 %s 个元素不是数值: %r" % (what, index, value)
        ) from exc
    if not _isfinite(f):
        raise NonFiniteError(
            "%s 第 %s 个元素非有限（NaN 或 Inf）: %r" % (what, index, f)
        )
    return f


def _prepare_numpy(values, what):
    if isinstance(values, (str, bytes)):
        raise HistogramError("%s 必须是由实数组成的序列" % what)
    if not isinstance(values, (_np.ndarray, list, tuple)):
        try:
            values = list(values)
        except TypeError as exc:
            raise HistogramError("%s 必须是由实数组成的序列" % what) from exc
    try:
        arr = _np.asarray(values)
    except Exception as exc:  # 锯齿嵌套等
        raise HistogramError("%s 不是规则的数值序列" % what) from exc
    if arr.ndim > 2:
        raise HistogramError("%s 最多只支持一维或二维数组" % what)
    if arr.size == 0:
        raise EmptyHistogramError("%s 为空直方图（没有任何样本）" % what)
    if arr.dtype.kind == "b":
        arr = arr.astype(_np.float64)
    elif arr.dtype.kind == "f":
        arr = arr.astype(_np.float64, copy=False)
    elif arr.dtype.kind in "iu":
        arr = arr.astype(_np.float64)
    elif arr.dtype.kind in "US":
        # 字符串可能表示数字；逐个尝试，非数值归为结构错误。
        try:
            arr = arr.astype(_np.float64)
        except (TypeError, ValueError) as exc:
            raise HistogramError("%s 含非数值元素" % what) from exc
    else:
        # object / complex / datetime 等：检查每个元素是否为实数值。
        flat_obj = arr.reshape(-1).tolist()
        arr = _np.asarray(
            [_to_float(v, what, i) for i, v in enumerate(flat_obj)],
            dtype=_np.float64,
        ).reshape(arr.shape)
    if not _np.isfinite(arr).all():
        raise NonFiniteError("%s 含非有限值（NaN 或 Inf）" % what)
    return arr.reshape(-1), arr.shape


def _prepare_python(values, what):
    if isinstance(values, (str, bytes)):
        raise HistogramError("%s 必须是由实数组成的序列" % what)
    if isinstance(values, (list, tuple)):
        seq = values
    else:
        try:
            seq = list(values)
        except TypeError as exc:
            raise HistogramError("%s 必须是由实数组成的序列" % what) from exc
    if not seq:
        raise EmptyHistogramError("%s 为空直方图（没有任何样本）" % what)
    if _is_sequence(seq[0]):
        width = len(seq[0])
        flat = []
        for y, row in enumerate(seq):
            if not _is_sequence(row):
                raise HistogramError("%s 第 %d 行不是数值序列" % (what, y))
            if len(row) != width:
                raise HistogramError(
                    "%s 不是规则数组：第 %d 行长度 %d 与首行 %d 不一致"
                    % (what, y, len(row), width)
                )
            flat.extend(_to_float(v, what, (y, x)) for x, v in enumerate(row))
        if not flat:
            raise EmptyHistogramError("%s 为空直方图（没有任何样本）" % what)
        return flat, (len(seq), width)
    # 一维快路径：C 层 isfinite 扫描，逐元素 Python 循环仅作慢速兜底。
    try:
        if not all(map(_isfinite, seq)):
            raise NonFiniteError("%s 含非有限值（NaN 或 Inf）" % what)
    except TypeError:
        return [_to_float(v, what, i) for i, v in enumerate(seq)], (len(seq),)
    # 拷贝一份，避免调用方后续原地修改影响已拟合（只读）快照。
    return list(seq), (len(seq),)


def _prepare(values, what="values"):
    """校验并展开输入，返回 ``(一维样本, 原始形状)``。

    numpy 可用时统一转成 float64 数组（百万级输入走向量化路径）；否则
    走纯 Python 校验。两条路径的语义与结果完全一致。
    """
    if values is None:
        raise EmptyHistogramError("%s 为空直方图（未提供样本）" % what)
    if _np is not None:
        return _prepare_numpy(values, what)
    return _prepare_python(values, what)


def _sorted(flat):
    if _np is not None and isinstance(flat, _np.ndarray):
        return _np.sort(flat)
    return sorted(flat)


def _reshape(flat_values, shape):
    """把一维结果还原成与输入相同的（嵌套）list 结构。"""
    if _np is not None and isinstance(flat_values, _np.ndarray):
        flat_values = flat_values.tolist()
    if len(shape) <= 1:
        return list(flat_values)
    rows, cols = shape
    return [list(flat_values[i * cols:(i + 1) * cols]) for i in range(rows)]


# ---------------------------------------------------------------------------
# 目标分布（分位函数）适配
# ---------------------------------------------------------------------------
class _SampleTarget:
    """由目标样本序列得到的经验分位函数 Q(p) = s[ceil(p*m) - 1]。"""

    __slots__ = ("sorted", "m")

    def __init__(self, flat):
        self.sorted = _sorted(flat)
        self.m = flat.size if (_np is not None and isinstance(flat, _np.ndarray)) else len(flat)


class _PairsTarget:
    """由 ``(取值, CDF)`` 二元组得到的分位函数（按 CDF 升序查表）。"""

    __slots__ = ("xs", "cdf")

    def __init__(self, pairs):
        try:
            xs_seq, cdf_seq = pairs
        except (TypeError, ValueError):
            raise HistogramError("target 二元组必须形如 (取值序列, CDF 序列)") from None
        xs = _prepare(xs_seq, "target 取值")[0]
        cs = _prepare(cdf_seq, "target CDF")[0]
        if len(xs) != len(cs):
            raise HistogramError("target 取值与 CDF 长度不一致")
        for i in range(1, len(cs)):
            if float(cs[i]) < float(cs[i - 1]) - 1e-12:
                raise HistogramError("target CDF 必须单调不减")
        # 允许用户乱序给出 (取值, CDF) 对：按 CDF 升序重排，取值随之。
        order = sorted(range(len(cs)), key=lambda i: float(cs[i]))
        self.xs = [float(xs[i]) for i in order]
        self.cdf = [float(cs[i]) for i in order]

    def at(self, p):
        """目标分位函数 Q(p)：把 (取值, CDF) 视为分段线性 CDF 求其反解。"""
        xs = self.xs
        cdf = self.cdf
        if p <= cdf[0]:
            return xs[0]
        if p >= cdf[-1]:
            return xs[-1]
        # 找到满足 cdf[j-1] < p <= cdf[j] 的第一个 j。
        j = bisect_left(cdf, p)
        if j <= 0:
            return xs[0]
        lo_p = cdf[j - 1]
        hi_p = cdf[j]
        if hi_p <= lo_p:
            return xs[j]
        frac = (p - lo_p) / (hi_p - lo_p)
        return xs[j - 1] + frac * (xs[j] - xs[j - 1])

    def array_at(self, ps):
        return [self.at(float(p)) for p in ps]


class _CallableTarget:
    """把可调用对象当作目标分位函数 Q(p) 使用。"""

    __slots__ = ("fn",)

    def __init__(self, fn):
        self.fn = fn

    def at(self, p):
        return float(self.fn(p))


def _target_spec(target):
    """把 target 归一化为带 ``at(p)`` 的对象；样本目标另返回原始对象。"""
    if callable(target) and not _is_sequence(target):
        return _CallableTarget(target)
    if hasattr(target, "quantile") and not _is_sequence(target):
        return _CallableTarget(target.quantile)
    if isinstance(target, tuple) and len(target) == 2 and all(
        _is_sequence(part) for part in target
    ):
        return _PairsTarget(target)
    return _SampleTarget(_prepare(target, "target")[0])


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------
class HistMap:
    """直方图分位映射、分位反解与规定化器。

    可直接传入 ``values``（拟合源分布样本）与 ``target``（规定化目标），
    之后省略相应参数；也可以无状态使用，每次调用显式传入。

    实例在构造后只保存不可变的样本与排序数组，公开查询方法不写共享状态，
    因此可被多个线程并发调用。构造参数在首次使用时才校验，因此
    ``HistMap([])`` 本身不报错，而随后的 ``cdf()`` / ``quantile()`` 会抛出
    可区分的 :class:`EmptyHistogramError`。
    """

    __slots__ = (
        "_fitted_raw",
        "_fitted",
        "_fitted_sorted",
        "_fitted_shape",
        "_target_raw",
        "_target_spec",
    )

    def __init__(self, values=None, target=None):
        self._fitted_raw = values
        self._fitted = None
        self._fitted_sorted = None
        self._fitted_shape = (0,)
        self._target_raw = target
        self._target_spec = None

    # -- 拟合 -----------------------------------------------------------
    def fit(self, values):
        """用源分布样本拟合（立即校验），返回自身以便链式调用。"""
        flat, shape = _prepare(values, "values")
        ordered = _sorted(flat)
        # 先发布排序数组，最后发布 ``_fitted`` 作为提交点：并发调用
        # ``_ensure_fitted`` 的线程一旦看到 ``_fitted`` 非空，``_fitted_sorted``
        # 必然已经就绪（单次引用赋值在 CPython 下是原子的）。
        self._fitted_raw = values
        self._fitted_shape = shape
        self._fitted_sorted = ordered
        self._fitted = flat
        return self

    def fit_target(self, target):
        """用目标分布拟合规定化目标（立即校验），返回自身。"""
        self._target_raw = target
        self._target_spec = _target_spec(target)
        return self

    def _ensure_fitted(self):
        if self._fitted is None and self._fitted_raw is not None:
            self.fit(self._fitted_raw)
        return self._fitted

    def _ensure_target(self):
        if self._target_spec is None and self._target_raw is not None:
            self._target_spec = _target_spec(self._target_raw)
        return self._target_spec

    # -- 内部工具 -------------------------------------------------------
    @staticmethod
    def _prob(p):
        if not _is_number(p):
            raise InvalidQuantileError("分位概率必须是 [0, 1] 内的实数: %r" % (p,))
        q = float(p)
        if not _isfinite(q):
            raise NonFiniteError("分位概率必须有限，不能为 NaN/Inf: %r" % (q,))
        if q < 0.0 or q > 1.0:
            raise InvalidQuantileError("分位概率越界，必须在 [0, 1] 内: %r" % (q,))
        return q

    def _sample(self, values, what="values"):
        """返回 ``(一维样本, 形状, 升序样本)``；None 表示使用拟合样本。"""
        if values is None:
            fitted = self._ensure_fitted()
            if fitted is None:
                raise EmptyHistogramError("空直方图：没有可用样本")
            return fitted, self._fitted_shape, self._fitted_sorted
        flat, shape = _prepare(values, what)
        return flat, shape, _sorted(flat)

    # -- 经验 CDF -------------------------------------------------------
    def cdf(self, values=None):
        """经验 CDF：对每个样本返回 ``#(<= x) / n``，与排序结果完全一致。

        无参数时使用 ``fit`` / 构造参数提供的样本。返回与输入同结构的原生
        ``list``（二维输入返回嵌套 list）。空输入抛
        :class:`EmptyHistogramError`，非有限值抛 :class:`NonFiniteError`。

        需要查询「拟合分布在任意查询点处的 CDF」时用 :meth:`cdf_at`。
        """
        flat, shape, ordered = self._sample(values)
        return self._cdf_of(ordered, flat, shape)

    def cdf_at(self, queries, values=None):
        """拟合分布（或给定样本）在查询点处的 CDF 值 ``F(queries)``。

        ``values`` 省略时使用拟合样本；否则以给定样本作为分布。返回与
        ``queries`` 同结构的原生 list。
        """
        if values is None:
            _flat, _shape, ordered = self._sample(None)
        else:
            _flat, _shape, ordered = self._sample(values)
        query, qshape = _prepare(queries, "queries")
        return self._cdf_of(ordered, query, qshape)

    @staticmethod
    def _cdf_of(ordered, query, shape):
        if _np is not None and isinstance(query, _np.ndarray):
            n = ordered.size if isinstance(ordered, _np.ndarray) else len(ordered)
            counts = _np.searchsorted(_np.asarray(ordered), query, side="right")
            return _reshape(counts / n, shape)
        n = len(ordered)
        counts = list(map(bisect_right, repeat(ordered), query))
        return _reshape(list(map(truediv, counts, repeat(n))), shape)

    # -- 分位反解 -------------------------------------------------------
    def quantile(self, p=None, values=None, method="nearest_rank"):
        """分位反解 ``Q(p) = inf{x : F(x) >= p}``（默认 nearest-rank）。

        - ``quantile(p)``：对拟合样本求分位；
        - ``quantile(p, values)``：对显式样本求分位；
        - ``quantile(values, p)``：兼容参数顺序（标量 p）；
        - ``quantile([p1, p2, ...], values)``：对一组概率批量求分位，返回 list。

        ``p`` 必须在 ``[0, 1]`` 内。默认返回值恒为排序数组中的一个元素，
        与排序对照误差为 0（<= 1e-9）；``p=0`` 返回最小值，``p=1`` 返回
        最大值。``method`` 另可选 ``"lower"``、``"higher"``、``"midpoint"``、
        ``"linear"``（与 numpy 的 ``quantile`` 同名取法逐位一致）。
        """
        if p is None:
            raise InvalidQuantileError("缺少分位概率 p")
        # 兼容 quantile(values, p)：仅当 p 不是数值、values 恰为数值时交换；
        # 若两者都不是数值（如两个序列）则按「概率向量 + 样本」解释。
        if values is not None and not _is_number(p) and _is_number(values):
            p, values = values, p
        if _is_sequence(p) and not isinstance(p, (str, bytes)):
            probs = [self._prob(item) for item in p]
            _flat, _shape, ordered = self._sample(values)
            return [self._quantile_from_sorted(ordered, q, method) for q in probs]
        q = self._prob(p)
        flat, _shape, ordered = self._sample(values)
        return self._quantile_from_sorted(ordered, q, method)

    @staticmethod
    def _quantile_from_sorted(ordered, q, method="nearest_rank"):
        """对升序样本求分位。

        默认 ``"nearest_rank"``（等价于 numpy 的 ``"inverted_cdf"``，也是
        ``F(x) >= p`` 的广义逆）；其余取法与 numpy 的 ``quantile`` 在同名
        ``method`` 下逐位一致（``library`` 参数为 ``"linear"``）：以
        ``pos = q*(n-1)``、``i = floor(pos)``、``j = ceil(pos)`` 定义。
        """
        n = ordered.size if (_np is not None and isinstance(ordered, _np.ndarray)) else len(ordered)
        if method in (None, "nearest_rank", "nearest-rank", "inverse_cdf", "inverse-cdf", "inverted_cdf", "inverted-cdf"):
            # 广义逆：最小 x 使 F(x) >= q，即 s[ceil(q*n) - 1]（下标钳制）。
            idx = math.ceil(q * n) - 1
            if idx < 0:
                idx = 0
            elif idx >= n:
                idx = n - 1
            return float(ordered[idx])
        if n == 1:
            return float(ordered[0])
        pos = q * (n - 1)
        i = int(math.floor(pos))
        if i >= n - 1:
            i = n - 1
        j = i if pos == i else i + 1
        if j >= n:
            j = n - 1
        lo = float(ordered[i])
        hi = float(ordered[j])
        if method == "lower":
            return lo
        if method == "higher":
            return hi
        if method == "midpoint":
            return (lo + hi) / 2.0
        if method == "linear":
            frac = pos - i
            return lo + frac * (hi - lo)
        raise InvalidQuantileError("未知的分位取法 method=%r" % (method,))

    def ppf(self, q, values=None, method="nearest_rank"):
        """``quantile`` 的别名（百分位函数）。"""
        return self.quantile(q, values, method)

    # -- 分位映射 -------------------------------------------------------
    def map(self, values, q=None, method="nearest_rank", target=None):
        """分位映射 / 规定化，按第二参数形态分派：

        - ``map(values, q)``（q 为标量）：分位映射 ``Q_values(q)``，即
          ``quantile(q, values)``；关于 q 单调不减，``q=0`` 取最小值、
          ``q=1`` 取最大值。对归一化数据（含 0 与 1）端点即 0→0、1→1。
        - ``map(source, target)`` / ``map(source, target=...)``（第二参数为
          序列、分位函数或带 ``quantile`` 的对象）：直接做规定化，等价于
          ``specify(source, target)``。
        - ``map(values)``（单参数、values 为序列）：值→归一化秩映射
          ``(rank - 1) / (n - 1)``，关于值单调不减，最小值→0、最大值→1。
          单参数标量在拟合模式下等价于 ``quantile(value)``。

        需要一次求多个概率的分位时请用 ``quantile([p1, p2, ...], values)``。
        非法 q 抛 :class:`InvalidQuantileError`（ValueError 子类）。
        """
        if target is not None:
            return self.specify(values, target)
        if q is None:
            if _is_number(values):
                return self.quantile(values, method=method)
            return self._rank_map(values)
        if _is_sequence(q) or callable(q) or hasattr(q, "quantile"):
            return self.specify(values, q)
        return self.quantile(q, values, method)

    def _rank_map(self, values):
        """值→归一化秩 ``(rank - 1)/(n - 1)``（rank 为右秩 / 计数秩）。"""
        flat, shape, ordered = self._sample(values)
        n = flat.size if (_np is not None and isinstance(flat, _np.ndarray)) else len(flat)
        if n <= 1:
            return _reshape([0.0] * n, shape)
        if _np is not None and isinstance(flat, _np.ndarray):
            ranks = _np.searchsorted(_np.asarray(ordered), flat, side="right")
            return _reshape((ranks - 1) / (n - 1), shape)
        counts = map(bisect_right, repeat(ordered), flat)
        return _reshape(list(map(truediv, map(int.__sub__, counts, repeat(1)), repeat(n - 1))), shape)

    # -- 直方图规定化 ---------------------------------------------------
    def specify(self, values=None, target=None):
        """到目标 CDF 的规定化：``y = Q_target(F_source(x))``。

        ``values`` / ``target`` 省略时使用 ``fit`` / ``fit_target``（或构造
        参数）提供的样本；若只拟合了一组样本而显式传入了另一组，则把拟合
        样本当作目标、传入样本当作源（即 ``HistMap(target).specify(src)``）。

        返回与输入同结构、顺序一致的 list。映射关于 x 单调不减；源与目标
        为同一多重集时逐位恒等；映射后经验 CDF 与目标 CDF 的偏差不超过
        ``1/n + 1/m``（远小于 0.02）。
        """
        if values is None and target is None:
            spec = self._ensure_target()
            if spec is None:
                raise EmptyHistogramError("规定化缺少目标分布（target）")
            src, shape, src_sorted = self._sample(None)
            return self._apply_spec(src, shape, src_sorted, spec)

        src, shape, src_sorted = self._sample(values)

        if target is None:
            spec = self._ensure_target()
            if spec is None:
                if values is not None and self._ensure_fitted() is not None:
                    # HistMap(target).specify(src)：构造样本作为目标分布。
                    spec = _target_spec(self._fitted_raw)
                else:
                    raise EmptyHistogramError("规定化缺少目标分布（target）")
        else:
            spec = _target_spec(target)
        return self._apply_spec(src, shape, src_sorted, spec)

    @staticmethod
    def _apply_spec(src, shape, src_sorted, spec):
        n = src.size if (_np is not None and isinstance(src, _np.ndarray)) else len(src)

        if isinstance(spec, _SampleTarget):
            tgt_sorted = spec.sorted
            m = spec.m
            if _np is not None and isinstance(src, _np.ndarray):
                ordered = _np.asarray(src_sorted)
                ranks = _np.searchsorted(ordered, src, side="right").astype(_np.int64)
                # ceil(rank * m / n) - 1：纯整数运算，恒等映射逐位精确
                idx = (ranks * m + (n - 1)) // n - 1
                _np.clip(idx, 0, m - 1, out=idx)
                # 端点精确：源最小值 -> 目标最小值、源最大值 -> 目标最大值。
                # 只在源最小/最大值各为唯一秩组时生效；该调整只移动最低/最高
                # 秩组，组间顺序不变，因此不破坏单调性，CDF 偏差影响 <= 1/n。
                if m > 1 and ordered[0] != ordered[-1]:
                    first_rank = int(_np.searchsorted(ordered, ordered[0], side="right"))
                    last_rank = n - int(_np.searchsorted(ordered, ordered[-1], side="left"))
                    if first_rank == 1:
                        idx[ranks == 1] = 0
                    if last_rank == 1:
                        idx[ranks == n] = m - 1
                return _reshape(_np.asarray(tgt_sorted)[idx], shape)
            tgt_list = list(tgt_sorted)
            br = bisect_right
            out = []
            edge_ok = m > 1 and src_sorted[0] != src_sorted[-1]
            if edge_ok:
                first_rank = br(src_sorted, src_sorted[0])
                last_rank = n - (len(src_sorted) - bisect_left(src_sorted, src_sorted[-1]))
            else:
                first_rank = last_rank = 0
            for v in src:
                rank = br(src_sorted, v)
                if edge_ok and rank == n and last_rank == 1:
                    i = m - 1
                elif edge_ok and rank == 1 and first_rank == 1:
                    i = 0
                else:
                    i = (rank * m + (n - 1)) // n - 1
                    if i < 0:
                        i = 0
                    elif i >= m:
                        i = m - 1
                out.append(tgt_list[i])
            return _reshape(out, shape)

        # 解析式目标（(取值, CDF) 二元组 / 分位函数）：按 p = rank / n 求值。
        if _np is not None and isinstance(src, _np.ndarray):
            ranks = _np.searchsorted(
                _np.asarray(src_sorted), src, side="right"
            )
            ps = (ranks / n).tolist()
        else:
            br = bisect_right
            ps = [br(src_sorted, v) / n for v in src]
        if isinstance(spec, _PairsTarget):
            return _reshape(spec.array_at(ps), shape)
        return _reshape([spec.at(p) for p in ps], shape)

    # -- 便捷入口 -------------------------------------------------------
    def equalize(self, values=None):
        """直方图均衡化：以 ``[0, 1]`` 上的均匀目标做规定化。"""
        return self.specify(values, lambda p: p)


# ---------------------------------------------------------------------------
# 规定化的别名（同一实现，便于不同调用习惯）
# ---------------------------------------------------------------------------
def _targeted(name):
    def method(self, values=None, target=None):
        return self.specify(values, target)

    method.__name__ = name
    method.__qualname__ = "HistMap." + name
    method.__doc__ = "``specify`` 的别名：直方图规定化/匹配。"
    return method


for _alias in (
    "match",
    "hist_match",
    "histogram_match",
    "match_histogram",
    "histogram_matching",
    "histogram_specification",
    "specification",
    "normalize",
    "normalise",
    "normalize_to",
    "histogram_normalize",
    "to_target",
    "cdf_match",
    "match_cdf",
):
    setattr(HistMap, _alias, _targeted(_alias))
del _alias


# ---------------------------------------------------------------------------
# 模块级便捷函数
# ---------------------------------------------------------------------------
def cdf(values):
    """``HistMap().cdf(values)`` 的模块级快捷方式。"""
    return HistMap().cdf(values)


def quantile(p, values=None, method="nearest_rank"):
    """``HistMap().quantile(p, values)`` 的模块级快捷方式。

    也接受 ``quantile(values, p)`` 参数顺序。
    """
    return HistMap().quantile(p, values, method)


def specify(values, target):
    """``HistMap().specify(values, target)`` 的模块级快捷方式。"""
    return HistMap().specify(values, target)


def equalize(values):
    """``HistMap().equalize(values)`` 的模块级快捷方式。"""
    return HistMap().equalize(values)


#: 规定化的常见别名。
match = specify
hist_match = specify
histogram_match = specify
match_histogram = specify
histogram_matching = specify
histogram_specification = specify
specification = specify
normalize = specify
normalise = specify
normalize_to = specify
histogram_normalize = specify
to_target = specify
cdf_match = specify
match_cdf = specify
