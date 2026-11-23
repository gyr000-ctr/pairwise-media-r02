"""tilecull：瓦片视锥剔除。

坐标约定
--------
瓦片网格覆盖世界坐标 ``[0, w * tile_size) x [0, h * tile_size)``，
瓦片 ``(i, j)``（``0 <= i < w``、``0 <= j < h``）占地
``[i * tile_size, (i + 1) * tile_size) x [j * tile_size, (j + 1) * tile_size)``。

相机参数 ``cam = (x0, y0, x1, y1)`` 表示轴对齐矩形 ``[x0, x1) x [y0, y1)``
（要求 ``x0 < x1``、``y0 < y1`` 且四个值均为有限实数）。

可见性判定：瓦片与相机矩形有正面积交集时可见，即

    tx0 < x1 且 tx1 > x0 且 ty0 < y1 且 ty1 > y0

其中 ``tx1 = tx0 + tile_size``、``ty1 = ty0 + tile_size``。边界归属于
半开区间：仅与相机边界相切的瓦片不计入，因此结果与逐瓦片做同样的
重叠判定得到的集合严格相等，且每个瓦片只可能属于一侧，归属确定。
相机完全落在某个瓦片内部时，该瓦片与相机矩形有正面积交集，不会被漏掉。

``visible`` 是纯函数：不读写任何可变共享状态，返回新列表，可安全并发调用。
"""

import math
from numbers import Real

__all__ = ["TileCull"]


def _as_finite_real(value, name):
    """把单个参数规范化为有限 float，非法输入分类报错。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _parse_cam(cam):
    """校验并解析相机矩形，返回 ``(x0, y0, x1, y1)``。"""
    if isinstance(cam, (str, bytes)):
        raise TypeError("cam must be a sequence of four real numbers")
    try:
        items = list(cam)
    except TypeError:
        raise TypeError(
            f"cam must be an iterable of four real numbers, got {type(cam).__name__}"
        ) from None
    if len(items) != 4:
        raise ValueError(f"cam must contain exactly 4 values, got {len(items)}")
    x0 = _as_finite_real(items[0], "cam[0]")
    y0 = _as_finite_real(items[1], "cam[1]")
    x1 = _as_finite_real(items[2], "cam[2]")
    y1 = _as_finite_real(items[3], "cam[3]")
    if not (x0 < x1 and y0 < y1):
        raise ValueError(
            "cam must be a non-degenerate rectangle with x0 < x1 and y0 < y1"
        )
    return x0, y0, x1, y1


def _parse_count(value, name):
    """校验网格尺寸（正整数，允许整值浮点数）。"""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a positive integer, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v) or v != int(v):
        raise ValueError(f"{name} must be a finite integer, got {value!r}")
    n = int(v)
    if n <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return n


class TileCull:
    """无状态瓦片剔除器；实例可在线程间共享。"""

    __slots__ = ()

    def visible(self, cam, tile_size, w, h):
        """返回与相机矩形相交的瓦片 ``(i, j)`` 列表。

        列表按行优先（先 ``j`` 后 ``i``）排序，结果确定。
        """
        x0, y0, x1, y1 = _parse_cam(cam)
        ts = _as_finite_real(tile_size, "tile_size")
        if ts <= 0.0:
            raise ValueError(f"tile_size must be positive, got {tile_size!r}")
        w = _parse_count(w, "w")
        h = _parse_count(h, "h")

        # 网格覆盖的世界范围；先把相机矩形夹到网格内，避免极端值下的除法溢出。
        # (n - 1) * ts + ts 与 n * ts 可能相差 1 ulp，取较大者作为网格终点，
        # 避免把紧贴末边界的瓦片误判到网格外。
        world_w = max(w * ts, (w - 1) * ts + ts)
        world_h = max(h * ts, (h - 1) * ts + ts)
        if x1 <= 0.0 or y1 <= 0.0 or x0 >= world_w or y0 >= world_h:
            return []

        # 候选索引范围：先按算术上下界收窄，再逐块用精确谓词过滤，
        # 保证与逐瓦片暴力重叠判定逐位一致（含浮点舍入情形）。
        def candidates(lo, hi, n, grid_end):
            # 左侧从 0 起或相机左边界所在列的前一列开始（下界允许宽一列，
            # 由后面的精确谓词过滤）；右侧若相机右边界已到达网格终点，
            # 则直接取到末列，避免除法舍入把末列误排除。
            r0 = max(lo, 0.0) / ts
            c0 = max(0, math.floor(r0) - 1) if math.isfinite(r0) else 0
            if math.isfinite(hi) and hi < grid_end:
                c1 = math.floor(hi / ts) + 1
            else:
                c1 = n - 1
            return c0, min(n - 1, c1)

        i0, i1 = candidates(x0, x1, w, world_w)
        j0, j1 = candidates(y0, y1, h, world_h)
        if i0 > i1 or j0 > j1:
            return []

        vis_i = []
        for i in range(i0, i1 + 1):
            tx0 = i * ts
            if tx0 < x1 and tx0 + ts > x0:
                vis_i.append(i)
        if not vis_i:
            return []

        out = []
        append = out.append
        for j in range(j0, j1 + 1):
            ty0 = j * ts
            if ty0 < y1 and ty0 + ts > y0:
                for i in vis_i:
                    append((i, j))
        return out
