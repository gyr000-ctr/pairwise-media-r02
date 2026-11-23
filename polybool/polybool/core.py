"""polybool 核心 API。

- :meth:`Poly.contains`：射线法（向右水平射线、半开边规则），与网格采样
  判定天然一致，凹多边形同样适用。
- :meth:`Poly.area`：鞋带公式，返回无符号面积。
- 非法输入分类报错：结构/类型错误抛 :class:`TypeError`；顶点不足、坐标
  非有限（NaN/Inf）、自交等抛 :class:`ValueError`。
- 多边形只编译一次并按对象身份缓存（带锁，线程安全），内容指纹层兜住
  对象替换与 id 复用；海量包含判定时每次调用只需付出射线扫描本身的成本。
"""

import math
import threading

__all__ = ["Poly"]

# --------------------------------------------------------------------------
# 输入校验
# --------------------------------------------------------------------------
def _check_finite_pair(seq, name):
    """校验一个二维点，返回 ``(x, y)``（保持原始数值类型）。"""
    if not isinstance(seq, (list, tuple)) or len(seq) != 2:
        raise TypeError("%s 必须是恰好包含两个实数的序列" % name)
    x, y = seq
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError("%s 的坐标必须是实数" % name)
    if isinstance(y, bool) or not isinstance(y, (int, float)):
        raise TypeError("%s 的坐标必须是实数" % name)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("%s 的坐标必须有限，不能为 NaN/Inf" % name)
    return x, y


# --------------------------------------------------------------------------
# 线段相交（用于自交检测，含端点接触与共线重叠）
# --------------------------------------------------------------------------
def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p):
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def _segments_intersect(a, b, c, d):
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    if o4 == 0 and _on_segment(c, d, b):
        return True
    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))


# --------------------------------------------------------------------------
# 编译后的多边形
# --------------------------------------------------------------------------
class _CompiledPolygon:
    """预校验、预展开的不可变多边形表示。

    四条平行元组分别存放每条边的两个端点坐标，配合 ``zip`` 的 C 层迭代
    让射线扫描热循环尽可能短。
    """

    __slots__ = ("x1s", "y1s", "x2s", "y2s", "minx", "miny", "maxx", "maxy", "area2")

    def __init__(self, points):
        n = len(points)
        x1s = [0.0] * n
        y1s = [0.0] * n
        x2s = [0.0] * n
        y2s = [0.0] * n
        minx = math.inf
        miny = math.inf
        maxx = -math.inf
        maxy = -math.inf
        twice = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            x1s[i] = x1
            y1s[i] = y1
            x2s[i] = x2
            y2s[i] = y2
            twice += x1 * y2 - x2 * y1
            if x1 < minx:
                minx = x1
            if x1 > maxx:
                maxx = x1
            if y1 < miny:
                miny = y1
            if y1 > maxy:
                maxy = y1
        self.x1s = tuple(x1s)
        self.y1s = tuple(y1s)
        self.x2s = tuple(x2s)
        self.y2s = tuple(y2s)
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        # 鞋带和（带符号），面积取绝对值
        self.area2 = twice

    def contains(self, px, py):
        """射线法：点向右发射水平射线，半开边规则统计穿越次数。"""
        # 包围盒快速排除
        if px < self.minx or px > self.maxx or py < self.miny or py > self.maxy:
            return False
        inside = False
        # 半开边规则：水平边两端 y 相等，比较结果相同，自然不计入
        for x1, y1, x2, y2 in zip(self.x1s, self.y1s, self.x2s, self.y2s):
            if (y1 > py) != (y2 > py):
                if x1 + (py - y1) * (x2 - x1) / (y2 - y1) > px:
                    inside = not inside
        return inside


def _compile(poly):
    """完整校验并编译多边形。"""
    if not isinstance(poly, (list, tuple)):
        raise TypeError("poly 必须是由顶点组成的序列（list/tuple）")
    if len(poly) < 3:
        raise ValueError("非法多边形：至少需要 3 个顶点")

    points = []
    for vertex in poly:
        x, y = _check_finite_pair(vertex, "多边形顶点")
        points.append((float(x), float(y)))

    # 折叠连续重复顶点（含首尾闭合点），避免零长度边
    cleaned = []
    for p in points:
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        raise ValueError("非法多边形：去重后有效顶点不足 3 个")

    # 自交检测：所有非相邻边对不得相交（含端点接触、共线重叠）
    n = len(cleaned)
    for i in range(n - 1):
        a = cleaned[i]
        b = cleaned[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # 首尾相邻边
            c = cleaned[j]
            d = cleaned[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                raise ValueError("非法多边形：多边形存在自交")

    return _CompiledPolygon(cleaned)


# --------------------------------------------------------------------------
# 对外 API
# --------------------------------------------------------------------------
class Poly:
    """多边形包含判定与面积计算。

    方法本身无状态、可重入；编译结果按对象身份缓存，受锁保护，可在
    多线程下并发使用。同一 ``Poly`` 实例对同一个多边形对象只做一次
    O(n²) 校验，之后的调用只付出射线扫描本身的成本。

    缓存契约：多边形在查询期间应视为只读几何数据。原地修改顶点后，
    请以新的列表对象传入（或新建 ``Poly`` 实例），以触发重新编译；
    内容指纹层保证不同对象即便发生 id 复用也不会拿到陈旧结果。
    """

    _CACHE_MAX = 256

    def __init__(self):
        # 两层缓存：
        #   _by_id ：id(poly) -> (poly, compiled)，热路径仅一次 dict 查找
        #   _by_key：内容指纹 -> (poly, compiled)，兜住 id 复用/对象替换
        # 写入全部在锁内完成；读取依赖 GIL 下 dict 单次取值的原子性。
        self._by_id = {}
        self._by_key = {}
        self._lock = threading.Lock()

    @staticmethod
    def _fingerprint(poly):
        # C 层嵌套元组哈希：覆盖长度、每个顶点及坐标修改
        return hash(tuple(tuple(v) for v in poly))

    def _get_compiled(self, poly):
        if not isinstance(poly, (list, tuple)):
            # 非法类型走完整校验，抛出分类好的 TypeError
            return _compile(poly)
        entry = self._by_id.get(id(poly))
        if entry is not None and entry[0] is poly:
            return entry[1]
        try:
            key = self._fingerprint(poly)
        except TypeError:
            # 顶点结构非法（不可哈希等），走完整校验报分类错误
            return _compile(poly)
        with self._lock:
            entry = self._by_id.get(id(poly))
            if entry is not None and entry[0] is poly:
                return entry[1]
            hit = self._by_key.get(key)
            if hit is not None and hit[0] == poly:
                compiled = hit[1]
            else:
                compiled = _compile(poly)
                self._by_key[key] = (poly, compiled)
                if len(self._by_key) > self._CACHE_MAX:
                    self._by_key.pop(next(iter(self._by_key)))
            self._by_id[id(poly)] = (poly, compiled)
            if len(self._by_id) > self._CACHE_MAX:
                self._by_id.pop(next(iter(self._by_id)))
            return compiled

    def contains(self, poly, point):
        """点是否在多边形内部（射线法，边界按半开边规则确定）。

        返回严格的 :class:`bool`；非法多边形或点抛出
        :class:`TypeError` / :class:`ValueError`。
        """
        compiled = self._get_compiled(poly)
        px, py = _check_finite_pair(point, "point")
        if type(px) is not float:
            px = float(px)
        if type(py) is not float:
            py = float(py)
        return bool(compiled.contains(px, py))

    def area(self, poly):
        """鞋带公式计算无符号面积；非法多边形抛出对应异常。"""
        compiled = self._get_compiled(poly)
        return abs(compiled.area2) * 0.5
