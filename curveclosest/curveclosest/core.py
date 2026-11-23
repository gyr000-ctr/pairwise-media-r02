"""贝塞尔最近点反解核心 API。

算法：
  对三次贝塞尔曲线 C(t) 与查询点 P，距离平方 f(t)=|C(t)-P|^2 的驻点满足
  f'(t) = 2 (C(t)-P)·C'(t) = 0。这是一个 5 次多项式。

  1. 将 f' 写成 5 次 Bernstein 形式（卷积公式，无需幂基转换）；
  2. 利用 Bernstein 多项式的变差缩减性质，用 de Casteljau 细分做根隔离，
     数学上保证 [0,1] 内的每一个实根都不会被遗漏（即不会漏掉任何局部极小盆）；
  3. 对每个含单根的隔离区间用夹逼 Newton（失败时回退二分）精化到机器精度；
  4. 比较全部驻点与两个端点的距离平方，取全局最小。

实现为无状态纯函数：不持有任何可变状态，因此天然线程安全、结果确定。
仅依赖标准库 math。
"""

import math


def _valid_point(q):
    """把控制点/查询点规整为 (float, float)；畸形或非有限输入抛 ValueError。"""
    try:
        x = float(q[0])
        y = float(q[1])
    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
        raise ValueError("输入点必须是含两个数值分量的序列")
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError("输入包含非有限数值 (nan/inf)")
    return x, y


class Closest:
    def t_of(self, p0, p1, p2, p3, point):
        """返回使曲线到 point 距离最小的参数 t ∈ [0,1]（float）。

        退化（四点重合的点曲线）或非法/非有限输入抛 ValueError。
        """
        x0, y0 = _valid_point(p0)
        x1, y1 = _valid_point(p1)
        x2, y2 = _valid_point(p2)
        x3, y3 = _valid_point(p3)
        px, py = _valid_point(point)

        # 退化曲线：四个控制点完全重合 => 曲线退化为单点，参数无定义。
        if (x0, y0) == (x1, y1) == (x2, y2) == (x3, y3):
            raise ValueError("退化贝塞尔曲线：四个控制点重合")

        # C(t)-P 的 Bernstein 系数（3 次）
        a0, a1, a2, a3 = x0 - px, x1 - px, x2 - px, x3 - px
        b0, b1, b2, b3 = y0 - py, y1 - py, y2 - py, y3 - py

        # C'(t)/3 的 Bernstein 系数（2 次）
        d0x, d1x, d2x = x1 - x0, x2 - x1, x3 - x2
        d0y, d1y, d2y = y1 - y0, y2 - y1, y3 - y2

        # f'(t)/6 在 5 次 Bernstein 基下的系数 c_k，k=0..5。
        # 由 (C-P)·C' = 3 * sum_k c_k B_{5,k}，常因子 3 不影响求根。
        # 公式：c_k = (1/C(5,k)) * 对所有 i+j=k 的项
        #   C(3,i) C(2,j) (a_i d_j_x + b_i d_j_y)
        c = [
            a0 * d0x + b0 * d0y,
            (
                2.0 * (a0 * d1x + b0 * d1y)
                + 3.0 * (a1 * d0x + b1 * d0y)
            ) / 5.0,
            (
                (a0 * d2x + b0 * d2y)
                + 6.0 * (a1 * d1x + b1 * d1y)
                + 3.0 * (a2 * d0x + b2 * d0y)
            ) / 10.0,
            (
                3.0 * (a1 * d2x + b1 * d2y)
                + 6.0 * (a2 * d1x + b2 * d1y)
                + (a3 * d0x + b3 * d0y)
            ) / 10.0,
            (
                3.0 * (a2 * d2x + b2 * d2y)
                + 2.0 * (a3 * d1x + b3 * d1y)
            ) / 5.0,
            a3 * d2x + b3 * d2y,
        ]

        # 曲线求值所需的差分（Horner，幂基）
        u3 = -x0 + 3.0 * x1 - 3.0 * x2 + x3
        u2 = 3.0 * x0 - 6.0 * x1 + 3.0 * x2
        u1 = -3.0 * x0 + 3.0 * x1
        v3 = -y0 + 3.0 * y1 - 3.0 * y2 + y3
        v2 = 3.0 * y0 - 6.0 * y1 + 3.0 * y2
        v1 = -3.0 * y0 + 3.0 * y1

        def curve(t):
            mt = 1.0 - t
            b0c = mt * mt * mt
            b1c = 3.0 * mt * mt * t
            b2c = 3.0 * mt * t * t
            b3c = t * t * t
            return (
                b0c * x0 + b1c * x1 + b2c * x2 + b3c * x3,
                b0c * y0 + b1c * y1 + b2c * y2 + b3c * y3,
            )

        def fun(t):
            """距离平方 f(t)。"""
            x, y = curve(t)
            dx = x - px
            dy = y - py
            return dx * dx + dy * dy

        def dfun(t):
            """f'(t) = 2(C-P)·C'。"""
            t2 = t * t
            cx = u3 * t2 * t + u2 * t2 + u1 * t + x0
            cy = v3 * t2 * t + v2 * t2 + v1 * t + y0
            cpx = 3.0 * u3 * t2 + 2.0 * u2 * t + u1
            cpy = 3.0 * v3 * t2 + 2.0 * v2 * t + v1
            return 2.0 * ((cx - px) * cpx + (cy - py) * cpy)

        def d2fun(t):
            """f''(t) = 2(C'·C' + (C-P)·C'')。"""
            t2 = t * t
            cx = u3 * t2 * t + u2 * t2 + u1 * t + x0
            cy = v3 * t2 * t + v2 * t2 + v1 * t + y0
            cpx = 3.0 * u3 * t2 + 2.0 * u2 * t + u1
            cpy = 3.0 * v3 * t2 + 2.0 * v2 * t + v1
            cpx2 = 6.0 * u3 * t + 2.0 * u2
            cpy2 = 6.0 * v3 * t + 2.0 * v2
            return 2.0 * (
                cpx * cpx + cpy * cpy
                + (cx - px) * cpx2 + (cy - py) * cpy2
            )

        def variations(coeffs):
            """Bernstein 系数去掉零后的符号变化次数。"""
            prev = 0.0
            n = 0
            for z in coeffs:
                if z > 0.0:
                    if prev < 0.0:
                        n += 1
                    prev = 1.0
                elif z < 0.0:
                    if prev > 0.0:
                        n += 1
                    prev = -1.0
            return n

        def refine(a, b):
            """对已知位于 [a,b] 的根做夹逼 Newton（解析二阶导），回退二分。

            注意：内部变量不得与外层细分循环的 lo/hi 同名，否则闭包赋值会
            污染主循环的区间变量（Python 作用域规则）。
            """
            fa = dfun(a)
            fb = dfun(b)
            if fa == 0.0:
                return a
            if fb == 0.0:
                return b
            rt = 0.5 * (a + b)
            for _ in range(60):
                fr = dfun(rt)
                if fr == 0.0:
                    return rt
                # 维护夹逼括号：同号则替换对应端点
                if (fr > 0.0) == (fa > 0.0):
                    a = rt
                    fa = fr
                else:
                    b = rt
                    fb = fr
                slope = d2fun(rt)
                if slope != 0.0 and math.isfinite(slope):
                    nr = rt - fr / slope
                    if math.isfinite(nr) and a < nr < b:
                        rt_new = nr
                    else:
                        rt_new = 0.5 * (a + b)
                else:
                    rt_new = 0.5 * (a + b)
                if abs(rt_new - rt) <= 1e-15 * max(1.0, abs(rt)):
                    return rt_new
                rt = rt_new
            return rt

        # ---- 根隔离：栈式 de Casteljau 细分（变差缩减保证不漏根）----
        roots = []
        # 栈元素：(区间左, 区间右, Bernstein 系数, 深度)
        stack = [(0.0, 1.0, c, 0)]
        while stack:
            lo, hi, coeffs, depth = stack.pop()
            v = variations(coeffs)
            if v == 0:
                continue
            if v == 1 or depth >= 48:
                roots.append(refine(lo, hi))
                continue
            mid = 0.5 * (lo + hi)
            # de Casteljau 在 t=0.5 处细分（原地更新）
            q = list(coeffs)
            left = [q[0]]
            n = len(q)
            for k in range(1, n):
                nq = n - k
                for j in range(nq):
                    q[j] = 0.5 * (q[j] + q[j + 1])
                left.append(q[0])
            # 原地更新后 q[0..m] 正序即右段控制点
            # （q[j] = q^(m-j)_j），left 为各轮首个点。
            right = list(q)
            # 根恰落在细分点：两段公共端点系数为 0，会被变差计数跳过，
            # 必须显式登记，否则漏根。
            if left[-1] == 0.0:
                roots.append(mid)
            stack.append((lo, mid, left, depth + 1))
            stack.append((mid, hi, right, depth + 1))

        # ---- 全局最优：端点 + 全部驻点 ----
        best_t = 0.0
        best_f = fun(0.0)
        f1 = fun(1.0)
        if f1 < best_f:
            best_f = f1
            best_t = 1.0
        for t in roots:
            if t < 0.0 or t > 1.0:
                continue
            ft = fun(t)
            if ft < best_f:
                best_f = ft
                best_t = t

        return float(best_t)

    def distance(self, c, point):
        """曲线到 point 的欧氏距离（非负 float）。"""
        try:
            pts = list(c)
        except TypeError:
            raise ValueError("曲线控制点必须是含 4 个点的序列")
        if len(pts) != 4:
            raise ValueError("三次贝塞尔曲线需要 4 个控制点")
        t = self.t_of(pts[0], pts[1], pts[2], pts[3], point)
        x0, y0 = _valid_point(pts[0])
        x1, y1 = _valid_point(pts[1])
        x2, y2 = _valid_point(pts[2])
        x3, y3 = _valid_point(pts[3])
        px, py = _valid_point(point)
        mt = 1.0 - t
        b0 = mt * mt * mt
        b1 = 3.0 * mt * mt * t
        b2 = 3.0 * mt * t * t
        b3 = t * t * t
        x = b0 * x0 + b1 * x1 + b2 * x2 + b3 * x3
        y = b0 * y0 + b1 * y1 + b2 * y2 + b3 * y3
        return math.hypot(x - px, y - py)
