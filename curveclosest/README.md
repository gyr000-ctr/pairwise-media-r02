# curveclosest

## 题面（可直接粘贴）

贝塞尔最近点反解。约束：返回值 t 使距离与暴力采样（1e4 点）差 ≤1e-6；t 在 [0,1]；多局部极小取全局最小；退化曲线报错；1 万次反解不超过 400 ms；并发安全。验收：cd curveclosest && python -m pytest tests -q -k closest。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd curveclosest && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
