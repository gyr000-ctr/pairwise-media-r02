# polybool

## 题面（可直接粘贴）

多边形包含与面积。约束：包含判定用射线法且与网格采样一致（差 ≤1%）；面积用鞋带公式；自交多边形报错；凹多边形正确；10 万次包含判定不超过 400 ms；非法多边形报错。验收：cd polybool && python -m pytest tests -q -k poly。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd polybool && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
