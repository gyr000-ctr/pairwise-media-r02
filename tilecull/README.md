# tilecull

## 题面（可直接粘贴）

瓦片视锥剔除。约束：结果必须与逐瓦片投影+包含判定一致（集合相等）；边界瓦片归属确定；相机在瓦片内时不得漏；10 万瓦片不超过 300 ms；非法相机参数报错；并发安全。验收：cd tilecull && python -m pytest tests -q -k cull。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd tilecull && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
