# audioms

## 题面（可直接粘贴）

M/S 立体声处理。约束：M/S 变换可逆（往返误差 ≤1e-9）；宽度参数 0 时左右相等且总能量不变；宽度放大后单声道下混能量差 ≤0.5 dB；跨缓冲区状态连续；10 万样本不超过 200 ms；非法宽度报错。验收：cd audioms && python -m pytest tests -q -k ms。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd audioms && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
