# torch_npu.profiler.ProfilerAction

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Ascend 950DT</term> |  √  |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

Profiler采集行为状态，枚举类型。由`torch_npu.profiler.schedule`在每个step返回，用于控制`torch_npu.profiler.profile`在该step执行的采集行为（无操作、预热、采集、采集并保存）。

## 类签名

```python
class ProfilerAction(Enum)
```

## 成员说明

> 所有枚举值均为只读，不可在运行时修改。

| 成员名 | 值 | 描述 |
| :--- | :--- | :--- |
| `NONE` | `0` | 无任何行为。 |
| `WARMUP` | `1` | 性能数据采集预热。 |
| `RECORD` | `2` | 性能数据采集。 |
| `RECORD_AND_SAVE` | `3` | 性能数据采集并保存。 |

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

```python
import torch
import torch_npu

...
with torch_npu.profiler.profile(
    schedule=torch_npu.profiler.ProfilerAction.RECORD,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    ) as prof:
            for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```
