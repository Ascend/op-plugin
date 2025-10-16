# torch_npu.profiler.ProfilerAction

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

用于控制Profiler行为状态，如性能数据采集、性能数据采集预热、性能数据采集并保存，Enum类型。

## 函数原型

```
torch_npu.profiler.ProfilerAction
```

## 参数说明

- **torch_npu.profiler.ProfilerAction.NONE**：可选参数，无任何行为。
- **torch_npu.profiler.ProfilerAction.WARMUP**：可选参数，性能数据采集预热。
- **torch_npu.profiler.ProfilerAction.RECORD**：可选参数，性能数据采集。
- **torch_npu.profiler.ProfilerAction.RECORD_AND_SAVE**：可选参数，性能数据采集并保存。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

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