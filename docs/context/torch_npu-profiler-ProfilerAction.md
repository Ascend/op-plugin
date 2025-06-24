# torch_npu.profiler.ProfilerAction

## 函数原型

```
torch_npu.profiler.ProfilerAction
```

## 功能说明

Profiler状态，Enum类型。

## 参数说明

- NONE：无任何行为。
- WARMUP：性能数据采集预热。
- RECORD：性能数据采集。
- RECORD_AND_SAVE：性能数据采集并保存。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu

...

torch_npu.profiler.ProfilerAction.WARMUP
with torch_npu.profiler.profile(
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
    ) as prof:
            for step in range(steps): # 训练函数
                train_one_step() # 训练函数
                prof.step()
```

