# torch_npu.profiler.ProfilerActivity

## 产品支持情况

| 产品                               | 是否支持 |
| ---------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品</term> |    √     |
| <term>Atlas 训练系列产品</term>    |    √     |

## 功能说明

事件采集列表，Enum类型。用于赋值给torch_npu.profiler.profile的activities参数。

## 函数原型

```
torch_npu.profiler.ProfilerActivity
```

## 参数说明

- **torch_npu.profiler.ProfilerActivity.CPU**：可选参数，框架侧数据采集的开关。
- **torch_npu.profiler.ProfilerActivity.NPU**：可选参数，CANN软件栈及NPU数据采集的开关。

默认情况下两个开关同时开启。

## 返回值说明

无

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝编译运行，仅供参考。

```python
import torch
import torch_npu

...

with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
            ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result")
) as prof:
        for step in range(steps): # 训练函数
            train_one_step() # 训练函数
            prof.step()
```