# torch_npu.profiler.ProfilerAction

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

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
