# torch_npu.profiler.ProfilerActivity

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

事件采集列表，Enum类型。用于赋值给torch_npu.profiler.profile的activities参数。

## 类签名

```python
torch_npu.profiler.ProfilerActivity
```

## 成员说明

- **torch_npu.profiler.ProfilerActivity.CPU**：枚举成员，用于开启框架侧数据采集。
- **torch_npu.profiler.ProfilerActivity.NPU**：枚举成员，用于开启CANN软件栈及NPU数据采集。

## 调用示例

以下是关键步骤的代码示例，不可直接拷贝运行，仅供参考。

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
