# torch_npu.npu.ExternalEvent().wait()

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id7 -->

## 功能说明

阻塞指定Stream的运行，直到指定的Event完成，仅支持单个Stream等待单个Event的场景。

## 函数原型

```python
torch_npu.npu.ExternalEvent().wait(stream) -> None
```

## 参数说明

**stream** (`torch_npu.npu.Stream`)：必选参数，指定用于下发Event事件阻塞任务的流。

## 返回值说明

无

## 约束说明

- 该接口是异步接口，调用接口成功仅表示任务下发成功，不表示任务执行成功。
- 接口调用顺序：torch_npu.npu.ExternalEvent().wait()-->torch_npu.npu.ExternalEvent().record()或torch_npu.npu.ExternalEvent().record()-->torch_npu.npu.ExternalEvent().wait()。
- 该接口会自动复位Event，不需要调用torch_npu.npu.ExternalEvent().reset()接口手动复位Event。

## 调用示例

```python
import torch
import torch_npu

torch.npu.set_device(0)

event = torch_npu.npu.ExternalEvent()
default_stream = torch_npu.npu.current_stream()
stream = torch.npu.Stream()

event.wait(default_stream)
event.record(stream)
```
