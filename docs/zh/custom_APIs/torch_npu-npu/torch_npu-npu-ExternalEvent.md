# torch_npu.npu.ExternalEvent

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id7 -->

## 功能说明

ExternalEvent是AscendCL Event的封装。NPUGraph场景在执行图捕获时，ExternalEvent会被作为图外部节点被捕获，用于非图内时序控制场景。

## 函数原型

```python
torch_npu.npu.ExternalEvent()
```

## 返回值说明

返回创建好的ExternalEvent对象，用于下发Event相关任务。

## 约束说明

ExternalEvent创建时，系统内部会在Device上分配32字节的内存，创建数量受芯片硬件规格限制。

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
