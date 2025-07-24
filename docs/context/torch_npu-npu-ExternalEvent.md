# torch_npu.npu.ExternalEvent
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

ExternalEvent是AscendCL Event的封装。NPUGraph场景在执行图捕获时，ExternalEvent会被作为图外部节点被捕获，用于控制非图内时序控制场景。

## 函数原型

```
torch_npu.npu.ExternalEvent()
```

## 返回值说明
返回创建好的ExternalEvent对象，用于下发Event相关任务。

## 约束说明

 ExternalEvent创建时，系统内部会申请Event资源，创建数量受芯片硬件规格限制。

## 调用示例
```python
import torch
import torch_npu

torch.npu.set_device(0)

event = torch_npu.npu.ExternalEvent()
default_stream = torch_npu.npu.current_stream()
stream = torch.npu.Stream()

event.wait(default_stream)
event.reset(default_stream)
event.record(stream)
```
