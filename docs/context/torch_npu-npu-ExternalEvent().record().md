# torch_npu.npu.ExternalEvent().record()
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

在指定stream上记录Event事件。本接口被调用时，会捕获当前Stream上已下发的任务，并记录到Event事件中，因此后续若调用wait接口，会等待该Event事件中所捕获的任务都已经完成。

## 函数原型

```
torch_npu.npu.ExternalEvent().record(stream) -> None
```

## 参数说明

 **stream** (`torch_npu.npu.Stream`)：必选参数，指定用于下发Event事件记录任务的流。

## 返回值说明

无

## 约束说明

- 该接口是异步接口，调用接口成功仅表示任务下发成功，不表示任务执行成功。
- 接口调用顺序：torch_npu.npu.ExternalEvent().wait()-->torch_npu.npu.ExternalEvent().reset()-->torch_npu.npu.ExternalEvent().record()或torch_npu.npu.ExternalEvent().record()-->torch_npu.npu.ExternalEvent().wait()-->torch_npu.npu.ExternalEvent().reset()。

## 调用示例
```python
import torch
import torch_npu

torch.npu.set_device(0)

event = torch.npu.ExternalEvent()
default_stream = torch_npu.npu.current_stream()
stream = torch.npu.Stream()

event.wait(default_stream)
event.reset(default_stream)
event.record(stream)
```
