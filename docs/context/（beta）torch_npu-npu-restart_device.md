# （beta）torch_npu.npu.restart_device

>**须知：**<br>
>本接口为预留接口，暂不支持。

## 函数原型

```
torch_npu.npu.restart_device(device_id: int, rebuild_all_resource: bool = False) -> None
```

## 功能说明

恢复对应device上的状态，后续在此device上可以继续进行计算执行。

## 参数说明

- “device_id”(int)需要处理的device id。
- “rebuild_all_resource”(bool)，是一个保留参数，在当前接口中没有实际的功能。

## 输入说明

要确保是一个有效的device，这个device可以是被stop过的也可以是没有被stop。

## 支持的型号

- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>

## 调用示例

```python
import torch
import torch_npu  
torch.npu.set_device(0) 
torch_npu.npu.stop_device(0)
torch_npu.npu.restart_device(0)
```

