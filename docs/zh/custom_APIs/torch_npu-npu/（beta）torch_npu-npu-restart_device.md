# （beta）torch_npu.npu.restart_device

> [!NOTICE]  
> 本接口为预留接口，暂不支持。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>  |   √   |
|<term>Atlas A2 训练系列产品</term>  |   √   |

## 功能说明

恢复对应device上的状态，后续在此device上可继续进行计算。

## 函数原型

```python
torch_npu.npu.restart_device(device_id, rebuild_all_resources=False, disable_tensor_unsafe_check=False) -> None
```

## 参数说明

- **device_id** (`int`)：必选参数，需要处理的device ID。
- **rebuild_all_resources** (`bool`)：可选参数，是否重建该device上的资源，默认为False。设置为True时，会恢复该device上的所有NPU stream，并联动`disable_tensor_unsafe_check`决定是否进行数据标脏。
- **disable_tensor_unsafe_check** (`bool`)：可选参数，是否禁用数据标脏，默认为False。仅在`rebuild_all_resources=True`时生效。为False时将该device上的所有NPU tensor进行标脏；为True时跳过标脏。

## 约束说明

要确保是一个有效的device，这个device可以是被stop过的也可以是没有被stop。

## 调用示例

 ```python
>>> import torch
>>> import torch_npu  
>>> torch.npu.set_device(0) 
>>> torch_npu.npu.stop_device(0)
>>> torch_npu.npu.restart_device(0)
 ```
