# （beta）torch_npu.npu.stop_device

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。<br>
> 本接口为预留接口，暂不支持。

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

## 功能说明

停止对应device上的计算，对于没有执行的计算进行清除，后续在此device上执行计算会报错。

## 函数原型

```python
torch_npu.npu.stop_device(device_id: int) -> int 
```

## 参数说明

**device_id**（`int`）：需要处理的device id，确保是一个有效的device。

## 返回值说明

`int`

返回值为`int`，代表执行结果，0表示执行成功，1表示执行失败。

## 调用示例

```python
>>> import torch
>>> import torch_npu  
>>> torch.npu.set_device(0) 
>>> torch_npu.npu.stop_device(0)
```
