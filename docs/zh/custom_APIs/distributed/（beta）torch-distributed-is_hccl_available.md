# （beta）torch.distributed.is_hccl_available

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas 训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

判断HCCL通信后端是否可用，与torch.distributed.is_nccl_available类似，具体请参考[https://pytorch.org/docs/stable/distributed.html\#torch.distributed.is_nccl_available](https://pytorch.org/docs/stable/distributed.html#torch.distributed.is_nccl_available)。

## 函数原型

```python
torch.distributed.is_hccl_available()
```

## 返回值说明

`bool`：True为可用，False为不可用。

## 调用示例

```python
import torch
import torch_npu

print(torch.distributed.is_hccl_available())

True
```
