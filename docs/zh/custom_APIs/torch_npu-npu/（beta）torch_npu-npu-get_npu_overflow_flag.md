# （beta）torch\_npu.npu.get\_npu\_overflow\_flag

## 产品支持情况

<!-- npu="910" id1 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id1 -->

## 功能说明

检测NPU计算过程中是否有数值溢出。

## 函数原型

```python
torch_npu.npu.get_npu_overflow_flag()
```

## 调用示例

```python
import torch
import torch_npu
a = torch.Tensor([65535]).npu().half()
a = a + a
ret = torch_npu.npu.get_npu_overflow_flag()
```
