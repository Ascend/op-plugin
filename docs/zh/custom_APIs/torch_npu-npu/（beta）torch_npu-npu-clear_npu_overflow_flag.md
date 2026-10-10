# （beta）torch\_npu.npu.clear\_npu\_overflow\_flag

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="310p" id2 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->

## 功能说明

对NPU溢出检测进行清零。

## 函数原型

```python
torch_npu.npu.clear_npu_overflow_flag()
```

## 约束说明

仅在饱和模式下生效。INF_NAN模式下接口仅发出warning后直接返回，不执行清零，建议使用 [torch_npu.npu.utils.npu_check_overflow](./（beta）torch_npu-npu-utils-npu_check_overflow.md)。

## 调用示例

```python
import torch
import torch_npu

a = torch.Tensor([65535]).npu().half()
a = a + a
if torch_npu.npu.get_npu_overflow_flag():
    torch_npu.npu.clear_npu_overflow_flag()
```
