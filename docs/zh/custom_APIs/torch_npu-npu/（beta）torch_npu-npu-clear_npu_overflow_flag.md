# （beta）torch\_npu.npu.clear\_npu\_overflow\_flag

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

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
