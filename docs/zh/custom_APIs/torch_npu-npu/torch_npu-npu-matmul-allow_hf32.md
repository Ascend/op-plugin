# torch_npu.npu.matmul.allow_hf32

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
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id5 -->

## 功能说明

matmul类算子开启对hf32类型的支持。

`torch_npu.npu.matmul.allow_hf32`功能和调用方式与`torch.backends.cuda.matmul.allow_tf32`类似，`torch.backends.cuda.matmul.allow_tf32`的功能具体请参考[PyTorch的allow_tf32功能说明](https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.matmul.allow_tf32)。

## 函数原型

```python
torch_npu.npu.matmul.allow_hf32 = bool
```

## 参数说明

`allow_hf32`参数类型为bool，取值范围为True或False（True表示开启matmul类算子对hf32类型的支持，False表示不开启），默认值为False。

## 返回值说明

`bool`

## 调用示例

```python
>>>import torch
>>>import torch_npu
>>>print(torch_npu.npu.matmul.allow_hf32)
False
>>>torch_npu.npu.matmul.allow_hf32=True
>>>print(torch_npu.npu.matmul.allow_hf32)
True
>>>torch_npu.npu.matmul.allow_hf32=False
>>>print(torch_npu.npu.matmul.allow_hf32)
False
```
