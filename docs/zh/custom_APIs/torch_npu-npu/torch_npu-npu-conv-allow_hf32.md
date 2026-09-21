# torch_npu.npu.conv.allow_hf32

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

conv类算子开启支持hf32类型能力。

`torch_npu.npu.conv.allow_hf32`功能和调用方式与`torch.backends.cudnn.allow_tf32`类似，`torch.backends.cudnn.allow_tf32`的功能具体请参考[https://pytorch.org/docs/stable/backends.html\#torch.backends.cudnn.allow_tf32](https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.allow_tf32)。

## 函数原型

```python
torch_npu.npu.conv.allow_hf32: bool
```

## 参数说明

bool型配置属性，无返回值。直接赋值修改，默认值为True。

## 调用示例

```python
>>>import torch
>>>import torch_npu
>>>print(torch_npu.npu.conv.allow_hf32)
True
>>>torch_npu.npu.conv.allow_hf32=False
>>>print(torch_npu.npu.conv.allow_hf32)
False
>>>torch_npu.npu.conv.allow_hf32=True
>>>print(torch_npu.npu.conv.allow_hf32)
True
```
