# torch_npu.npu.aclnn.allow_hf32

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

设置或查询conv类算子是否支持hf32。

## 函数原型

```python
torch_npu.npu.aclnn.allow_hf32:bool
```

## 参数说明

**bool**：用于开启或关闭hf32属性的支持。

## 返回值说明

返回`bool`类型，表示当前allow_hf32是否开启，默认值为True。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
True
>>> torch_npu.npu.aclnn.allow_hf32 = False
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
False
>>> torch_npu.npu.aclnn.allow_hf32 = True
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> print(res)
True
```
