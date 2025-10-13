# torch_npu.npu.aclnn.allow_hf32

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

设置或查询conv类算子是否支持hf32。

## 函数原型

```
torch_npu.npu.aclnn.allow_hf32:bool
```

## 参数说明

**bool**：开启和关闭hf32属性的支持。

## 返回值说明

返回`bool`类型，返回当前allow_hf32是否开启，默认值为True。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> res
True
>>> torch_npu.npu.aclnn.allow_hf32 = False
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> res
False
>>> torch_npu.npu.aclnn.allow_hf32 = True
>>> res = torch_npu.npu.aclnn.allow_hf32
>>> res
True
```

