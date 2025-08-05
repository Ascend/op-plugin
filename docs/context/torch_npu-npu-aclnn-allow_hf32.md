# torch_npu.npu.aclnn.allow_hf32

## 函数原型

```
torch_npu.npu.aclnn.allow_hf32:bool
```

## 功能说明

设置conv算子是否支持hf32，一个属性值，对aclnn的allow_hf32属性的设置和查询。

## 参数说明

bool类型，开启和关闭hf32属性的支持。

## 输出说明

返回bool类型，返回当前allow_hf32是否开启，默认值为True。

## 支持的型号

- <term> Atlas 训练系列产品</term> 
- <term> Atlas A2 训练系列产品</term> 
- <term> Atlas A3 训练系列产品</term> 
- <term> Atlas 推理系列产品</term> 

## 调用示例

```python
>>> import torch
>>>import torch_npu
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

