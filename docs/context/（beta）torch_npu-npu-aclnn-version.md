# （beta）torch_npu.npu.aclnn.version

## 函数原型

```
torch_npu.npu.aclnn.version(): -> None
```

## 功能说明

查询aclnn版本信息。

## 约束说明

当前aclnn暂时不支持查询版本，默认返回None。待aclnn支持后可以返回正确版本信息。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> res = torch_npu.npu.aclnn.version()
```

