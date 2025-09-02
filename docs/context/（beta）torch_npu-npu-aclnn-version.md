# （beta）torch_npu.npu.aclnn.version

## 函数原型

```
torch_npu.npu.aclnn.version(): -> None
```

## 功能说明

查询aclnn算子版本信息。aclnn算子详情可参考《CANN AOL算子加速库接口》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/aolapi/operatorlist_00001.html">接口简介</a>”章节。

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

