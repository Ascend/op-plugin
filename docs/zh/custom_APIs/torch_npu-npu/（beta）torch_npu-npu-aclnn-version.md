# （beta）torch_npu.npu.aclnn.version

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

查询aclnn算子版本信息。aclnn算子详情可参考《CANN 算子库》中的“<a href="https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/API/aolapi/operatorlist_00001.html">简介</a>”章节。

## 函数原型

```python
torch_npu.npu.aclnn.version(): -> None
```

## 约束说明

当前aclnn暂时不支持查询版本，默认返回None。待aclnn支持后可以返回正确版本信息。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> res = torch_npu.npu.aclnn.version()
```
