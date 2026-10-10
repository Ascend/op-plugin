# （beta）torch_npu.npu.aclnn.version

> [!NOTICE]  
> 此接口在本版本中有变更，具体变更内容请参考《版本说明》中的“[接口变更说明](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/zh/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E)”。

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

查询aclnn算子版本信息。aclnn算子详情可参考《CANN 算子库》中的“<a href="https://gitcode.com/cann/docs/blob/9.2.0/docs/zh/ops-lib/introduction.md">简介</a>”章节。

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
