# （beta）torch_npu.npu.aclnn.version

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

查询aclnn算子版本信息。aclnn算子详情可参考《CANN 算子库接口参考》中的“<a href="https://www.hiascend.com/document/detail/zh/canncommercial/850/API/aolapi/operatorlist_00001.html">简介</a>”章节。

## 函数原型

```
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

