# （beta）torch_npu.npu.aclnn.version

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |



## 功能说明

查询aclnn版本信息。

## 函数原型

```
torch_npu.npu.aclnn.version(): -> None
```


## 约束说明

当前`aclnn`暂时不支持查询版本，默认返回None。待`aclnn`支持后可以返回正确版本信息。


## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> res = torch_npu.npu.aclnn.version()
```

