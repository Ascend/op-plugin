# （beta）torch_npu.npu.set_autocast_dtype

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

设置设备在AMP场景支持的数据类型。

## 函数原型

```
torch_npu.npu.set_autocast_dtype(dtype)
```

## 参数说明

 **dtype** ：数据类型。


## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.set_autocast_dtype(torch.float16)
```

