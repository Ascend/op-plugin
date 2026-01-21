# （beta）torch_npu.npu.is_autocast_enabled

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

确认autocast是否可用。

## 函数原型

```
torch_npu.npu.is_autocast_enabled()
```
## 返回值说明

`bool`



## 调用示例

``` python
import torch
import torch_npu
torch_npu.npu.is_autocast_enabled()
```

