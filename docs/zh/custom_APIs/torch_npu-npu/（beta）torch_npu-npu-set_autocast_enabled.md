# （beta）torch_npu.npu.set_autocast_enabled

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

在设备上开启或关闭AMP。

## 函数原型

```python
torch_npu.npu.set_autocast_enabled(bool)
```

## 参数说明

**bool** ：入参为True时，在设备上开启AMP，否则，不开启AMP。

## 调用示例

```python
import torch
import torch_npu
torch_npu.npu.set_autocast_enabled(True)
```
