# （beta）torch_npu.npu.get_autocast_dtype


## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |


## 功能说明

在amp场景获取设备支持的数据类型，该`dtype`由torch_npu.npu.set_autocast_dtype设置或者默认数据类型`float16`。


## 函数原型

```
torch_npu.npu.get_autocast_dtype()
```
## 返回值说明

`torch.dtype`

## 调用示例

```python
import torch
import torch_npu

current_dtype = torch_npu.npu.get_autocast_dtype()

```

