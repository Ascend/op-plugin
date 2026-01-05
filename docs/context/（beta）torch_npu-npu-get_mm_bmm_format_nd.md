# （beta）torch_npu.npu.get_mm_bmm_format_nd
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

确认线性module里面的mm和bmm算子是否有使能ND格式，如果使能了ND，返回True，否则，返回False。

## 函数原型

```
torch_npu.npu.get_mm_bmm_format_nd()
```

## 返回值说明

`bool`


## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> torch_npu.npu.get_mm_bmm_format_nd()
True
```

