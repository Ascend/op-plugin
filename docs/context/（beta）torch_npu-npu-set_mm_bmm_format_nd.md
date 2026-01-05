# （beta）torch_npu.npu.set_mm_bmm_format_nd
## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

设置线性module里面的mm和bmm算子是否用ND格式。

## 函数原型

```
torch_npu.npu.set_mm_bmm_format_nd(bool)
```

## 调用示例

```python
import torch
import torch_npu
torch_npu.npu.set_mm_bmm_format_nd(True)
```
