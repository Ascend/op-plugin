# （beta）torch_npu.npu.set_aoe

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

AOE调优使能。

## 函数原型

```
torch_npu.npu.set_aoe(dump_path)
```

## 参数说明

**dump_path**：dump算子图保存路径。

## 调用示例

```python
import torch
import torch_npu
import os

os.mkdir("./aoe_dump")
dump_path = "./aoe_dump"
torch_npu.npu.set_aoe(dump_path)

```
