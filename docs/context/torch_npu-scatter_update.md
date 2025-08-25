# torch_npu.scatter_update

# 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>    | √  |
|<term>Atlas 训练系列产品</term>    | √  |

## 功能说明

将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据不变。

## 函数原型

```
torch_npu.scatter_update(data, indices, updates, axis) -> Tensor
```

## 参数说明

- **data** (`Tensor`)：必选参数。代表更新前的原数据，`data`只支持2-8维，且维度大小需要与`updates`一致；支持非连续的tensor；数据格式支持$ND$；不支持空Tensor。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
    - <term>Atlas A3 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
    - <term>Atlas 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`int32`。

- **indices** (`Tensor`)：必选参数。代表索引，数据类型支持`int32`、`int64`；目前仅支持一维和二维；支持非连续的tensor；数据格式支持$ND$；不支持空Tensor。
- **updates** (`Tensor`)：必选参数。代表更新的数据，`updates`的维度大小需要与`data`一致；支持非连续的tensor；数据格式支持$ND$；不支持空Tensor。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
    - <term>Atlas A3 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
    - <term>Atlas 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`int32`。

- **axis** (`int`)：必选参数。代表轴，用来scatter的维度，数据类型为`int64`。

## 返回值说明
`Tensor`

计算输出，只支持2-8维，且维度大小需要与`data`一致；支持非连续的tensor；数据格式支持$ND$；不支持空Tensor。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
- <term>Atlas A3 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`bfloat16`、`int32`。
- <term>Atlas 训练系列产品</term>：数据类型支持`int8`、`float16`、`float32`、`int32`。

## 约束说明

- `data`与`updates`的秩一致。
- 不支持索引越界，索引越界不校验。

## 调用示例

```python
import torch
import torch_npu
import numpy as np
data = torch.tensor([[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]]]], dtype=torch.float32).npu()
indices = torch.tensor ([1],dtype=torch.int64).npu()
updates = torch.tensor([[[[3,3,3,3,3,3,3,3]]]] , dtype=torch.float32).npu()
out = torch_npu.scatter_update(data, indices, updates, axis=-2)
```

