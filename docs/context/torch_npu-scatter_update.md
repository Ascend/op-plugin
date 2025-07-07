# torch_npu.scatter_update

## 功能说明

将tensor updates中的值按指定的轴axis和索引indices更新tensor data中的值，并将结果保存到输出tensor，data本身的数据不变。

## 函数原型

```
torch_npu.scatter_update(Tensor data, Tensor indices, Tensor updates, int axis) -> Tensor
```

## 参数说明

- data：Tensor类型，data只支持2-8维，且维度大小需要与updates一致；支持非连续的tensor；数据格式支持ND；不支持空Tensor。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持int8、float16、float32、bfloat16、int32。
    - <term>Atlas A3 训练系列产品</term>：数据类型支持int8、float16、float32、bfloat16、int32。
    - <term>Atlas 训练系列产品</term>：数据类型支持int8、float16、float32、int32。

- indices：Tensor类型，数据类型支持int32、int64；目前仅支持一维和二维；支持非连续的tensor；数据格式支持ND；不支持空Tensor。
- updates：Tensor类型，updates的维度大小需要与data一致；支持非连续的tensor；数据格式支持ND；不支持空Tensor。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持int8、float16、float32、bfloat16、int32。
    - <term>Atlas A3 训练系列产品</term>：数据类型支持int8、float16、float32、bfloat16、int32。
    - <term>Atlas 训练系列产品</term>：数据类型支持int8、float16、float32、int32。

- axis：整型，用来scatter的维度，数据类型为int64。

## 输出说明

out：Tensor类型，计算输出，out只支持2-8维，且维度大小需要与data一致；支持非连续的tensor；数据格式支持ND；不支持空Tensor。

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持int8、float16、float32、bfloat16、int32。
- <term>Atlas A3 训练系列产品</term>：数据类型支持int8、float16、float32、bfloat16、int32。
- <term>Atlas 训练系列产品</term>：数据类型支持int8、float16、float32、int32。

## 约束说明

- data与updates的秩一致。
- 不支持索引越界，索引越界不校验。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
- <term>Atlas A3 训练系列产品</term>


## 调用示例

单算子模式调用：

```python
import torch
import torch_npu
import numpy as np
data = torch.tensor([[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2]]]], dtype=torch.float32).npu()
indices = torch.tensor ([1],dtype=torch.int64).npu()
updates = torch.tensor([[[[3,3,3,3,3,3,3,3]]]] , dtype=torch.float32).npu()
out = torch_npu.scatter_update(data, indices, updates, axis=-2)
```

