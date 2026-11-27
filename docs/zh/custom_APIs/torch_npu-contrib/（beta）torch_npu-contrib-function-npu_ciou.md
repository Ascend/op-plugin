# （beta）torch_npu.contrib.function.npu_ciou

> [!NOTICE]  
> 该接口计划废弃，底层算子Kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

基于NPU的CIoU操作，计算预测边界框与真实边界框之间的CIoU（Complete Intersection over Union）分数。可在损失函数中使用该分数。

## 函数原型

```python
torch_npu.contrib.function.npu_ciou(boxes1, boxes2, trans=True, is_cross=False, mode=0)
```

## 参数说明

- **boxes1**（`Tensor`）：格式为xywh、shape为(4, n)的预测边界框。
- **boxes2**（`Tensor`）：相应的真实边界框，shape为(4, n)。
- **trans**（`bool`）：是否有偏移，默认值为True。
- **is_cross**（`bool`）：控制两组框的比较方式。默认值为`False`，按对应位置逐对计算。
- **mode**（`int`）：选择CIoU的计算方式。`0`（默认值）基于交并比（IoU），`1`基于前景交叠比（IoF）。

## 返回值说明

`Tensor`。在默认的一一对应模式下，形状为 `(1, n)`，每项为一对框的CIoU分数。

## 约束说明

- 反向计算要求`trans=True`、`is_cross=False`、`mode=0`。

## 调用示例

示例为32对预测框和真值框计算CIoU分数，再用分数之和验证反向传播。

```python
import torch
import torch_npu
from torch_npu.contrib.function import npu_ciou
box1 = torch.randn(4, 32).npu()
box1.requires_grad = True
box2 = torch.randn(4, 32).npu()
box2.requires_grad = True
ciou = npu_ciou(box1, box2)
l = ciou.sum()
l.backward()
```

输出含32个得分，形状为 `(1, 32)`。对其求和后得到标量，`backward()`计算两个输入张量的梯度。
