# （beta）torch_npu.contrib.function.npu_diou

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

该接口用于实现基于NPU的DIoU（距离交并比）计算。该算法综合考虑预测框与真实框中心点之间的距离以及边界框的重叠率，使边界框回归趋于稳定。

## 函数原型

```python
torch_npu.contrib.function.npu_diou(boxes1, boxes2, trans=True, is_cross=False, mode=0)
```

## 参数说明

- **boxes1**（`Tensor`）：格式为xywh、shape为(4, n)的预测检测框。
- **boxes2**（`Tensor`）：相应的真实框，shape为(4, n)。
- **trans**（`bool`）：是否有偏移，默认值为True。
- **is_cross**（`bool`）：boxes1和boxes2之间是否有交叉操作，默认值为False。
- **mode**（`int`）：选择DIoU的计算方式。`0`（默认值）基于交并比（IoU），`1`基于前景交叠比（IoF）。

## 返回值说明

返回DIoU得分张量。设第一组有N个框、第二组有M个框。`is_cross=False`时输出形状为 (1, N)，每两个位置对应的框一一计算。`is_cross=True`时输出形状为 (M, N)。

## 约束说明

到目前为止，DIoU反向只支持当前版本中的trans==True、is_cross==False、mode==0。如果需要反向传播，确保参数正确。

## 调用示例

示例为32对预测框和真值框计算DIoU分数，再用分数之和验证反向传播。

```python
import torch
import torch_npu
from torch_npu.contrib.function import npu_diou
box1 = torch.randn(4, 32).npu()
box1.requires_grad = True
box2 = torch.randn(4, 32).npu()
box2.requires_grad = True
diou = npu_diou(box1, box2)
l = diou.sum()
l.backward()
```

输出含32个得分，形状为 `(1, 32)`。对其求和后得到标量，`backward()`计算两个输入张量的梯度。
