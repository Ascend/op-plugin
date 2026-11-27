# （beta）torch_npu.contrib.npu_giou

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

提供NPU版本的GIoU计算接口。

## 函数原型

```python
torch_npu.contrib.npu_giou(boxes1, boxes2, is_permuted=True)
```

## 参数说明

- **boxes1** (`Tensor`)：必选参数，预测框，建议用`torch.float32`。每行依次为中心坐标 `(x, y)`和宽、高 `(w, h)`。
- **boxes2** (`Tensor`)：必选参数，真值框，与`boxes1`一一对应，形状相同。
- **is_permuted** (`bool`)：可选参数，默认值为`True`，两组输入均为 `(N, 4)`，接口将其转置为算子使用的 `(4, N)`。如果输入已经是 `(4, N)`，请传入`False`。

## 调用示例

计算两对预测框与真值框的GIoU。第一对框相同，得分为1。

```python
import torch
import torch_npu
from torch_npu.contrib import npu_giou

boxes1 = torch.tensor([[10., 10., 4., 2.], [20., 20., 4., 4.]], device="npu")
boxes2 = torch.tensor([[10., 10., 4., 2.], [21., 20., 4., 4.]], device="npu")
scores = npu_giou(boxes1, boxes2)
print(scores)
```
