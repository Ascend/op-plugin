# （beta）torch_npu.npu_rotated_overlaps

> [!NOTICE]  
> 该接口计划废弃，底层算子kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

计算旋转框的重叠面积。

## 函数原型

```python
torch_npu.npu_rotated_overlaps(self, query_boxes, trans=False) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：必选参数，第一组旋转框数据，shape为(B, N, 5)，数据类型为`torch.float32`的3D张量。
- **query_boxes** (`Tensor`)：必选参数，标注框，shape为(B, K, 5)，数据类型为`torch.float32`的3D张量。
- **trans** (`bool`)：可选参数，指定最后一维的框参数顺序。True时为 (x1, y1, x2, y2, θ)，前四项为未旋转框的两个对角顶点坐标。False时为 (x, y, w, h, θ)，前四项为中心坐标、宽和高。θ为旋转角度，单位为度。默认值为False。

## 返回值说明

Tensor。输出形状为 (B, N, K)，每个元素对应一对框的重叠面积。B为批大小，N和K分别为两组框的数量。

## 调用示例

计算两组旋转框两两之间的重叠面积。

```python
import torch
import torch_npu

# 一个批次中分别有 2 个和 3 个旋转框。每行依次为中心坐标、宽、高和角度。
boxes = torch.tensor([[[10., 10., 4., 2., 0.], [30., 30., 2., 2., 0.]]], device="npu")
query_boxes = torch.tensor(
    [[[10., 10., 4., 2., 0.], [10., 10., 4., 2., 90.], [50., 50., 4., 2., 0.]]],
    device="npu",
)
output = torch_npu.npu_rotated_overlaps(boxes, query_boxes, trans=False)
print(output.shape)
```

输出：

```text
torch.Size([1, 2, 3])
```
