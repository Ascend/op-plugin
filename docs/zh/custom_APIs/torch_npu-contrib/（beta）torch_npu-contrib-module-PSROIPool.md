# （beta）torch_npu.contrib.module.PSROIPool

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

对输入特征图中的感兴趣区域执行位置敏感池化，得到固定空间尺寸的特征。可用于目标检测中从不同大小的候选区域提取特征。

## 函数原型

```python
torch_npu.contrib.module.PSROIPool(pooled_height=7, pooled_width=7, spatial_scale=0.0625, group_size=7, output_dim=22)
```

## 参数说明

**计算参数**

- **pooled_height** (`int`) - 池化高度。
- **pooled_width** (`int`) - 池化宽度。
- **spatial_scale** (`float`) - 按此参数值缩放输入框。
- **group_size** (`int`) - 编码位置敏感分数图的组数。
- **output_dim** (`int`) - 输出通道数。

**计算输入**

- **features** (`Tensor`) - 形状为 `(N, C, H, W)`的特征图，其中`C = output_dim * group_size * group_size`。
- **rois** (`Tensor`) - ROI信息张量，至少三维，第二维大小为5，依次为批次索引和`x1, y1, x2, y2`坐标。形状为 `(N, 5, R)`时，`R`为每张图像的ROI数量。

## 返回值说明

`Tensor` 

PSROIPool计算结果，shape为(`rois.size(0) * rois.size(2)`, `output_dim`, `pooled_height`, `pooled_width`)。

## 约束说明

仅实现了pooled_height == pooled_width == group_size。

## 调用示例

示例对一张包含8个通道的特征图执行位置敏感池化。池化高、宽和组大小均为2，输出通道数为2。使用`spatial_scale=1.0`，因为ROI坐标已在特征图尺度上。

```python
import torch
import torch_npu
from torch_npu.contrib.module import PSROIPool

features = torch.ones((1, 8, 4, 4), device="npu")
rois = torch.tensor([[[0.], [0.], [0.], [3.], [3.]]], device="npu")
model = PSROIPool(pooled_height=2, pooled_width=2, spatial_scale=1.0, group_size=2, output_dim=2).npu()
print(model(features, rois).shape)
```

示例输出如下。

```text
torch.Size([1, 2, 2, 2])
```
