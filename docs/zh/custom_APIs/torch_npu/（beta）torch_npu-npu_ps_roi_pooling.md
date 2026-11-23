# （beta）torch_npu.npu_ps_roi_pooling

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

执行Position Sensitive ROI Pooling。

## 函数原型

```python
torch_npu.npu_ps_roi_pooling(x, rois, spatial_scale, group_size, output_dim) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：必选参数，输入特征图，形状为`(N, C, H, W)`。`N`为批大小，`C = output_dim * group_size**2`，`H`和`W`分别为特征图的高和宽。
- **rois** (`Tensor`)：shape为[batch, 5, rois_num]的张量，用于描述ROI。每个ROI由五个元素组成：“batch_id”、“x1”、“y1”、“x2”和“y2”，其中“batch_id”表示输入特征图的index，“x1”、“y1”、“x2”和“y2”必须大于或等于“0.0”。
- **spatial_scale** (`float`)：将输入坐标映射到ROI坐标的缩放系数。数据类型支持`float32`。
- **group_size** (`int`)：指定用于编码position-sensitive评分图的组数。该值必须在(0,128)范围内。数据类型支持`int32`。
- **output_dim** (`int`)：指定输出通道数。必须大于0。数据类型支持`int32`。

## 调用示例

```python
>>> import torch
>>> import torch_npu

>>> x = torch.ones((2, 8, 4, 4), dtype=torch.float16)
>>> x[1] = 2
>>> x = x.npu()

>>> # rois 的形状为 (2, 5, 1)，每个 ROI 依次包含 batch_id、x1、y1、x2、y2。
>>> rois = torch.tensor(
    [[[0], [0], [0], [3], [3]], [[1], [0], [0], [3], [3]]],
    dtype=torch.float16,
    ).npu()

>>> out = torch_npu.npu_ps_roi_pooling(x, rois, spatial_scale=1.0, group_size=2, output_dim=2)
>>> print(out.shape)  # torch.Size([2, 2, 2, 2])
```
