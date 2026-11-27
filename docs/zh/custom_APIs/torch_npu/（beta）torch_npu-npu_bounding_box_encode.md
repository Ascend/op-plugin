# （beta）torch_npu.npu_bounding_box_encode

> [!NOTICE]  
> 该接口计划废弃，底层算子Kernel实现不再维护，性能、精度等指标无法保障，不建议使用该接口。

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

计算锚点框到对应真值框的中心位置和大小偏移。自定义Faster R-CNN算子。

## 函数原型

```python
torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3) -> Tensor
```

## 参数说明

- **anchor_box** (`Tensor`)：锚点框张量，数据类型为`torch.float32`，形状为`(N, 4)`。`N`为锚点框数量，每个框按`(x0, y0, x1, y1)`排列。
- **ground_truth_box** (`Tensor`)：真值框张量，数据类型为`torch.float32`，形状为`(N, 4)`。`N`为真值框数量，每个框按`(x0, y0, x1, y1)`排列。
- **means0** (`float`)：必选参数，归一化水平中心偏移时减去的均值。
- **means1** (`float`)：必选参数，归一化垂直中心偏移时减去的均值。
- **means2** (`float`)：必选参数，归一化宽度对数缩放量时减去的均值。
- **means3** (`float`)：必选参数，归一化高度对数缩放量时减去的均值。
- **stds0** (`float`)：必选参数，归一化水平中心偏移时除以的标准差。
- **stds1** (`float`)：必选参数，归一化垂直中心偏移时除以的标准差。
- **stds2** (`float`)：必选参数，归一化宽度对数缩放量时除以的标准差。
- **stds3** (`float`)：必选参数，归一化高度对数缩放量时除以的标准差。

## 返回值说明

`Tensor`

返回形状为 `(N, 4)`、数据类型为`torch.float32`的边界框变换量张量。每行的 `(dx, dy, dw, dh)`依次表示归一化后的水平中心偏移、垂直中心偏移、宽度对数缩放量和高度对数缩放量。

## 调用示例

为两对边界框编码偏移量。第一对框相同，对应的四列偏移量均为0。均值及标准差按输出列顺序传入。

```python
import torch
import torch_npu

anchors = torch.tensor([[0., 0., 10., 10.], [5., 5., 20., 20.]], device="npu")
boxes = torch.tensor([[0., 0., 10., 10.], [6., 7., 21., 22.]], device="npu")
means = (0., 0., 0., 0.)
stds = (0.1, 0.1, 0.2, 0.2)
output = torch_npu.npu_bounding_box_encode(anchors, boxes, *means, *stds)
print(output.shape)
```

输出：

```text
torch.Size([2, 4])
```
