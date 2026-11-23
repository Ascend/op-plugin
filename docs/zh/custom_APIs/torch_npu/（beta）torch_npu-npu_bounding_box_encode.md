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

计算锚点框和真值框之间的坐标变化。自定义Faster R-CNN算子。

## 函数原型

```python
torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3) -> Tensor
```

## 参数说明

- **anchor_box** (`Tensor`)：锚点框张量，数据类型为`torch.float32`，形状为`(N, 4)`。`N`为锚点框数量，每个框按`(x0, y0, x1, y1)`排列。
- **ground_truth_box** (`Tensor`)：真值框张量，数据类型为`torch.float32`，形状为`(N, 4)`。`N`为真值框数量，每个框按`(x0, y0, x1, y1)`排列。
- **means0** (`float`)：“dx”的偏差值。
- **means1** (`float`)：“dy”的偏差值。
- **means2** (`float`)：“dw”的偏差值。
- **means3** (`float`)：“dh”的偏差值。
- **stds0** (`float`)：“dx”的缩放值。
- **stds1** (`float`)：“dy”的缩放值。
- **stds2** (`float`)：“dw”的缩放值。
- **stds3** (`float`)：“dh”的缩放值。

## 返回值说明

`Tensor`

返回形状为`(N, 4)`、数据类型为`torch.float32`的边界框变换量张量，每行依次为`(dx, dy, dw, dh)`。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> anchor_box = torch.tensor([[1., 2., 3., 4.]], dtype = torch.float32).to("npu")
>>> ground_truth_box = torch.tensor([[5., 6., 7., 8.]], dtype = torch.float32).to("npu")
>>> output = torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
>>> print(output)
tensor([[13.3281, 13.3281,  0.0000,  0.0000]], device='npu:0')
```
