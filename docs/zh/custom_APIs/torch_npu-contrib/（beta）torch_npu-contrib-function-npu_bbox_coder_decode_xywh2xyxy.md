# （beta）torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy

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

将基础框上的预测偏移量解码为 (x1, y1, x2, y2) 格式的边界框。

## 函数原型

```python
torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, means=None, stds=None, max_shape=None, wh_ratio_clip=16 / 1000)
```

## 参数说明

- **bboxes** (`Tensor`)：基础框，形状为 (N, 4)，每行按 (x1, y1, x2, y2) 给出两个对角顶点坐标。支持的数据类型为`torch.float`，`torch.half`。
- **pred_bboxes** (`Tensor`)：预测偏移量，形状为 (N, 4)。依次为水平和垂直中心偏移量、宽和高的对数缩放量。支持的数据类型为`torch.float`、`torch.half`。
- **means** (`List[float]`)：对delta坐标的目标去归一化的均值参数，默认值为None。该参数需要与编码参数对齐。
- **stds** (`List[float]`)：对delta坐标的目标去归一化的标准差，默认值为None。该参数需要与编码参数对齐。
- **max_shape** (`tuple[int, int] | list[int] | None`)：可选参数，按 `(H, W)`指定图像高和宽，解码坐标裁剪至该尺寸。默认值为`None`，内部使用 `[9999, 9999]`。
- **wh_ratio_clip** (`float`)：可选参数，可允许的宽高比裁剪阈值，用于限制解码后框宽高的变化幅度，取值需大于0，默认值为16/1000（即0.016）。

## 返回值说明

`Tensor`

代表shape为(N, 4)的框，其中4表示tl_x、tl_y、br_x、br_y。

## 调用示例

示例以1024个基础框和对应的回归偏移为输入，将解码坐标裁剪至512×512的图像范围。

```python
import torch, torch_npu
from torch_npu.contrib.function import npu_bbox_coder_decode_xywh2xyxy
A = 1024
max_shape = 512
bboxes = (torch.rand(A, 4) * max_shape).npu()
pred_bboxes = torch.randn(A, 4).npu()
out = npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, max_shape=(max_shape, max_shape))
torch.npu.synchronize()
print(out.shape)
```

示例输出如下。

```text
torch.Size([1024, 4])
```
