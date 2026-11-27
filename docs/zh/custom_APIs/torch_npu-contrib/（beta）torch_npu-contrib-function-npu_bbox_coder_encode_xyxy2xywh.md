# （beta）torch\_npu.contrib.function.npu\_bbox\_coder\_encode\_xyxy2xywh

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

应用基于NPU的bbox格式编码操作，将格式从xyxy编码为xywh。

## 函数原型

```python
torch_npu.contrib.function.npu_bbox_coder_encode_xyxy2xywh(bboxes,gt_bboxes, means=None, stds=None, is_normalized=False, normalized_scale=10000.)
```

## 参数说明

- **bboxes** (`Tensor`)：待转换的框，shape为\(N, 4\)。支持的数据类型为`torch.float`，`torch.half`。
- **gt_bboxes** (`Tensor`)：用作基准的gt\_bboxes，shape为\(N, 4\)。支持的数据类型为`torch.float`、`torch.half`。
- **means** (`List[float]`)：四个delta分量的去归一化均值。提供列表时使用前四项。默认值为None，对应四个0。
- **stds** (`List[float]`)：四个delta分量的去归一化标准差。提供列表时使用前四项。默认值为None，对应四个1。
- **is_normalized** (`bool`)：坐标值是否已归一化，默认值为False。
- **normalized_scale** (`float`)：设置坐标恢复的归一化比例，默认值为10000.。

## 返回值说明

`Tensor`

框转换deltas。

## 约束说明

此接口调用的`torch_npu.npu_bounding_box_encode`已标记为废弃，运行时可能发出弃用警告。

不支持动态shape。由于算子语义限制，仅支持二维\(n, 4\)场景。`bboxes`和`gt_bboxes`的shape和dtype必须相同，dtype只可为`torch.float16`和`torch.float32`。

## 调用示例

示例将每对源框和真值框编码为四个回归增量。两组输入各包含1024个框。

```python
import torch, torch_npu
from torch_npu.contrib.function import npu_bbox_coder_encode_xyxy2xywh
A = 1024
bboxes = torch.randint(0, 512, size=(A, 4)).float().npu()
gt_bboxes = torch.randint(0, 512, size=(A, 4)).float().npu()
out = npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes)
torch.npu.synchronize()
print(out.shape)
```

示例输出如下。

```text
torch.Size([1024, 4])
```
