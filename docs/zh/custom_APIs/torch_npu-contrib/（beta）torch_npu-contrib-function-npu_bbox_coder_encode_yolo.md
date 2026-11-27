# （beta）torch_npu.contrib.function.npu_bbox_coder_encode_yolo

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

通过NPU算子来计算从源框（bbox）到目标框（gt_bbox）的YOLO风格框回归转换deltas。

## 函数原型

```python
torch_npu.contrib.function.npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
```

## 参数说明

- **bboxes** (`Tensor`)：必选参数，形状为 `(N, 4)`的源框，例如锚点框，支持`torch.float32`和`torch.float16`。
- **gt_bboxes** (`Tensor`)：必选参数，转换目标框，例如真值框，形状和数据类型须与`bboxes`一致。
- **stride** (`Tensor`)：必选参数，形状为 `(N,)`的步长张量，每个框对应一个步长。算子接收`torch.int32`，封装会将输入步长转为`torch.int32`。

## 返回值说明

`Tensor`

框转换deltas。

## 调用示例

示例为1024对边界框各提供一个步长，并计算每对框的四个编码增量。

```python
import torch, torch_npu
from torch_npu.contrib.function import npu_bbox_coder_encode_yolo
A = 1024
bboxes = torch.randn((A, 4), dtype=torch.float32).npu()
gt_bboxes = torch.randn((A, 4), dtype=torch.float32).npu()
stride = torch.randint(1, 32, size=(A,), dtype=torch.int32).npu()
out = npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
torch.npu.synchronize()
print(out.shape)
```

示例输出如下。

```text
torch.Size([1024, 4])
```
