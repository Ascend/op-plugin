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

通过 NPU OP来计算从源框（bbox）到目标框（gt_bbox）的 YOLO 风格框回归转换 deltas。

## 函数原型

```python
torch_npu.contrib.function.npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
```

## 参数说明

- **bboxes** (`Tensor`)：源框，例如锚点框。支持的数据类型为`float`、`half`。
- **gt_bboxes** (`Tensor`)：转换目标框，例如真值框。支持的数据类型为`float`、`half`。
- **stride** (`Tensor`)：bbox步长。仅支持`int`张量。

## 返回值说明

`Tensor`

框转换deltas。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import npu_bbox_coder_encode_yolo
>>> A = 1024
>>> bboxes = torch.randn((A, 4), dtype=torch.float32).npu()
>>> gt_bboxes = torch.randn((A, 4), dtype=torch.float32).npu()
>>> stride = torch.randint(0, 32, size=(A,)).npu()
>>> out = npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_encode_yolo done. output shape is ', out.shape)
npu_bbox_coder_encode_yolo done. output shape is torch.Size([1024, 4])
```
