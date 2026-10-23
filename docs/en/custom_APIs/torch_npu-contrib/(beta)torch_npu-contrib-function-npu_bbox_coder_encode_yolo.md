# (beta) torch_npu.contrib.function.npu_bbox_coder_encode_yolo

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Computes YOLO-style bounding box regression transformation deltas from source bounding boxes (`bboxes`) to target bounding boxes (`gt_bboxes`) through an NPU operator.

## Prototype

```python
torch_npu.contrib.function.npu_bbox_coder_encode_yolo(bboxes, gt_bboxes, stride)
```

## Parameters

- **`bboxes`** (`Tensor`): Source bounding boxes, such as anchor boxes. The data type can be `torch.float` or `torch.half`.
- **`gt_bboxes`** (`Tensor`): Target bounding boxes, such as ground truth bounding boxes. The data type can be `torch.float` or `torch.half`.
- **`stride`** (`Tensor`): Bounding box stride. Only `torch.int` tensors are supported.

## Return Values

`Tensor`

Bounding box transformation deltas.

## Example

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
