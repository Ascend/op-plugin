# (beta) torch_npu.contrib.function.npu_bbox_coder_encode_xyxy2xywh

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Applies an NPU-based bounding box format encoding operation to convert format from `xyxy` to `xywh`.

## Prototype

```python
torch_npu.contrib.function.npu_bbox_coder_encode_xyxy2xywh(bboxes,gt_bboxes, means=None, stds=None, is_normalized=False, normalized_scale=10000.)
```

## Parameters

- **`bboxes`** (`Tensor`): Bounding boxes to be converted. This parameter must be 2D with shape (N, 4). The data type can be `float` or `half`.
- **`gt_bboxes`** (`Tensor`): Ground truth bounding boxes used as a reference. This parameter must be 2D with shape (N, 4). The data type can be `float` or `half`.
- **`means`** (`List[float]`): Method to denormalize delta coordinates. The default value is `None`.
- **`stds`** (`List[float]`): Standard deviations used to denormalize delta coordinates. The default value is `None`.
- **`is_normalized`** (`bool`): Indicates whether the coordinate values have been normalized. The default value is `False`.
- **`normalized_scale`** (`float`): Normalization scale used to restore coordinates. The default value is `10000`.

## Return Values

`Tensor`

Bounding box transformation deltas.

## Constraints

Dynamic shapes are not supported. Due to operator semantic limitations, only 2D scenarios with shape (N, 4) are supported. The shapes and data types of `bboxes` and `gt_bboxes` must be identical. The data type must be `float16` or `float32`. The third input (`stride`) must be a 1D tensor, and its first dimension must match that of the first input (`bboxes`).

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import npu_bbox_coder_encode_xyxy2xywh
>>> A = 1024
>>> bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> gt_bboxes = torch.randint(0, 512, size=(A, 4)).npu()
>>> out = npu_bbox_coder_encode_xyxy2xywh(bboxes, gt_bboxes)
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_encode_xyxy2xywh done. output shape is ', out.shape)
npu_bbox_coder_encode_xyxy2xywh done. output shape is torch.Size([1024, 4])
```
