# (beta) torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Applies an NPU-based bounding box format encoding operation to convert format from `xywh` to `xyxy`.

## Prototype

```python
torch_npu.contrib.function.npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, means=None, stds=None, max_shape=[9999, 9999], wh_ratio_clip=16 / 1000)
```

## Parameters

- **`bboxes`** (`Tensor`): Base bounding boxes. This parameter must be 2D with shape `(N, 4)`. The data type can be `float` or `half`.
- **`pred_bboxes`** (`Tensor`): Encoded bounding boxes. This parameter must be 2D with shape `(N, 4)`. The data type can be `float` or `half`.
- **`means`** (`List[float]`): Method to denormalize delta coordinates. The default value is `None`. This parameter must match the encoding parameters.
- **`stds`** (`List[float]`): Standard deviations used to denormalize delta coordinates. The default value is `None`. This parameter must match the encoding parameters.
- **`max_shape`** (`Tuple[int]`): Optional. Maximum bounding box shape `(H, W)`, which typically corresponds to the size of the original image. The default value is `[9999, 9999]`, indicating no restriction.
- **`wh_ratio_clip`** (`float`): Optional. Maximum allowed aspect ratio. The default value is `16/1000`.

## Return Values

`Tensor`

Bounding boxes with shape `(N, 4)`, where `4` represents `tl_x`, `tl_y`, `br_x`, and `br_y`.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.function import npu_bbox_coder_decode_xywh2xyxy
>>> A = 1024
>>> max_shape = 512
>>> bboxes = torch.randint(0, max_shape, size=(A, 4)).npu()
>>> pred_bboxes = torch.randn(A, 4).npu()
>>> out = npu_bbox_coder_decode_xywh2xyxy(bboxes, pred_bboxes, max_shape=(max_shape, max_shape))
>>> torch.npu.synchronize()
>>> print('npu_bbox_coder_decode_xywh2xyxy done. output shape is ', out.shape)
npu_bbox_coder_decode_xywh2xyxy done. output shape is torch.Size([1024, 4])
```
