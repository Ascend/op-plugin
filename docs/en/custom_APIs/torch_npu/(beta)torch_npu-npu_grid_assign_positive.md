# (beta) torch_npu.npu_grid_assign_positive

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Computes the position-sensitive candidate region pooling gradients.

## Prototype

```python
torch_npu.npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. The data type can be `float16` or `float32`. The shape of this parameter is `(n,)`.
- **`overlaps`** (`Tensor`): Required. IoU between `gt_bboxes` and `bboxes`. The data type must be identical to that of `self`. The shape of this parameter is `(k, n)`.
- **`box_responsible_flags`** (`Tensor`): Required. Indicates whether a box is responsible. The data type can be `uint8`.
- **`max_overlaps`** (`Tensor`): Required. The data type must be identical to that of `self`.
- **`argmax_overlaps`** (`Tensor`): Required. The data type can be `int32`.
- **`gt_max_overlaps`** (`Tensor`): Required. The data type must be identical to that of `self`.
- **`gt_argmax_overlaps`** (`Tensor`): Required. The data type can be `int32`.
- **`num_gts`** (`Tensor`): Required. Real $k$. The data type can be `int32`. The shape is `(1,)`.
- **`pos_iou_thr`** (`float`): Required. IoU threshold for positive bounding boxes.
- **`min_pos_iou`** (`float`): Required. Minimum IoU for a bounding box to be considered a positive bounding box.
- **`gt_max_assign_all`** (`bool`): Required. Specifies whether to assign all bounding boxes that have the same maximum overlap with a ground-truth box to that ground-truth box.

## Return Values

`Tensor`

Computation results of the position-sensitive candidate region pooling gradients.

## Example

```python
>>> import torch, torch_npu
>>> assigned_gt_inds = torch.rand(4).npu()
>>> overlaps = torch.rand(2,4).npu()
>>> box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).npu()
>>> max_overlap = torch.rand(4).npu()
>>> argmax_overlaps = torch.tensor([1, 0, 1, 0], dtype=torch.int32).npu()
>>> gt_max_overlaps = torch.rand(2).npu()
>>> gt_argmax_overlaps = torch.tensor([1, 0],dtype=torch.int32).npu()
>>> num_gts = torch.tensor([128], dtype=torch.int32).npu()
>>> output = torch_npu.npu_grid_assign_positive(assigned_gt_inds, overlaps, box_responsible_flags, max_overlap, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, 0.5, 0., True)
>>> print(output.shape)
torch.Size([4])
```
