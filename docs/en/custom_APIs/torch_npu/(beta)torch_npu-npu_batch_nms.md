# (beta) torch_npu.npu_batch_nms

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas training products</term>                                      |    √     |
|<term>Atlas inference products</term>                                      |    √     |

## Function

Performs batched per-class non-maximum suppression (NMS) on bounding boxes to remove redundant detections, and outputs the remaining bounding boxes along with their corresponding classes and scores.

## Prototype

```python
torch_npu.npu_batch_nms(self, scores, score_threshold, iou_threshold, max_size_per_class, max_total_size, change_coordinate_frame=False, transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Input box tensor containing the batch size. The data type must be `float16`. Example shape is `(batch_size, num_anchors, q, 4)`, where `q` is `1` or `num_classes`.
- **`scores`** (`Tensor`): Required. Input score tensor. The data type is `float16`. Example shape is `(batch_size, num_anchors, num_classes)`.
- **`score_threshold`** (`float`): Required. Score filter threshold used to screen boxes and remove boxes with low scores. The data type is `float32`.
- **`iou_threshold`** (`float`): Required. NMS IoU threshold used to set a threshold and remove boxes higher than the threshold. The data type is `float32`.
- **`max_size_per_class`** (`int`): Required. Maximum optional number of boxes for each class.
- **`max_total_size`** (`int`): Required. Maximum optional number of boxes for each batch.
- **`change_coordinate_frame`** (`bool`): Optional. Specifies whether to normalize the coordinates matrix of output boxes. The default value is `False`.
- **`transpose_box`** (`bool`): Optional. Specifies whether to insert a transpose before this operator. The default value is `False`. If set to `True`, boxes use the `(4, N)` layout. If set to `False`, boxes use the `(N, 4)` layout.

## Return Values

- **`nmsed_boxes`** (`Tensor`): NMS boxes output for each batch. This parameter must be a 3D tensor with shape `(batch, max_total_size, 4)`. The data type is `float16`.
- **`nmsed_scores`** (`Tensor`): NMS scores output for each batch. This parameter must be a 2D tensor with shape `(batch, max_total_size)`. The data type is `float16`.
- **`nmsed_classes`** (`Tensor`): NMS classes output for each batch. This parameter must be a 2D tensor with shape `(batch, max_total_size)`. The data type is `float16`.
- `nmsed_num` (`Tensor`): Valid number of `nmsed_boxes`. This parameter must be a 1D tensor with shape `(batch)`. The data type is `int32`.

## Example

```python
>>> import torch, torch_npu
>>> boxes = torch.randn(8, 2, 4, 4, dtype = torch.float32).to("npu")
>>> scores = torch.randn(3, 2, 4, dtype = torch.float32).to("npu")
>>> nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_num = torch_npu.npu_batch_nms(boxes, scores, 0.3, 0.5, 3, 4)
```
