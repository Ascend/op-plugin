# (beta) torch_npu.npu_nms_v4

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Selects a subset of bounding boxes in descending order of scores.

## Prototype

```python
torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size=False) -> (Tensor, Tensor)
```

## Parameters

- **`boxes`** (`Tensor`): Required. This parameter must be a 2D floating-point tensor with shape `(num_boxes, 4)`.
- **`scores`** (`Tensor`): Required. Score corresponding to each box. This parameter must be a 1D floating-point tensor with shape `(num_boxes,)`.
- **`max_output_size`** (`Scalar`): Required. Scalar representing the maximum number of boxes to select during non-maximum suppression (NMS).
- **`iou_threshold`** (`Tensor`): Required. Upper threshold for IoU overlap between boxes. This parameter must be a 0D floating-point tensor.
- **`scores_threshold`** (`Tensor`): Required. Score threshold used to decide when to remove boxes. This parameter must be a 0D floating-point tensor.
- **`pad_to_max_output_size`** (`bool`): Optional. If set to `True`, the output `selected_indices` is padded to a length of `max_output_size`. The default value is `False`.

## Return Values

- **`selected_indices`** (`Tensor`): Indices selected from the `boxes` tensor. This parameter must be a 1D integer tensor with shape `(M,)`, where $M \le \text{max\_output\_size}$.
- **`valid_outputs`** (`Tensor`): Number of valid elements in `selected_indices`. This parameter must be a 0D integer tensor, and valid elements appear first.

## Example

```python
>>> import torch
>>> import torch_npu
>>> boxes=torch.randn(100,4).npu()
>>> scores=torch.randn(100).npu()
>>> boxes.uniform_(0,100)
>>> scores.uniform_(0,1)
>>> max_output_size = 20
>>> iou_threshold = torch.tensor(0.5).npu()
>>> scores_threshold = torch.tensor(0.3).npu()
>>> npu_output = torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold)
>>> print(npu_output)
(tensor([57, 65, 25, 45, 43, 12, 52, 91, 23, 78, 53, 11, 24, 62, 22, 67,  9, 94,
        54, 92], device='npu:0', dtype=torch.int32), tensor(20, device='npu:0', dtype=torch.int32))
```
