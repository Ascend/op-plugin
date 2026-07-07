# (beta) torch_npu.npu_nms_rotated

> [!NOTICE]  
> This API is planned for deprecation. For details about the replacement, see [Small Operator Concatenation Solution](https://gitcode.com/Ascend/op-plugin/blob/26.0.0/test/test_base_ops/test_nms_rotated.py).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Selects a subset of rotated bounding boxes in descending order of scores.

## Prototype

```python
torch_npu.npu_nms_rotated(dets, scores, iou_threshold, scores_threshold=0, max_output_size=-1, mode=0) -> (Tensor, Tensor)
```

## Parameters

- **`dets`** (`Tensor`): This parameter must be a 2D floating-point tensor with shape `(num_boxes, 5)`.
- **`scores`** (`Tensor`): This parameter must be a 1D floating-point tensor with shape `(num_boxes,)`, representing a score corresponding to each box.
- **`iou_threshold`** (`float`): Scalar representing the upper threshold for IoU overlap between boxes.
- **`scores_threshold`** (`float`): Scalar representing the score threshold used to decide when to remove boxes. The default value is `0`.
- **`max_output_size`** (`int`): Maximum number of boxes to select during non-maximum suppression. The default value is `-1`, indicating that no constraint is imposed.
- **`mode`** (`int`): Layout type of `dets`. The default value is `0`. If `mode` is set to `0`, the input values of `dets` are `x`, `y`, `w`, `h`, and angle. If `mode` is set to `1`, the input values of `dets` are `x1`, `y1`, `x2`, `y2`, and angle.

## Return Values

- **`selected_index`** (`Tensor`): Indices selected from the `dets` tensor. This parameter must be a 1D integer tensor with shape `(M,)`, where $M \le \text{max\_output\_size}$.
- **`selected_num`** (`Tensor`): Number of valid elements in `selected_index`. This parameter must be a 1D integer tensor.

## Constraints

Currently, scenarios where `mode=1` are not supported.

## Example

```python
>>> import torch
>>> import torch_npu
>>> dets=torch.randn(100,5).npu()
>>> scores=torch.randn(100).npu()
>>> dets.uniform_(0,100)
>>> scores.uniform_(0,1)
>>> output1, output2 = torch_npu.npu_nms_rotated(dets, scores, 0.2, 0, -1, 0)
>>> print(output1)
tensor([76, 48, 15, 65, 91, 82, 21, 96, 62, 90, 13, 59,  0, 18, 47, 23,  8, 56,
        55, 63, 72, 39, 97, 81, 16, 38, 17, 25, 74, 33, 79, 44, 36, 88, 83, 37,
        64, 45, 54, 41, 22, 28, 98, 40, 30, 20,  1, 86, 69, 57, 43,  9, 42, 27,
        71, 46, 19, 26, 78, 66,  3, 52], device='npu:0', dtype=torch.int32)
>>> print(output2)
tensor([62], device='npu:0', dtype=torch.int32)
```
