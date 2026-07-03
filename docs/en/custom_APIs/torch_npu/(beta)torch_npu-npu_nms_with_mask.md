# (beta) torch_npu.npu_nms_with_mask

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

Generates values `0` or `1`, which are used by the NMS operator to determine valid bits.

## Prototype

```python
torch_npu.npu_nms_with_mask(input, iou_threshold) -> (Tensor, Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Input tensor.
- **`iou_threshold`** (`Scalar`): Threshold. If this threshold is exceeded, the value is `1`. Otherwise, the value is `0`.

## Return Values

- **`selected_boxes`** (`Tensor`): Filtered boxes, including proposal boxes and corresponding confidence scores. This parameter must be a 2D tensor with shape `(N, 5)`.
- **`selected_idx`** (`Tensor`): Indices of the input proposal boxes. This parameter must be a 1D integer tensor with shape `(N,)`.
- **`selected_mask`** (`Tensor`): Determines whether the output proposal boxes are valid. This parameter must be a 1D tensor with shape `(N,)`.

## Constraints

The second dimension of `input` must be `8`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 0.6, 0.5, 0.4, 0.3], [6.0, 7.0, 8.0, 9.0, 0.4, 0.5, 0.6, 0.7]], dtype=torch.float16).to("npu")
>>> iou_threshold = 0.5
>>> output1, output2, output3, = torch_npu.npu_nms_with_mask(input, iou_threshold)
>>> print(output1)
tensor([[0.0000, 1.0000, 2.0000, 3.0000, 0.6001],
        [6.0000, 7.0000, 8.0000, 9.0000, 0.3999]], device='npu:0',      
        dtype=torch.float16)
>>> print(output2)
tensor([0, 1], device='npu:0', dtype=torch.int32)
>>> print(output3)
tensor([1, 1], device='npu:0', dtype=torch.uint8)
```
