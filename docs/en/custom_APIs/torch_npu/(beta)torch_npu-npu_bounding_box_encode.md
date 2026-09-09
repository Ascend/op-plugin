# (beta) torch_npu.npu_bounding_box_encode

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

Computes the coordinate changes between anchor boxes and ground-truth boxes. This is a custom Faster R-CNN operator.

## Prototype

```python
torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3) -> Tensor
```

## Parameters

- **`anchor_box`** (`Tensor`): Required. Input tensor. Anchor boxes. This parameter must be a 2D tensor with shape `(N, 4)`. The data type can be `float32`. `N` indicates the number of bounding boxes, and `4` indicates `x0`, `x1`, `y0`, and `y1`.
- **`ground_truth_box`** (`Tensor`): Required. Input tensor. Ground-truth boxes. This parameter must be a 2D tensor with shape `(N, 4)`. The data type can be `float32`. `N` indicates the number of bounding boxes, and `4` indicates `x0`, `x1`, `y0`, and `y1`.
- **`means0`** (`float`): Offset value for `x0`.
- **`means1`** (`float`): Offset value for `y0`.
- **`means2`** (`float`): Offset value for `x1`.
- **`means3`** (`float`): Offset value for `y1`.
- **`stds0`** (`float`): Scaling value for `x0`.
- **`stds1`** (`float`): Scaling value for `y0`.
- **`stds2`** (`float`): Scaling value for `x1`.
- **`stds3`** (`float`): Scaling value for `y1`.

## Return Values

`Tensor`

Output tensor representing the encoded coordinate tensor with shape `(N, 4)`, where 4 indicates `x0`, `y0`, `x1`, and `y1`. The data type is `float32`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> anchor_box = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
>>> ground_truth_box = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
>>> output = torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
>>> print(output)
tensor([[13.3281, 13.3281,  0.0000,  0.0000],
        [13.3281,  6.6641,  0.0000,     nan]], device='npu:0')
```
