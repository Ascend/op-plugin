# (beta) torch_npu.npu_ps_roi_pooling

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

Performs position-sensitive ROI pooling.

## Prototype

```python
torch_npu.npu_ps_roi_pooling(x, rois, spatial_scale, group_size, output_dim) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. `NC1HWC0` tensor describing the feature map. The dimension `C1` must be equal to `(int(output_dim + 15) / C0) * group_size`.
- **`rois`** (`Tensor`): ROIs. The shape of this parameter is `(batch, 5, rois_num)`. Each RoI consists of five elements: `batch_id`, `x1`, `y1`, `x2`, and `y2`, where `batch_id` indicates the index of the input feature map, and `x1`, `y1`, `x2`, and `y2` must be greater than or equal to `0.0`.
- **`spatial_scale`** (`float`): Scaling coefficient used to map input coordinates to RoI coordinates. The data type can be `float32`.
- **`group_size`** (`int`): Number of groups used to encode the position-sensitive score maps. The value must be within the range `(0, 128)`. The data type can be `int32`.
- `output_dim` (`int`): Number of output channels. The value must be greater than 0. The data type can be `int32`.

## Example

```python
>>> import torch
>>> import torch_npu
>>> roi = torch.tensor([[[1], [2], [3], [4], [5]],
                        [[6], [7], [8], [9], [10]]], dtype = torch.float16).npu()
>>> x = torch.tensor([[[[ 1]], [[ 2]], [[ 3]], [[ 4]],
                      [[ 5]], [[ 6]], [[ 7]], [[ 8]]],
                      [[[ 9]], [[10]], [[11]], [[12]],
                      [[13]], [[14]], [[15]], [[16]]]], dtype = torch.float16).npu()
>>> out = torch_npu.npu_ps_roi_pooling(x, roi, 0.5, 2, 2)
>>> print(out)
tensor([[[[0., 0.],
          [0., 0.]],
        [[0., 0.],
          [0., 0.]]],
        [[[0., 0.],
          [0., 0.]],
        [[0., 0.],
          [0., 0.]]]], device='npu:0', dtype=torch.float16)
```
