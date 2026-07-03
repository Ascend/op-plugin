# (beta) torch_npu.npu_roi_align

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>         |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas inference products</term> | √   |
|<term>Atlas training products</term> | √   |

## Function

Obtains the candidate region feature matrix from a feature map. This is a custom Faster R-CNN operator.

## Prototype

```python
torch_npu.npu_roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode) -> Tensor
```

## Parameters

- **`features`** (`Tensor`): Required. Feature map to be processed. This parameter must be a 4D tensor.
- **`rois`** (`Tensor`): Required. RoI locations. This parameter must be a 2D tensor with shape `(N, 4)`. `N` indicates the number of RoIs, and `4` indicates the index of the image where the RoI is located, which are `x0`, `y0`, `x1`, and `y1` respectively.
- **`spatial_scale`** (`float`): Required. Scaling ratio between `features` and the original image. The data type can be `float32`.
- **`pooled_height`** (`int`): Required. Height of the output image. The data type can be `int32`.
- **`pooled_width`** (`int`): Required. Width of the output image. The data type can be `int32`.
- **`sample_num`** (`int`): Required. Sampling frequency of each output element in the H and W directions. The default value is `2`. If set to `0`, the sampling frequency is equal to the ceiling value of `rois` (a floating-point number). The data type can be `int32`.
- **`roi_end_mode`** (`int`): Required. The default value is `1`. The data type can be `int32`.

## Constraints

This operator is implemented consistently with NumPy. Compared with the CUDA implementation, its backward computation has precision issues.

## Example

```python
>>> import torch
>>> import torch_npu
>>> x = torch.FloatTensor([[[[1, 2, 3 , 4, 5, 6],
                            [7, 8, 9, 10, 11, 12],
                            [13, 14, 15, 16, 17, 18],
                            [19, 20, 21, 22, 23, 24],
                            [25, 26, 27, 28, 29, 30],
                            [31, 32, 33, 34, 35, 36]]]]).npu()
>>> rois = torch.tensor([[0, -2.0, -2.0, 22.0]]).npu()
>>> out = torch_npu.npu_roi_align(x, rois, 0.25, 3, 3, 2, 0)
>>> print(out)
tensor([[[[ 4.5000,  6.5000,  8.5000],
          [16.5000, 18.5000, 20.5000],
          [28.5000, 30.5000, 32.5000]]]], device='npu:0')
```
