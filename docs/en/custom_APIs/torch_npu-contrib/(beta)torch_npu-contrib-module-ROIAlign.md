# (beta) torch_npu.contrib.module.ROIAlign

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Performs a region of interest alignment (ROIAlign) operation using the NPU API.

## Prototype

```python
torch_npu.contrib.module.ROIAlign(output_size, spatial_scale, sampling_ratio, aligned=True)
```

## Parameters

**Computation Parameters**

- **`output_size`** (`Tuple[int, int]`): Output size `(height, width)` representing the target shape.
- **`spatial_scale`** (`float`): Scale factor applied to the input bounding boxes.
- **`sampling_ratio`** (`int`): Number of input samples to take for each output sample. The value `0` indicates dense sampling.
- **`aligned`** (`bool`): If set to `False`, the original Detectron implementation is used. If set to `True`, the results are aligned more accurately.

    > [!NOTE]  
    > Meaning of `aligned=True`:
    > Given a continuous coordinate `c`, the two neighboring pixel indices in the pixel model are computed using `floor(c - 0.5)` and `ceil(c - 0.5)`. For example, `c = 1.3` has neighboring pixels with discrete indices `[0]` and `[1]`, corresponding to samples of the underlying signal over the continuous coordinate range `[0.5, 1.5]`. However, the original ROIAlign (`aligned=False`) does not subtract `0.5` when computing neighboring pixel indices. As a result, bilinear interpolation is performed on pixels that are slightly misaligned with respect to the pixel model. When `aligned=True`, the ROI is first scaled appropriately and then shifted by `-0.5` before ROIAlign is applied. This produces the correct neighboring pixels. For verification details, see [detectron2/tests/test_roi_align.py](https://github.com/facebookresearch/detectron2/blob/v0.2/tests/layers/test_roi_align.py). When ROIAlign is used together with convolution layers, this difference does not affect model performance.

**Computation Input**

- **`input_tensor`** (`Tensor`): Input tensor. The data layout must be `NCHW`.
- **`rois`** (`Tensor`): ROI bounding boxes. This parameter must be 2D with shape (N, 5). The first column contains the ROI index, and the remaining four columns contain the ROI coordinates.

## Return Values

`Tensor` 

ROIAlign computation result.

## Example

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import ROIAlign
>>> input1 = torch.FloatTensor([[[[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24], [25, 26, 27, 28, 29, 30], [31, 32, 33, 34, 35, 36]]]]).npu()
>>> roi = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
>>> output_size = (3, 3)
>>> spatial_scale = 0.25
>>> sampling_ratio = 2
>>> aligned = False
>>> input1.requires_grad = True
>>> roi.requires_grad = True
>>> model = ROIAlign(output_size, spatial_scale, sampling_ratio, aligned=aligned).npu()
>>> output = model(input1, roi)
>>> output.sum().backward()
>>> print(output.shape)
torch.Size([1, 1, 3, 3])
```
