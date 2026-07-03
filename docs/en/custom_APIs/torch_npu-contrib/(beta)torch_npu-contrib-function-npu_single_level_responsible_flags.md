# (beta) torch_npu.contrib.function.npu_single_level_responsible_flags

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Uses an NPU operator to generate responsible flags for anchors in a single feature map.

## Prototype

```python
torch_npu.contrib.function.npu_single_level_responsible_flags(featmap_size, gt_bboxes, stride, num_base_anchors)
```

## Parameters

- **`featmap_size`** (`Tuple[int]`): Size of the feature map.
- **`gt_bboxes`** (`Tensor`): Ground truth bounding boxes.
- **`stride`** (`Tuple[int]`): Stride of the current level.
- **`num_base_anchors`** (`int`): Number of base anchors.

## Return Values

`Tensor` 

The valid flags for all anchors in the single feature map. The output has shape `(featmap_size[0] * featmap_size[1] * num_base_anchors,)`.

## Example

```python
>>> import torch
>>> from torch_npu.contrib.function import npu_single_level_responsible_flags
>>> featmap_sizes = [[10, 10], [20, 20], [40, 40]]
>>> stride = [[32, 32], [16, 16], [8, 8]]
>>> gt_bboxes = torch.randint(0, 512, size=(128, 4))
>>> num_base_anchors = 3
>>> featmap_level = len(featmap_sizes)
>>> for i in range(featmap_level):
...     gt_bboxes = gt_bboxes.npu()
...     out = npu_single_level_responsible_flags(featmap_sizes[i],gt_bboxes,stride[i],num_base_anchors)
...     print(out.shape, out.max(), out.min())
torch.Size([300]) tensor(1, device='npu:0', dtype=torch.uint8) tensor(0, device='npu:0', dtype=torch.uint8)
torch.Size([1200]) tensor(1, device='npu:0', dtype=torch.uint8) tensor(0, device='npu:0', dtype=torch.uint8)
torch.Size([4800]) tensor(1, device='npu:0', dtype=torch.uint8) tensor(0, device='npu:0', dtype=torch.uint8)

```
