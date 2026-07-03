# (beta) torch_npu.npu_anchor_response_flags

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

* Description: Generates anchor response flags in a single feature map to identify which anchors participate in training or inference.

* Equivalent computation logic:

    The `anchor_response_flags_golden` operator can be used as an equivalent replacement for `torch_npu.npu_anchor_response_flags`. The computation logic of the two operators is identical.

    ```python
    import torch
    
    def anchor_response_flags_golden(self, featmap_size, strides, num_base_anchors):
        feat_h, feat_w = featmap_size
        gt_bboxes_cx = (self[:, 0] + self[:, 2]) * 0.5
        gt_bboxes_cy = (self[:, 1] + self[:, 3]) * 0.5
        try:
            gt_bboxes_grid_x = torch.floor(gt_bboxes_cx / strides[0]).int()
            gt_bboxes_grid_y = torch.floor(gt_bboxes_cy / strides[1]).int()
        except ZeroDivisionError:
            print("There is 0 in strides.")
        gt_bboxes_grid_idx = gt_bboxes_grid_y * feat_w + gt_bboxes_grid_x
        responsible_grid = torch.zeros(feat_h * feat_w, dtype=torch.uint8)
        gt_bboxes_grid_idx = gt_bboxes_grid_idx.long()
        responsible_grid[gt_bboxes_grid_idx] = 1
        responsible_grid = (
            responsible_grid[:, None]
            .expand(responsible_grid.size(0), num_base_anchors)
            .contiguous()
            .view(-1)
        )
        return responsible_grid
    
    ```

## Prototype

```python
torch_npu.npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors) -> Tensor
```

## Parameters

- **`self`** (`Tensor`): Required. Ground-truth bounding boxes. This parameter must be a 2D tensor with shape `[batch, 4]`.
- **`featmap_size`** (`List[int]`): Required. Feature map size. The length must be 2.
- **`stride`** (`List[int]`): Required. Strides of current axes. The length must be 2.
- **`num_base_anchors`** (`int`): Required. Number of base anchors.

## Return Values

`Tensor`

Output tensor containing the anchor response flags.

## Example

```python
>>> import torch, torch_npu
>>> x = torch.rand(100, 4).npu()
>>> y = torch_npu.npu_anchor_response_flags(x, [60, 60], [2, 2], 9)
>>> print(y.shape)
torch.Size([32400])
```
