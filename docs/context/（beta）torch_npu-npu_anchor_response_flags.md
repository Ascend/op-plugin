# （beta）torch_npu.npu_anchor_response_flags

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

* API功能：在单个特征图中生成锚点的响应标志，即标识哪些锚点需要参与训练或推理。

* 等价计算逻辑：

    可使用`anchor_response_flags_golden`等价替换`torch_npu.npu_anchor_response_flags`，两者计算逻辑一致。
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

## 函数原型

```
torch_npu.npu_anchor_response_flags(self, featmap_size, stride, num_base_anchors) -> Tensor
```

## 参数说明

- **self** (`Tensor`)：真值框，shape为`[batch, 4]`的2D张量。
- **featmap_size** (`List[int]`)：特征图大小，长度为2。
- **stride** (`List[int]`)：当前轴的步长，长度为2。
- **num_base_anchors** (`int`)：base anchors的数量。

## 返回值说明
`Tensor`

返回响应标志结果。

## 调用示例

```python
>>> import torch, torch_npu
>>> x = torch.rand(100, 4).npu()
>>> y = torch_npu.npu_anchor_response_flags(x, [60, 60], [2, 2], 9)
>>> y.shape
torch.Size([32400])
```

