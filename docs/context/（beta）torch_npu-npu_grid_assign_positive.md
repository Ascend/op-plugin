# （beta）torch_npu.npu_grid_assign_positive

## 函数原型

```
torch_npu.npu_grid_assign_positive(self, overlaps, box_responsible_flags, max_overlaps, argmax_overlaps, gt_max_overlaps, gt_argmax_overlaps, num_gts, pos_iou_thr, min_pos_iou, gt_max_assign_all) -> Tensor
```

## 功能说明

执行position-sensitive的候选区域池化梯度计算。

## 参数说明

- self (Tensor) - float16或float32类型的张量，shape为(n, )。
- overlaps (Tensor) - 数据类型与assigned_gt_inds相同，表示gt_bboxes和bboxes之间的IoU，shape为(k,n)。
- box_responsible_flags (Tensor) - 支持uint8数据类型。表示框是否responsible的标志。
- max_overlaps (Tensor) - 数据类型与assigned_gt_inds.overlaps.max(axis=0)相同。
- argmax_overlaps (Tensor) - 支持uint32数据类型，overlaps.argmax(axis=0)。
- gt_max_overlaps (Tensor) - 数据类型与assigned_gt_inds.overlaps.max(axis=1)相同。
- gt_argmax_overlaps (Tensor) - 支持uint32数据类型， overlaps.argmax(axis=1)。
- num_gts (Tensor) - 支持uint32数据类型，real k，shape为(1, )。
- pos_iou_thr (Float) - 正检测框的IoU阈值。
- min_pos_iou (Float) - 检测框被视为正检测框的最小IoU。
- gt_max_assign_all (Bool) - 是否将与某个gt有相同最高重叠的所有检测框分配给该gt。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> assigned_gt_inds = torch.rand(4).npu()
>>> overlaps = torch.rand(2,4).npu()
>>> box_responsible_flags = torch.tensor([1, 1, 1, 0], dtype=torch.uint8).npu()
>>> max_overlap = torch.rand(4).npu()
>>> argmax_overlap = torch.tensor([1, 0, 1, 0], dtype=torch.int32).npu()
>>> gt_max_overlaps = torch.rand(2).npu()
>>> gt_argmax_overlaps = torch.tensor([1, 0],dtype=torch.int32).npu()
>>> output = torch_npu.npu_grid_assign_positive(assigned_gt_inds, overlaps, box_responsible_flags, max_overlap, argmax_overlap, gt_max_overlaps, gt_argmax_overlaps, 128, 0.5, 0., True)
>>> output.shape
torch.Size([4])
```

