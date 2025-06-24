# （beta）torch_npu.npu_nms_v4

## 函数原型

```
torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold, pad_to_max_output_size=False) -> (Tensor, Tensor)
```

## 功能说明

按分数降序选择标注框的子集。

## 参数说明

- boxes (Tensor) - shape为[num_boxes, 4]的2D浮点张量。
- scores (Tensor) - shape为[num_boxes]的1D浮点张量，表示每个框（每行框）对应的一个分数。
- max_output_size (Scalar) - 表示non-max suppression下要选择的最大框数的标量。
- iou_threshold (Tensor) - 0D浮点张量，表示框与IoU重叠上限的阈值。
- scores_threshold (Tensor) - 0D浮点张量，表示决定何时删除框的分数阈值。
- pad_to_max_output_size (Bool，默认值为False) - 如果为True，则输出的selected_indices将填充为max_output_size长度。

## 输出说明

- selected_indices (Tensor) - shape为[M]的1D整数张量，表示从boxes张量中选定的index，其中M <= max_output_size。
- valid_outputs (Tensor) - 0D整数张量，表示selected_indices中有效元素的数量，有效元素首先呈现。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> boxes=torch.randn(100,4).npu()
>>> scores=torch.randn(100).npu()
>>> boxes.uniform_(0,100)
>>> scores.uniform_(0,1)
>>> max_output_size = 20
>>> iou_threshold = torch.tensor(0.5).npu()
>>> scores_threshold = torch.tensor(0.3).npu()
>>> npu_output = torch_npu.npu_nms_v4(boxes, scores, max_output_size, iou_threshold, scores_threshold)
>>> npu_output
(tensor([57, 65, 25, 45, 43, 12, 52, 91, 23, 78, 53, 11, 24, 62, 22, 67,  9, 94,
        54, 92], device='npu:0', dtype=torch.int32), tensor(20, device='npu:0', dtype=torch.int32))
```

