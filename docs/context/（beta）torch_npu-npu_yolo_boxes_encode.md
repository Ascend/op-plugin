# （beta）torch_npu.npu_yolo_boxes_encode

## 函数原型

```
torch_npu.npu_yolo_boxes_encode(self, gt_bboxes, stride, performance_mode=False) -> Tensor
```

## 功能说明

根据YOLO的锚点框（anchor box）和真值框（ground-truth box）生成标注框。自定义mmdetection算子。

## 参数说明

- self (Tensor) -  YOLO训练集生成的锚点框。shape为(N, 4)数据类型为float32或float16的2D张量。“N”表示ROI的数量，值“4”表示(tx, ty, tw, th)。
- gt_bboxes (Tensor) - 转换目标，例如真值框。shape为(N, 4)数据类型为float32或float16的2D张量。“N”表示ROI的数量，值“4”表示“dx”、“dy”、“dw”和“dh”。
- strides (Tensor) - 各框比例。shape为(N,)数据类型为int32的1D张量。“N”表示ROI的数量。
- performance_mode (Bool，默认值为False) - 选择性能模式为“high_precision”或“high_performance”。如果值为True，则性能模式为“high_performance”；如果值为False，则性能模式为“high_precision”。当输入数据类型为float32时，选择“high_precision”，输出张量精度将小于0.0001。当输入数据类型为float16时，选择“high_performance”，ops将是最佳性能，但精度将只小于0.005。

## 约束说明

输入锚点框支持的最大N为20480。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> anchor_boxes = torch.rand(2, 4).npu()
>>> gt_bboxes = torch.rand(2, 4).npu()
>>> stride = torch.tensor([2, 2], dtype=torch.int32).npu()
>>> output = torch_npu.npu_yolo_boxes_encode(anchor_boxes, gt_bboxes, stride, False)
>>> output.shape
torch.Size([2, 4])
```

