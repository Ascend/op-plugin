# （beta）torch_npu.npu_bounding_box_encode

## 函数原型

```
torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, means0, means1, means2, means3, stds0, stds1, stds2, stds3) -> Tensor
```

## 功能说明

计算标注框和ground truth真值框之间的坐标变化。自定义Faster R-CNN算子。

## 参数说明

- anchor_box (Tensor) - 输入张量。锚点框。shape为（N,4）数据类型为float32的2D张量。“N”表示标注框的数量，“4”表示“x0”、“x1”、“y0”和“y1”。
- ground_truth_box (Tensor) - 输入张量。真值框。shape为（N,4）数据类型为float32的2D张量。“N”表示标注框的数量，“4”表示“x0”、“x1”、“y0”和“y1”。
- means0 (Float，默认值为0) - “x0”的偏差值。
- means1 (Float，默认值为0) - “y0”的偏差值。
- means2 (Float，默认值为0) - “x1”的偏差值。
- means3 (Float，默认值为0) - “y1”的偏差值。
- stds0 (Float，默认值为1.0) - “x0”的缩放值。
- stds1 (Float，默认值为1.0) - “y0”的缩放值。
- stds2 (Float，默认值为1.0) - “x1”的缩放值。
- stds3 (Float，默认值为1.0) - “y1”的缩放值。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> anchor_box = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
>>> ground_truth_box = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
>>> output = torch_npu.npu_bounding_box_encode(anchor_box, ground_truth_box, 0, 0, 0, 0, 0.1, 0.1, 0.2, 0.2)
>>> output
tensor([[13.3281, 13.3281,  0.0000,  0.0000],
        [13.3281,  6.6641,  0.0000,     nan]], device='npu:0')
```

