# （beta）torch_npu.npu_bounding_box_decode

## 函数原型

```
torch_npu.npu_bounding_box_decode(rois, deltas, means0, means1, means2, means3, stds0, stds1, stds2, stds3, max_shape, wh_ratio_clip) -> Tensor
```

## 功能说明

根据rois和deltas生成标注框。自定义Faster R-CNN算子。

## 参数说明

- rois (Tensor) - 区域候选网络（RPN）生成的region of interests（ROI）。shape为（N,4）数据类型为float32或float16的2D张量。“N”表示ROI的数量， “4”表示“x0”、“x1”、“y0”和“y1”。
- deltas (Tensor) - RPN生成的ROI和真值框之间的绝对变化。shape为（N,4）数据类型为float32或float16的2D张量。“N”表示区域数，“4”表示“dx”、“dy”、“dw”和“dh”。
- means0 (Float) - index。
- means1 (Float) - index。
- means2 (Float) - index。
- means3 (Float，默认值为[0,0,0,0]) - index。"deltas" = "deltas" x "stds" + "means"。
- stds0 (Float) - index。
- stds1 (Float) - index。
- stds2 (Float) - index。
- stds3 (Float，默认值：[1.0,1.0,1.0,1.0]) - index。"deltas" = "deltas" x "stds" + "means"。
- max_shape (ListInt of length 2) - shape[h, w]，指定传输到网络的图像大小。用于确保转换后的bbox shape不超过“max_shape”。
- wh_ratio_clip (Float) -“dw”和“dh”的值在(-wh_ratio_clip, wh_ratio_clip)范围内。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> rois = torch.tensor([[1., 2., 3., 4.], [3.,4., 5., 6.]], dtype = torch.float32).to("npu")
>>> deltas = torch.tensor([[5., 6., 7., 8.], [7.,8., 9., 6.]], dtype = torch.float32).to("npu")
>>> output = torch_npu.npu_bounding_box_decode(rois, deltas, 0, 0, 0, 0, 1, 1, 1, 1, (10, 10), 0.1)
>>> output
tensor([[2.5000, 6.5000, 9.0000, 9.0000],
        [9.0000, 9.0000, 9.0000, 9.0000]], device='npu:0')
```

