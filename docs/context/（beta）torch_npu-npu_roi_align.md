# （beta）torch_npu.npu_roi_align

## 函数原型

```
torch_npu.npu_roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode) -> Tensor
```

## 功能说明

从特征图中获取ROI特征矩阵。自定义Faster R-CNN算子。

## 参数说明

- features (Tensor) - 5D张量。
- rois (Tensor) - ROI位置，shape为(N, 5)的2D张量。“N”表示ROI的数量，“5”表示ROI所在图像的index，分别为“x0”、“y0”、“x1”和“y1”。
- spatial_scale (Float32) - 指定“features”与原始图像的缩放比率。
- pooled_height (Int32) - 指定H维度。
- pooled_width (Int32) - 指定W维度。
- sample_num (Int32，默认值为2) - 指定每次输出的水平和垂直采样频率。若此属性设置为0，则采样频率等于“rois”的向上取整值（一个浮点数）。
- roi_end_mode (Int32，默认值为1)。

## 约束说明

该算子实现与numpy实现一致，跟cuda比反向存在精度问题。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.FloatTensor([[[[1, 2, 3 , 4, 5, 6],
                            [7, 8, 9, 10, 11, 12],
                            [13, 14, 15, 16, 17, 18],
                            [19, 20, 21, 22, 23, 24],
                            [25, 26, 27, 28, 29, 30],
                            [31, 32, 33, 34, 35, 36]]]]).npu()
>>> rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
>>> out = torch_npu.npu_roi_align(x, rois, 0.25, 3, 3, 2, 0)
>>> out
tensor([[[[ 4.5000,  6.5000,  8.5000],
          [16.5000, 18.5000, 20.5000],
          [28.5000, 30.5000, 32.5000]]]], device='npu:0')
```

