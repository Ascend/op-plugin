# （beta）torch_npu.npu_roi_align

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

从特征图中获取ROI特征矩阵。自定义Faster R-CNN算子。

## 函数原型

```python
torch_npu.npu_roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode) -> Tensor
```

## 参数说明

- **features** (`Tensor`)：必选参数，表示待处理的特征图，要求4维张量。
- **rois** (`Tensor`)：必选参数，表示ROI位置，支持shape为(N, 5)的2维张量。“N”表示ROI的数量，“5”表示ROI所在图像的index及ROI坐标，分别为“index”、“x0”、“y0”、“x1”和“y1”。
- **spatial_scale** (`float`)：必选参数，指定`features`与原始图像的缩放比率。数据类型支持`float32`。
- **pooled_height** (`int`)：必选参数，指定输出图像的高度。数据类型支持`int32`。
- **pooled_width** (`int`)：必选参数，指定输出图像的宽度。数据类型支持`int32`。
- **sample_num** (`int`)：必选参数，指定每个池化单元在高、宽两个方向上的采样点数。取值为`0`时，根据缩放后的ROI尺寸和池化输出尺寸自适应确定采样点数。数据类型支持`int32`。
- **roi_end_mode** (`int`)：必选参数，指定ROI结束边界的坐标处理方式。数据类型支持`int32`。

## 约束说明

算子实现与NumPy实现一致；但相较于CUDA实现，该算子的反向计算存在精度差异。

## 调用示例

```python
>>> import torch
>>> import torch_npu
>>> x = torch.FloatTensor([[[[1, 2, 3 , 4, 5, 6],
                            [7, 8, 9, 10, 11, 12],
                            [13, 14, 15, 16, 17, 18],
                            [19, 20, 21, 22, 23, 24],
                            [25, 26, 27, 28, 29, 30],
                            [31, 32, 33, 34, 35, 36]]]]).npu()
>>> rois = torch.tensor([[0, -2.0, -2.0, 22.0, 22.0]]).npu()
>>> out = torch_npu.npu_roi_align(x, rois, 0.25, 3, 3, 2, 0)
>>> print(out)
tensor([[[[ 4.5000,  6.5000,  8.5000],
          [16.5000, 18.5000, 20.5000],
          [28.5000, 30.5000, 32.5000]]]], device='npu:0')
```
