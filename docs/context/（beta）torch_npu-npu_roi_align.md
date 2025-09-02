# （beta）torch_npu.npu_roi_align

# 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term          |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>  | √   |
|<term>Atlas 训练系列产品</term>  | √   |

## 功能说明

从特征图中获取ROI特征矩阵。自定义Faster R-CNN算子。

## 函数原型

```
torch_npu.npu_roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode) -> Tensor
```

## 参数说明

- **features** (`Tensor`)：必选参数，表示待处理的特征图，要求5维张量。
- **rois** (`Tensor`)：必选参数，表示ROI位置，支持shape为(N, 5)的2维张量。“N”表示ROI的数量，“5”表示ROI所在图像的index，分别为“x0”、“y0”、“x1”和“y1”。
- **spatial_scale** (`float`)：必选参数，指定`features`与原始图像的缩放比率。数据类型支持`float32`。
- **pooled_height** (`int`)：必选参数，指定输出图像的高度。数据类型支持`int32`。
- **pooled_width** (`int`)：必选参数，输出图像的宽度。数据类型支持`int32`。
- **sample_num** (`int`)： 必选参数，默认值为2。指定每个输出元素在H和W方向上的采样频率。若此属性设置为0，则采样频率等于“rois”的向上取整值（一个浮点数）。数据类型支持`int32`。
- **roi_end_mode** (`int`)：必选参数，默认值为1。数据类型支持`int32`。

## 约束说明

该算子实现与numpy实现一致，相较于cuda反向存在精度问题。

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

