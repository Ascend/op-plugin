# （beta）torch_npu.contrib.module.PSROIPool

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

使用NPU API进行PSROIPool。

## 函数原型

```python
torch_npu.contrib.module.PSROIPool(nn.Module)
```

## 参数说明

**计算参数**

- **pooled_height** (`int`) - 池化高度。
- **pooled_width** (`int`) - 池化宽度。
- **spatial_scale** (`float`) - 按此参数值缩放输入框。
- **group_size** (`int`) - 编码位置敏感分数图的组数。
- **output_dim** (`int`) - 输出通道数。

**计算输入**

- **features** (`Tensor`) - 输入特征图。
- **rois** (`Tensor`) - ROI信息张量，维度不少于3，其中第1维大小为5。

## 返回值说明

`Tensor` 

PSROIPool计算结果，shape为(`rois.size(0) * rois.size(2)`, `output_dim`, `pooled_height`, `pooled_width`)。

## 约束说明

仅实现了pooled_height == pooled_width == group_size。

## 调用示例

```python
>>> from torch_npu.contrib.module import PSROIPool
>>> model = PSROIPool(pooled_height=7, pooled_width=7, spatial_scale=1 / 16.0, group_size=7, output_dim=22)
```
