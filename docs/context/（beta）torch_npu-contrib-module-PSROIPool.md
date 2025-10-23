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

```
torch_npu.contrib.module.PSROIPool(nn.Module)
```

## 参数说明

- **pooled_height** (`int`) - 池化高度。
- **pooled_width** (`int`) - 池化宽度。
- **spatial_scale** (`float`) - 按此参数值缩放输入框。
- **group_size** (`int`) - 编码位置敏感分数图的组数。
- **output_dim** (`int`) - 输出通道数。

## 返回值说明

`float` 

shape为(k, 5)和(k, 1)的张量。标签以0为基础。

## 约束说明

仅实现了pooled_height == pooled_width == group_size。


## 调用示例

```python
>>> from torch_npu.contrib.module import PSROIPool
>>> model = PSROIPool(pooled_height=7, pooled_width=7, spatial_scale=1 / 16.0, group_size=7, output_dim=22)
```

