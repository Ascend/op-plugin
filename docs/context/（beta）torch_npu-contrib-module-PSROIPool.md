# （beta）torch_npu.contrib.module.PSROIPool

## 函数原型

```
torch_npu.contrib.module.PSROIPool(nn.Module)
```

## 功能说明

使用NPU API进行PSROIPool。

## 参数说明

- pooled_height (Int) - 池化高度。
- pooled_width (Int) - 池化宽度。
- spatial_scale (Float) - 按此参数值缩放输入框。
- group_size (Int) - 编码位置敏感分数图的组数。
- output_dim (Int) - 输出通道数。

## 输出说明

Float - shape为(k, 5)和(k, 1)的张量。标签以0为基础。

## 约束说明

仅实现了pooled_height == pooled_width == group_size。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.module import PSROIPool
>>> model = PSROIPool(pooled_height=7, pooled_width=7, spatial_scale=1 / 16.0, group_size=7, output_dim=22)
```

