# （beta）torch_npu.npu_convolution

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.nn.functional.conv2d`或`torch.nn.functional.conv3d`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

在由多个输入平面组成的输入图像上应用一个2D或3D卷积。

## 函数原型

```python
torch_npu.npu_convolution(input, weight, bias, stride, padding, dilation, groups) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：输入张量，shape为(minibatch, in_channels, iH, iW)或(minibatch, in_channels, iT, iH, iW)。
- **weight**（`Tensor`）：过滤器，shape为(out_channels, in_channels/groups, kH, kW)或(out_channels, in_channels/groups, kT, kH, kW)。
- **bias**（`Tensor`）：允许为None，偏差，shape为(out_channels)。
- **stride**（`List[int]`）：卷积核步长。
- **padding**（`List[int]`）：输入两侧的隐式填充。
- **dilation**（`List[int]`）：内核元素间距。
- **groups**（`int`）：对输入进行分组。in_channels可被组数整除。
