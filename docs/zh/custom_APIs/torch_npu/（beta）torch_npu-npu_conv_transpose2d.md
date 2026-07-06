# （beta）torch_npu.npu_conv_transpose2d

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.nn.functional.conv_transpose2d`接口进行替换。

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品</term> | √   |
|<term>Atlas 训练系列产品</term> | √   |
|<term>Atlas 推理系列产品</term>| √   |

## 功能说明

在由多个输入平面组成的输入图像上应用一个2D转置卷积算子，有时这个过程也被称为“反卷积”。

## 函数原型

```python
torch_npu.npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：必选参数，输入张量，shape为(minibatch, in_channels, iH, iW)。
- **weight**（`Tensor`）：必选参数，过滤器张量，shape为(in_channels, out_channels/groups, kH, kW)。
- **bias**（`Tensor`）：可选参数，默认为None，偏置张量，shape为(out_channels)。
- **padding**（`List[int]`）：可选参数，默认为[0,0]，用零填充输入shape两侧,填充个数为(dilation \* (kernel_size - 1) - padding)。
- **output_padding**（`List[int]`）：可选参数，默认为[0,0]，添加到输出shape每个维度一侧的附加尺寸。
- **stride**（`List[int]`）：可选参数，默认为[1,1]，卷积核步长。
- **dilation**（`List[int]`）：可选参数，默认为[1,1]，内核元素间距。
- **groups**（`int`）：可选参数，默认为1，对输入进行分组。in_channels和out_channels均可被组数整除。
