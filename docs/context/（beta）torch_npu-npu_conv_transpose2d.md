# （beta）torch_npu.npu_conv_transpose2d

>**须知：**<br>
>该接口计划废弃，可以使用`torch.nn.functional.conv_transpose2d`接口进行替换。

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

```
torch_npu.npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：shape的输入张量，值为(minibatch, in_channels, iH, iW)。
- **weight**（`Tensor`）：shape过滤器，值为(in_channels, out_channels/groups, kH, kW)。
- **bias**（`Tensor`）：可选参数，shape偏差(out_channels)。
- **padding**（`List[int]`）：(dilation \* (kernel_size - 1) - padding)用零来填充输入每个维度的两侧。
- **output_padding**（`List[int]`）：添加到输出shape每个维度一侧的附加尺寸。
- **stride**（`List[int]`）：卷积核步长。
- **dilation**（`List[int]`）：内核元素间距。
- **groups**（`int`）：对输入进行分组。In_channels可被组数整除。