# （beta）torch_npu.npu_convolution

>**须知：**<br>
>该接口计划废弃，可以使用`torch.nn.functional.conv2d`、`torch.nn.functional.conv3d`或`torch._C._nn.slow_conv3d`接口进行替换。

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

```
torch_npu.npu_convolution(input, weight, bias, stride, padding, dilation, groups) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：shape的输入张量，值为(minibatch, in_channels, iH, iW)或(minibatch, in_channels, iT, iH, iW)。
- **weight**（`Tensor`）：shape过滤器，值为(out_channels, in_channels/groups, kH, kW)或(out_channels, in_channels/groups, kT, kH, kW)。
- **bias**（`Tensor`）：可选参数，shape偏差(out_channels)。
- **stride**（`List[int]`）：卷积核步长。
- **padding**（`List[int]`）：输入两侧的隐式填充。
- **dilation**（`List[int]`）：内核元素间距。
- **groups**（`int`）：对输入进行分组。In_channels可被组数整除。

