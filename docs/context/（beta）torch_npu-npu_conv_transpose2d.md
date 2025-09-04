# （beta）torch_npu.npu_conv_transpose2d

>**须知：**<br>
>该接口计划废弃，可以使用torch.nn.functional.conv_transpose2d接口进行替换。

## 函数原型

```
torch_npu.npu_conv_transpose2d(input, weight, bias, padding, output_padding, stride, dilation, groups) -> Tensor
```

## 功能说明

在由多个输入平面组成的输入图像上应用一个2D转置卷积算子，有时这个过程也被称为“反卷积”。

## 参数说明

- input (Tensor) - shape的输入张量，值为(minibatch, in_channels, iH, iW)。
- weight (Tensor) - shape的过滤器，值为(in_channels, out_channels/groups, kH, kW)。
- bias (Tensor，可选) - shape的偏差(out_channels)。
- padding (ListInt) - (dilation \* (kernel_size - 1) - padding)用零来填充输入每个维度的两侧。
- output_padding (ListInt) - 添加到输出shape每个维度一侧的附加尺寸。
- stride (ListInt) - 卷积核步长。
- dilation (ListInt) - 内核元素间距。
- groups (Int) - 对输入进行分组。In_channels可被组数整除。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

