# （beta）torch_npu.npu_conv2d

>**须知：**<br>
>该接口计划废弃，可以使用torch.nn.functional.conv2d接口进行替换。

## 函数原型

```
torch_npu.npu_conv2d(input, weight, bias, stride, padding, dilation, groups) -> Tensor
```

## 功能说明

在由多个输入平面组成的输入图像上应用一个2D卷积。

## 参数说明

- input (Tensor) - shape的输入张量，值为(minibatch, in_channels, iH, iW)。
- weight (Tensor) - shape过滤器，值为(out_channels, in_channels/groups, kH, kW)。
- bias (Tensor，可选) - shape偏差(out_channels)。
- stride (ListInt) - 卷积核步长。
- padding (ListInt) - 输入两侧的隐式填充。
- dilation (ListInt) - 内核元素间距。
- groups (Int) - 对输入进行分组。in_channels可被组数整除。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

