# （beta）torch_npu.npu_deformable_conv2d

## 函数原型

```
torch_npu.npu_deformable_conv2d(self, weight, offset, bias, kernel_size, stride, padding, dilation=[1,1,1,1], groups=1, deformable_groups=1, modulated=True) -> (Tensor, Tensor)
```

## 功能说明

使用预期输入计算变形卷积输出（deformable convolution output）。

## 参数说明

- self (Tensor) - 输入图像的4D张量。格式为“NHWC”，数据按以下顺序存储：[batch, in_height, in_width, in_channels]。
- weight (Tensor) - 可学习过滤器的4D张量。数据类型需与self相同。格式为“HWCN”，数据按以下顺序存储：[filter_height, filter_width, in_channels / groups, out_channels]。
- offset (Tensor) - x-y坐标偏移和掩码的4D张量。格式为“NHWC”，数据按以下顺序存储：[batch, out_height, out_width, deformable_groups \* filter_height \* filter_width \* 3]。
- bias (Tensor，可选) - 过滤器输出附加偏置（additive bias）的1D张量，数据按[out_channels]的顺序存储。
- kernel_size (ListInt of length 2) - 内核大小，2个整数的元组/列表。
- stride (ListInt) - 4个整数的列表，表示每个输入维度的滑动窗口步长。维度顺序根据self的数据格式解释。N维和C维必须设置为1。
- padding (ListInt) - 4个整数的列表，表示要添加到输入每侧（顶部、底部、左侧、右侧）的像素数。
- dilations (ListInt，默认值为[1, 1, 1, 1]) - 4个整数的列表，表示输入每个维度的膨胀系数（dilation factor）。维度顺序根据self的数据格式解释。N维和C维必须设置为1。
- groups (Int，默认值为1) - int32类型，表示从输入通道到输出通道的分组数。in_channels和out_channels需都可被“groups”数整除。
- deformable_groups (Int，默认值为1) - int32类型，表示可变形组分区的数量。in_channels需可被“deformable_groups”数整除。
- modulated (Bool，可选，默认值为True) - 指定DeformableConv2D版本。True表示v2版本，False表示v1版本，目前仅支持v2。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.rand(16, 32, 32, 32).npu()
>>> weight = torch.rand(32, 32, 5, 5).npu()
>>> offset = torch.rand(16, 75, 32, 32).npu()
>>> output, _ = torch_npu.npu_deformable_conv2d(x, weight, offset, None, kernel_size=[5, 5], stride = [1, 1, 1, 1], padding = [2, 2, 2, 2])
>>> output.shape
torch.Size([16, 32, 32, 32])
```

