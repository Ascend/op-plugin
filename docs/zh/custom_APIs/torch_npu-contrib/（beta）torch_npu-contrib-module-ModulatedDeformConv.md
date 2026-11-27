# （beta）torch_npu.contrib.module.ModulatedDeformConv

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id4 -->

## 功能说明

应用基于NPU的Modulated Deformable 2D卷积操作。

## 函数原型

```python
torch_npu.contrib.module.ModulatedDeformConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True, pack=True)
```

## 参数说明

**计算参数**

- **in_channels** (`int`)：必选参数，输入图像中的通道数。
- **out_channels** (`int`)：必选参数，卷积产生的通道数。
- **kernel_size** (`int`或`Tuple`)：必选参数，卷积核大小。
- **stride** (`int`)：可选参数，卷积步长。默认值为1。
- **padding** (`int`)：可选参数，添加到输入两侧的零填充。默认值为0。
- **dilation** (`int`)：可选参数，内核元素间距。默认值为1。
- **groups** (`int`)：可选参数，从输入通道到输出通道的阻塞连接数。默认值为1。
- **deformable_groups** (`int`)：可选参数，可变形分组的数量。默认值为1。
- **bias** (`bool`)：可选参数，如果值为True，则向输出添加可学习偏置。默认值为True。
- **pack** (`bool`)：默认值为`True`，模块内部使用`conv_offset`卷积生成偏移与掩码。取值为`False`时，调用者传入偏移与掩码。

**计算输入**

- **x** (`Tensor | tuple[Tensor, Tensor, Tensor]`)：当`pack=True`时为形状 `(N, C, H, W)`的输入张量。当`pack=False`时为 `(input, offset, mask)`。
- **offset** (`Tensor`)：每个卷积采样点的二维位移，形状为 `(N, 2 * K, H_out, W_out)`。
- **mask** (`Tensor`)：每个采样点的调制权重，形状为 `(N, K, H_out, W_out)`。内部生成的掩码经sigmoid映射到 `[0, 1]`。

其中`K = deformable_groups * kernel_height * kernel_width`，`H_out`和`W_out`为输出空间尺寸。

## 返回值说明

`Tensor`

卷积计算结果。

## 约束说明

ModulatedDeformConv仅实现`torch.float32`数据类型的操作。`pack=True`时，模块将内部卷积`conv_offset`的权重和偏置初始化为0。

## 调用示例

示例对批大小为2、通道数为32、空间大小为5×5的输入执行可变形卷积。模块使用1×1卷积核，在内部生成采样偏移和调制掩码。

```python
import torch, torch_npu
from torch_npu.contrib.module import ModulatedDeformConv
m = ModulatedDeformConv(32, 32, 1).npu()
input_tensor = torch.randn(2, 32, 5, 5).npu()
output = m(input_tensor)
print(output.shape)
```

示例输出如下。

```text
torch.Size([2, 32, 5, 5])
```
