# （beta）torch_npu.contrib.module.ModulatedDeformConv
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

应用基于NPU的Modulated Deformable 2D卷积操作。

## 函数原型

```
torch_npu.contrib.module.ModulatedDeformConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True, pack=True)
```

## 参数说明
**计算参数**

- **in_channels** (`int`)：输入图像中的通道数。
- **out_channels** (`int`)：卷积产生的通道数。
- **kernel_size**(`int`或`Tuple`)：卷积核大小。
- **stride**(`int`或`Tuple`)：卷积步长。默认值为1。
- **padding** (`int`或`Tuple`)：添加到输入两侧的零填充。默认值为0。
- **dilation** (`int`或`Tuple`)：内核元素间距。默认值为1。
- **groups** (`int`)：从输入通道到输出通道的阻塞连接数。默认值为1。
- **deform_groups** (`int`)：可变形组分区的数量。
- **bias** (`bool`)：如果值为True，则向输出添加可学习偏差。默认值为False。
- **pack** (`bool`)：如果值为True，此模块将包括conv_offset和掩码。默认值为True。

**计算输入**

- **x**(`Tensor`): 输入张量。

## 返回值说明
`Tensor`

卷积计算结果。

## 约束说明

ModulatedDeformConv仅实现float32数据类型的操作。conv_offset中权重和偏置必须初始化为0。


## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import ModulatedDeformConv
>>> m = ModulatedDeformConv(32, 32, 1).npu()
>>> input_tensor = torch.randn(2, 32, 5, 5).npu()
>>> output = m(input_tensor)
>>> output.shape
torch.Size([2, 32, 5, 5])
```

