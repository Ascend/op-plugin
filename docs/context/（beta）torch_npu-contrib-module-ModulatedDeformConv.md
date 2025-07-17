# （beta）torch_npu.contrib.module.ModulatedDeformConv

## 函数原型

```
torch_npu.contrib.module.ModulatedDeformConv(nn.Module)
```

## 功能说明

应用基于NPU的Modulated Deformable 2D卷积操作。

## 参数说明

- in_channels (Int) - 输入图像中的通道数。
- out_channels (Int) - 卷积产生的通道数。
- kernel_size(Int或Tuple) - 卷积核大小。
- stride(Int, Tuple，默认值为1) - 卷积步长。
- padding (Int或Tuple，默认值为0) - 添加到输入两侧的零填充。
- dilation (Int或Tuple，默认值为1) - 内核元素间距。
- groups (Int，默认值为1) - 从输入通道到输出通道的阻塞连接数。
- deform_groups (Int) - 可变形组分区的数量。
- bias (Bool，默认值为False) - 如果值为True，则向输出添加可学习偏差。
- pack (Bool，默认值为True) - 如果值为True，此模块将包括conv_offset和掩码。

## 约束说明

ModulatedDeformConv仅实现float32数据类型的操作。conv_offset中权重和偏置必须初始化为0。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> from torch_npu.contrib.module import ModulatedDeformConv
>>> m = ModulatedDeformConv(32, 32, 1).npu()
>>> input_tensor = torch.randn(2, 32, 5, 5).npu()
>>> output = m(input_tensor)
```

