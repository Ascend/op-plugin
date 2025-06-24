# （beta）torch_npu.contrib.DCNv2

>**须知：**<br>
>该接口计划废弃，可以使用torch_npu.contrib.ModulationDeformConv接口进行替换。

## 函数原型

```
torch_npu.contrib.DCNv2(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, deformable_groups=1, bias=True, pack=True)
```

## 功能说明

应用基于NPU的调制可变形2D卷积操作。ModulationDeformConv的实现主要是基于mmcv的实现进行设计和重构。

## 参数说明

- in_channels (int)：输入图像的channel数量。
- out_channels (int)：卷积输出channel的数量。
- kernel_size(int, tuple)：卷积核的尺寸。
- stride(int, tuple)：卷积的stride，默认值为1。
- padding (int or tuple)：对输入图像两侧添加0填充，默认值为0。
- dilation (int or tuple)：卷积核元素间距，默认值为1。
- groups (int)：输入通道和输出通道的组数量，默认值为1。
- deformable_groups (int)：可变形分区数量。
- bias (bool)：设置为True将对output添加bias，默认值为False。
- pack (bool)：设置为True则将在模型中添加conv_offset和mask，默认值为True。

## 约束说明

ModulationDeformConv仅实现fp32数据类型下的操作。注意，conv_offset中的权重和偏置必须初始化为0。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

## 调用示例

```python
>>> x = torch.randn((2, 2, 5, 5), dtype=torch.float32).npu()
>>> x.requires_grad = True
>>> model = torch_npu.contrib.DCNv2(2, 2, 3, 2, 1).npu()
>>> output = model(x)
>>> output.sum().backward()
```

