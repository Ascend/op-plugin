# （beta）torch_npu.npu_conv3d

> [!NOTICE]  
> 该接口计划废弃，可以使用`torch.nn.functional.conv3d`接口进行替换。

## 产品支持情况

<!-- npu="A3" id1 -->
- <term>Atlas A3训练系列产品</term>：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2训练系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910" id3 -->
- <term>Atlas训练系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

在由多个输入平面组成的输入图像上应用一个3D卷积。

## 函数原型

```python
torch_npu.npu_conv3d(input, weight, bias, stride, padding, dilation, groups) -> Tensor
```

## 参数说明

- **input**（`Tensor`）：必选参数，输入张量，shape为(minibatch, in_channels, iT, iH, iW)。
- **weight**（`Tensor`）：必选参数，卷积核张量，shape为(out_channels, in_channels/groups, kT, kH, kW)。
- **bias**（`Tensor`）：必选参数，可传None。传入张量时，形状为 (out_channels,)。
- **stride**（`List[int]`）：必选参数。按深度、高度和宽度指定卷积步长，各元素为正整数。
- **padding**（`List[int]`）：必选参数。按深度、高度和宽度指定输入两侧的填充量，各元素为非负整数。
- **dilation**（`List[int]`）：必选参数。按深度、高度和宽度指定卷积核元素间距，各元素为正整数。
- **groups**（`int`）：必选参数，取正整数。输入和输出的通道数须能被groups整除。
