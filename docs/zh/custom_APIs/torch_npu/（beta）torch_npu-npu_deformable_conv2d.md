# （beta）torch_npu.npu_deformable_conv2d

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √   |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

- API功能：使用预期输入计算变形卷积（deformable convolution）的输出。

- 计算公式：
  
  假定输入（self）的shape是[N, inC, inH, inW]，输出的（out）的shape为[N, outC, outH, outW]，根据已有参数计算outH、outW:
  
  $$
  outH = (inH + padding[0] + padding[1] - ((K_H - 1) * dilation[2] + 1)) // stride[2] + 1
  $$
  
  $$
  outW = (inW + padding[2] + padding[3] - ((K_W - 1) * dilation[3] + 1)) // stride[3] + 1
  $$
  
  标准卷积计算采样点下标：
  
  $$
  x = -padding[2] + ow*stride[3] + kw*dilation[3], kw的取值为(0, K\_W-1)
  $$
  
  $$
  y = -padding[0] + oh*stride[2] + kh*dilation[2], kh的取值为(0, K\_H-1)
  $$
  
  根据传入的offset，进行变形卷积，计算偏移后的下标：
  
  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  使用双线性插值计算偏移后点的值：
  
  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$
  
  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\ 
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\ 
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\ 
  $$
  
  $$
  deformOut(x, y) = weight_{00} * self(x0, y0) + weight_{01} * self(x0,y1) + weight_{10} * self(x1, y0) + weight_{11} * self(x1,y1)
  $$
  
  进行卷积计算得到最终输出：
  
  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{deformOut}(N_i, k)
  $$

## 函数原型

```
torch_npu.npu_deformable_conv2d(self, weight, offset, bias=None, kernel_size, stride, padding, dilation=[1,1,1,1], groups=1, deformable_groups=1, modulated=True) -> (Tensor, Tensor)
```

## 参数说明

- **self** (`Tensor`): 必选参数，表示输入图像的4D张量，对应公式中的`self`。不支持空Tensor，支持非连续的Tensor。数据格式为$NCHW$、$ND$，数据按以下顺序存储：`[batch, in_channels, in_height, in_width]`。其中in_height * in_width不能超过2147483647。
- **weight** (`Tensor`): 必选参数，可学习过滤器的4D张量，对应公式中的`weight`。不支持空Tensor，支持非连续的Tensor。数据格式为$NCHW$、$ND$，数据格式需与`self`相同。数据按以下顺序存储：`[out_channels, in_channels / groups, filter_height, filter_width]`。其中，`filter_height`表示卷积核的高度，`filter_width`表示卷积核的宽度。
- **offset** (`Tensor`): 必选参数，x-y坐标偏移和掩码的4D张量，对应公式中的`offset`。不支持空Tensor，支持非连续的Tensor。数据格式为$NCHW$、$ND$，数据格式需与`self`相同。当`modulated`为True时，数据按以下顺序存储：`[batch, deformable_groups * filter_height * filter_width * 3, out_height, out_width]`；当`modulated`为False时，数据按以下顺序存储为`[batch, deformable_groups * filter_height * filter_width * 2, out_height, out_width]`。
- **bias** (`Tensor`): 可选参数，过滤器输出附加偏置（additive bias）的1D张量，对应公式中的`bias`。不支持空Tensor，支持非连续的Tensor。数据格式为$ND$。默认值为None，如果不为None时数据按`[out_channels]`的顺序存储。
- **kernel_size** (`List[int]`): 必选参数，内核大小，对应公式中的`K_H`、`K_W`。2个整数的元组/列表(K_H, K_W)，各元素均大于零，K_H * K_W不能超过2048，K_H * K_W * in_channels/groups不能超过65535。
- **stride** (`List[int]`): 必选参数，4个整数的列表，表示每个输入维度的滑动窗口步长，对应公式中的`stride`。维度顺序根据`self`的数据格式解释。各元素均大于零，N维和C维必须设置为1。
- **padding** (`List[int]`): 必选参数，4个整数的列表，表示要添加到输入每侧（顶部、底部、左侧、右侧）的像素数，对应公式中的`padding`。
- **dilation** (`List[int]`): 可选参数，4个整数的列表，表示输入每个维度的膨胀系数（dilation factor），对应公式中的`dilation`。维度顺序根据`self`的数据格式解释。各元素均大于零，N维和C维必须设置为1。默认值为`[1, 1, 1, 1]`。
- **groups** (`int`): 可选参数，`int32`类型，表示从输入通道到输出通道的分组数。`in_channels`和`out_channels`需都可被`groups`数整除，且`groups`的值大于零。默认值为1。
- **deformable_groups** (`int`): 可选参数，`int32`类型，表示可变形组分区的数量。`in_channels`需可被`deformable_groups`数整除，且`deformable_groups`的值大于零。默认值为1。
- **modulated** (`bool`): 可选参数，表示`offset`中是否包含掩码。默认值为True，表示`offset`中包含掩码；若为False，则不包含。

## 返回值说明
- **conv_output** (`Tensor`): 经过变形卷积处理后的结果张量，对应公式中的`out`。不支持空Tensor，支持非连续的Tensor。shape为`[batch, out_channels, out_height, out_width]`。数据格式需与`self`相同。
- **deformable_offset** (`Tensor`): 变形卷积中用于调整采样位置的偏移量张量，对应公式中的`deformOut`。不支持空Tensor，支持非连续的Tensor。shape为`[batch, in_channels, out_height * K_H, out_width * K_W]`数据格式需与`self`相同。

## 约束说明

所有Tensor类型的输入参数，无论输入何种数据类型都会被自动转换为`float32`；输出参数的数据类型为`float32`。


## 调用示例

```python
>>> import torch, torch_npu
>>> x = torch.rand(16, 32, 32, 32).npu()
>>> weight = torch.rand(32, 32, 5, 5).npu()
>>> offset = torch.rand(16, 75, 32, 32).npu()
>>> output, deform_offset = torch_npu.npu_deformable_conv2d(x, weight, offset, None, kernel_size=[5, 5], stride = [1, 1, 1, 1], padding = [2, 2, 2, 2])
>>> output.shape
torch.Size([16, 32, 32, 32])
>>> deform_offset.shape
torch.Size([16, 32, 160, 160])
```

