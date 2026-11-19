# （beta）torch_npu.contrib.module.ChannelShuffle

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

- API功能：应用NPU兼容的通道Shuffle操作。

- 等价计算逻辑：
  
    split_shuffle=False场景可使用`cpu_channel_shuffle`等价替换`torch_npu.contrib.module.ChannelShuffle`，两者计算逻辑一致。

  ```python
  import torch
  def cpu_channel_shuffle(x, groups, split_shuffle):
      # cpu仅支持 split_shuffle=False场景
      batchsize, num_channels, height, width = x.size()
      channels_per_group = num_channels // groups
      x.requires_grad_(True)
      # reshape
      x = x.view(batchsize, groups, channels_per_group, height, width)
    
      x = torch.transpose(x, 1, 2).contiguous()
    
      # flatten
      x = x.view(batchsize, -1, height, width)
      output = x.view(batchsize, -1, height, width)
      return output
  ```
  
## 函数原型

```python
torch_npu.contrib.module.ChannelShuffle(in_channels, groups=2, split_shuffle=True)
```

## 参数说明

**计算参数**

- **in_channels** (`int`)：必选参数。输入张量中的通道总数，取值为正整数，需可被groups整除，且等于计算输入x1与x2的通道数之和。
- **groups** (`int`)：可选参数。Shuffle组数。默认值为2。
- **split_shuffle** (`bool`)：可选参数。Shuffle后是否执行chunk操作。默认值为True。

**计算输入**

- **x1** (`Tensor`)：必选参数。输入张量。shape为$(N, C_{in}, *)$，其中N为批处理大小，$C_{in}$为单路输入的通道数，需与x2的shape保持一致。
- **x2** (`Tensor`)：输入张量。shape为$(N, C_{in}, *)$。

## 返回值说明

- **out1** (`Tensor`)：输出张量。
  - 当`split_shuffle=True`（默认值）时，返回由两个Tensor组成的元组`(out1, out2)`，每个Tensor的shape为`(N, in_channels / 2, *)`。
  - 当`split_shuffle=False`时，返回一个Tensor，其shape为`(N, in_channels, *)`。

- **out2** (`Tensor`)：输出张量。shape为$(N, C_{out}, *)$。

## 约束说明

只实现了groups为2的场景，请自行修改其他groups场景。

## 调用示例

```python
>>> import torch, torch_npu
>>> from torch_npu.contrib.module import ChannelShuffle
>>> x1 = torch.randn(2, 32, 7, 7).npu()
>>> x2 = torch.randn(2, 32, 7, 7).npu()
>>> m = ChannelShuffle(64, split_shuffle=True)
>>> out1, out2 = m(x1, x2)
>>> print(out1.shape)
torch.Size([2, 32, 7, 7])
>>> print(out2.shape)
torch.Size([2, 32, 7, 7])
```
