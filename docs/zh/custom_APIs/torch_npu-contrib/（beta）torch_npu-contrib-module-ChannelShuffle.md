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

- `split_shuffle=False`时，计算逻辑如下。

    ```python
    import torch

    def cpu_channel_shuffle(x1, x2, groups=2):
        x = torch.cat((x1, x2), dim=1)
        batchsize, channels, height, width = x.shape
        x = x.reshape(batchsize, groups, channels // groups, height, width)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batchsize, channels, height, width)
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

- `split_shuffle=True`时返回两个Tensor组成的元组。第一个输出的通道数为`in_channels // groups`，第二个输出包含其余通道。
- `split_shuffle=False`时返回一个Tensor，其通道数为`in_channels`。

输出保持输入的批维度和空间维度。

## 约束说明

`groups`须为能整除总通道数的正整数。

## 调用示例

两路输入各有32个通道，因此`in_channels=64`。`split_shuffle=True`使模块在重排通道后将结果均分为两路。

```python
import torch, torch_npu
from torch_npu.contrib.module import ChannelShuffle
x1 = torch.randn(2, 32, 7, 7).npu()
x2 = torch.randn(2, 32, 7, 7).npu()
m = ChannelShuffle(64, split_shuffle=True)
out1, out2 = m(x1, x2)
print(out1.shape)
print(out2.shape)
```

示例输出如下。

```text
torch.Size([2, 32, 7, 7])
torch.Size([2, 32, 7, 7])
```
