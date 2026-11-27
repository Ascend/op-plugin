# （beta）torch_npu.contrib.function.fuse_add_softmax_dropout

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

先将原始注意力分数乘以`1 / sqrt(attn_head_size)`，再与掩码相加，沿最后一维计算softmax并应用dropout。

对于形状为 `(N, C, H, W)`的分数张量，当`H`为128、256、384或512且`N * C`是32的倍数时，接口使用融合算子。融合路径在训练时采用`p`，推理时将丢弃概率设为0。其他形状使用逐步计算路径，由传入的`dropout`模块决定丢弃行为。

## 函数原型

```python
torch_npu.contrib.function.fuse_add_softmax_dropout(training, dropout, attn_mask, attn_scores, attn_head_size, p=0.5, dim=-1) -> Tensor
```

## 参数说明

- **training** (`bool`)：必选参数。控制融合路径是否启用dropout。
- **dropout** (`nn.Module`)：必选参数。逐步计算路径使用的dropout层。按需要为该模块设置概率及训练或推理模式。
- **attn_mask** (`Tensor`)：必选参数，加到注意力分数上的掩码。建议与`attn_scores`使用相同的形状、数据类型和NPU设备。
- **attn_scores** (`Tensor`)：必选参数，形状为 `(N, C, H, W)`的原始注意力分数。
- **attn_head_size** (`float`)：必选参数，每个注意力头的维度，须为正数，用于计算缩放系数。
- **p** (`float`)：可选参数，融合路径在训练时的元素丢弃概率，范围为 `[0, 1]`，默认值为`0.5`。
- **dim** (`int`)：可选参数，默认值为 `-1`。当前实现固定沿最后一维计算softmax。

## 返回值说明

`Tensor`

返回计算结果。

## 调用示例

示例使用形状为 `(96, 12, 30, 30)`的注意力掩码与分数。此形状使用逐步计算路径，由传入模块以0.1的概率丢弃元素。

```python
import torch
import torch_npu
from torch_npu.contrib.module import DropoutWithByteMask
from torch_npu.contrib.function import fuse_add_softmax_dropout
training = True
dropout = DropoutWithByteMask(0.1)
attn_mask = torch.rand(96, 12, 30, 30).half().npu()
attn_scores = torch.rand(96, 12, 30, 30).half().npu()
attn_head_size = 64
axis = -1
output = fuse_add_softmax_dropout(training, dropout, attn_mask, attn_scores, attn_head_size, p=0.1, dim=axis)
print(output.shape)
```

示例输出如下。

```text
torch.Size([96, 12, 30, 30])
```
