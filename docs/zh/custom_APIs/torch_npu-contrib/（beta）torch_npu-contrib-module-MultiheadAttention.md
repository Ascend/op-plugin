# （beta）torch_npu.contrib.module.MultiheadAttention

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

Multi-head attention（多头注意力），将输入映射到多个注意力子空间并行计算注意力后拼接输出，本接口为torch_npu.contrib提供的多头注意力模块适配版本。

## 函数原型

```python
torch_npu.contrib.module.MultiheadAttention(embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8)
```

## 参数说明

- **embed_dim** (`int`)：必选参数，模型总维度。
- **num_heads** (`int`)：必选参数，多头注意力中并行计算的注意力头数量。
- **kdim** (`int`)：可选参数，key的特征总数。默认值为None。
- **vdim** (`int`)：可选参数，value的特征总数。默认值为None。
- **dropout** (`float`)：可选参数，Dropout概率。
- **bias** (`bool`)：可选参数，如果指定此参数，则向输入/输出投影层添加偏置。默认值为True。
- **add_bias_kv** (`bool`)：可选参数，如果指定此参数，则在dim=0处向键值序列添加偏置。默认值为False。
- **add_zero_attn** (`bool`)：可选参数，如果指定此参数，则在dim=1处向键值序列新加一批零。默认值为False。
- **self_attention** (`bool`)：可选参数，表示是否为自注意力层，若取值为True，要求`embed_dim`、`kdim`、`vdim`取值相等。默认值为False。
- **encoder_decoder_attention** (`bool`)：可选参数，输入为编码器输出和解码器自注意力输出，其中编码器self-attention用作key和value，解码器self-attention用作查询。默认值为False。
- **q_noise** (`float`)：可选参数，量化噪声量，默认值为0.0。
- **qn_block_size** (`int`)：可选参数，用于后续iPQ量化的块大小，默认值为8。

**前向计算的参数**

- **query** (`Tensor`)：查询，形状为 `(bsz * tgt_len, embed_dim)`。
- **key** (`Tensor`)：键，形状为 `(bsz * s_len, kdim)`，未指定`kdim`时使用`embed_dim`。
- **value** (`Tensor`)：值，形状为 `(bsz * s_len, vdim)`，未指定`vdim`时使用`embed_dim`。
- **bsz**、**tgt_len**、**s_len** (`int`)：批大小、查询序列长度和键值序列长度。
- **key_padding_mask** (`Tensor | None`)：可选参数，注意力计算使用的加性掩码。示例使用形状为 `(bsz * tgt_len, s_len * num_heads)`的掩码。

## 返回值说明

返回 `(attn, None)`，其中`attn`是注意力输出Tensor，形状为 `(bsz * tgt_len, embed_dim)`。使用前调用 `_MHAConfig.set_fussion()`启用融合实现。

## 调用示例

本例将批维度与序列维度展平，令批大小为16，查询和键值序列长度均为64。启用融合实现后，在NPU上执行前向计算。

```python
import torch
import torch_npu
from torch_npu.contrib.module import MultiheadAttention
from torch_npu.contrib.module.multihead_attention import _MHAConfig

_MHAConfig.set_fussion()
torch.npu.config.allow_internal_format = True
model = MultiheadAttention(embed_dim=1024, num_heads=16, dropout=0.1,
                           kdim=1024, vdim=1024, self_attention=True,
                           encoder_decoder_attention=True).npu()
bsz, tgt_len, s_len = 16, 64, 64
query, key, value = [
    torch_npu.npu_format_cast(torch.rand(1024, 1024, device="npu", dtype=torch.float16), 29)
    for _ in range(3)
]
key_padding_mask = torch_npu.npu_format_cast(
    torch.zeros((bsz * tgt_len, s_len * 16), device="npu", dtype=torch.float16), 29
)
output, weights = model(query, key, value, bsz, tgt_len, s_len, key_padding_mask)
print(output.shape)
print(weights)
```

示例输出如下。

```text
torch.Size([1024, 1024])
None
```
