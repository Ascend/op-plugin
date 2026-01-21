# （beta）torch_npu.contrib.module.MultiheadAttention
## 产品支持情况


| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品</term>  | √    |
|<term>Atlas 推理系列产品</term>                                       |    √     |
|<term>Atlas 训练系列产品</term>                                       |    √     |

## 功能说明

Multi-head attention。

## 函数原型

```
torch_npu.contrib.module.MultiheadAttention(embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8)
```

## 参数说明
- **embed_dim** (`int`)：模型总维度。
- **num_heads** (`int`)：并行attention head。
- **kdim**(`int`)：key的特性总数。默认值为None。
- **vdim**(`int`)：value的特性总数。默认值为None。
- **dropout** (`float`)：Dropout概率。
- **bias** (`bool`)：如果指定此参数，则向输入/输出投影层添加偏置。默认值为True。
- **add_bias_kv** (`bool`)：如果指定此参数，则在dim=0处向键值序列添加偏置。默认值为False。
- **add_zero_attn** (`bool`)：如果指定此参数，则在dim=1处向键值序列新加一批零。默认值为False。
- **self_attention**(`bool`)：表示是否为自注意力层，若取值为True，要求`embed_dim`、`kdim`、`vdim`取值相等。默认值为False。
- **encoder_decoder_attention** (`bool`)：输入为编码器输出和解码器self-attention输出，其中编码器self-attention用作key和value，解码器self-attention用作查询。默认值为False。
- **q_noise**(`float`)：量化噪声量。
- **qn_block_size**(`int`)：用于后续iPQ量化的块大小。

## 返回值说明

`Tensor` 

Multi-head attention的计算结果。

## 调用示例

```python
>>> from torch_npu.testing.common_utils import create_common_tensor
>>> from torch_npu.contrib.module import MultiheadAttention
>>> import numpy as np
>>> from torch_npu.contrib.module.multihead_attention import _MHAConfig
>>> _MHAConfig.set_fussion()
>>> model = MultiheadAttention(embed_dim=1024,num_heads=16,dropout=0.1,kdim=1024,vdim=1024,self_attention=True,encoder_decoder_attention=True)
>>> _, query = create_common_tensor([np.float16, 29, (1024,1024)], -1, 1)
>>> _, key = create_common_tensor([np.float16, 29, (1024,1024)], -1, 1)
>>> _, value = create_common_tensor([np.float16, 29, (1024,1024)], -1, 1)
>>> _, key_padding_mask = create_common_tensor([np.float16, 29, (1024,1024)], -1, 1)
>>> bsz = 16
>>> tgt_len = 64
>>> s_len=64
>>> model = model.to("npu")
>>> output = model(query, key, value, bsz, tgt_len, s_len, key_padding_mask)
>>> output
(tensor([[-0.0385,  0.0441,  0.2432,  ...,  0.0627,  0.0254,  0.0400],
        [-0.0999, -0.0258,  0.1002,  ...,  0.0632,  0.0344,  0.0573],
        [-0.0830,  0.0405,  0.1694,  ...,  0.0787, -0.0089,  0.0544],
        ...,
        [-0.0146, -0.0303, -0.1011,  ...,  0.0689,  0.1722, -0.1125],
        [-0.0305, -0.1129, -0.0944,  ...,  0.0280,  0.1777,  0.0410],
        [-0.0035, -0.1030, -0.0957,  ...,  0.0093,  0.1171,  0.0009]],
       device='npu:0', dtype=torch.float16,
       grad_fn=<NpuMultiHeadAttentionBackward0>), None)
```

