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
torch_npu.contrib.module.MultiheadAttention(nn.Module)
```

## 参数说明
- **embed_dim** (`int`)：模型总维度。
- **num_heads** (`int`)：并行attention head。
- **kdim**(`int`)：key的特性总数。默认值为None。
- **vdim**(`int`)：value的特性总数。默认值为None。
- **dropout** (`float`)：Dropout概率。
- **bias** (`bool`)：如果指定此参数，则向输入/输出投影层添加偏置。默认值为True。
- **add_bias_kv** (`bool`)：如果指定此参数，则在dim=0处向键值序列添加偏置。默认值为False。
- **add_zero_attn** (`bool`)：如果指定此参数，则在dim=1处向键值序列新加一批零。，默认值为False。
- **self_attention**(`bool`)：计算你自己的attention score。，默认值为False。
- **encoder_decoder_attention** (`bool`)：输入为编码器输出和解码器self-attention输出，其中编码器self-attention用作key和value，解码器self-attention用作查询。默认值为False。
- **q_noise**(`float`)：量化噪声量。
- **qn_block_size**(`int`)：用于后续iPQ量化的块大小。

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
```

