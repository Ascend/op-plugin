# （beta）torch_npu.contrib.module.MultiheadAttention

## 函数原型

```
torch_npu.contrib.module.MultiheadAttention(nn.Module)
```

## 功能说明

Multi-head attention。

## 参数说明

- embed_dim (Int) - 模型总维度。
- num_heads (Int) - 并行attention head。
- kdim(Int，默认值为None) - key的特性总数。
- vdim(Int，默认值为None) - value的特性总数。
- dropout (Float) - Dropout概率。
- bias (Bool，默认值为True) - 如果指定此参数，则向输入/输出投影层添加偏置。
- add_bias_kv (Bool，默认值为False) - 如果指定此参数，则在dim=0处向键值序列添加偏置。
- add_zero_attn (Bool，默认值为False) - 如果指定此参数，则在dim=1处向键值序列新加一批零。
- self_attention (Bool，默认值为False) - 计算你自己的attention score。
- encoder_decoder_attention (Bool，默认值为False) - 输入为编码器输出和解码器self-attention输出，其中编码器self-attention用作key和value，解码器self-attention用作查询。
- q_noise(Float) - 量化噪声量。
- qn_block_size(Int) - 用于后续iPQ量化的块大小。

## 支持的型号

- <term>Atlas 训练系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas A3 训练系列产品</term>
- <term>Atlas 推理系列产品</term>

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

