# (beta) torch_npu.contrib.module.MultiheadAttention

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Executes a multi-head attention operation.

## Prototype

```python
torch_npu.contrib.module.MultiheadAttention(embed_dim, num_heads, kdim=None, vdim=None, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0.0, qn_block_size=8)
```

## Parameters

- **`embed_dim`** (`int`): Total model dimension.
- **`num_heads`** (`int`): Number of parallel attention heads.
- **`kdim`** (`int`): Total number of key features. The default value is `None`.
- **`vdim`** (`int`): Total number of value features. The default value is `None`.
- **`dropout`** (`float`): Dropout probability.
- **`bias`** (`bool`): If set to `True`, a bias is added to the input and output projection layers. The default value is `True`.
- **`add_bias_kv`** (`bool`): If set to `True`, a bias is added to the key-value sequence at `dim=0`. The default value is `False`.
- **`add_zero_attn`** (`bool`): If set to `True`, a batch of zeros are appended to the key-value sequence at `dim=1`. The default value is `False`.
- **`self_attention`** (`bool`): Indicates whether this layer is a self-attention layer. If set to `True`, `embed_dim`, `kdim`, and `vdim` must have the same value. The default value is `False`.
- **`encoder_decoder_attention`** (`bool`): Specifies whether to enable encoder-decoder attention. The encoder self-attention output acts as the `key` and `value`. The decoder self-attention output acts as the `query`. The default value is `False`.
- **`q_noise`** (`float`): Quantization noise level.
- **`qn_block_size`** (`int`): Block size for subsequent iPQ quantization.

## Return Values

`Tensor` 

Multi-head attention computation result.

## Example

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
>>> print(output)
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
