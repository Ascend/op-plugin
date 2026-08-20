# (beta) torch_npu.contrib.npu_fused_attention_with_layernorm

> [!NOTICE]  
> This API is planned for deprecation. Use `torch_npu.npu_fusion_attention` or `torch.nn.LayerNorm` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Fuses BERT self-attention and layer normalization computations.

## Prototype

```python
torch_npu.contrib.npu_fused_attention_with_layernorm(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, gamma, beta, scale=1, keep_prob=0)
```

## Parameters

- **`hidden_states`** (`Tensor`): Hidden states tensor of the last layer.
- **`attention_mask`** (`Tensor`): Attention mask.
- **query_kernel** (Tensor): Query weight.
- **`key_kernel`** (`Tensor`): Key weight.
- **`value_kernel`** (`Tensor`): Value weight.
- **`query_bias`** (`Tensor`): Query bias.
- **`key_bias`** (`Tensor`): Key bias.
- **`value_bias`** (`Tensor`): Value bias.
- **`gamma`** (`Tensor`): Normalization weight tensor of type `torch.nn.LayerNorm.weight`.
- **`beta`** (`Tensor`): Normalization bias tensor of type `torch.nn.LayerNorm.bias`.
- **`scale`** (`float`): Scaling factor used to compute attention scores. The default value is `1`.
- **`keep_prob`** (`double`): Probability of retaining data elements during computation, equal to `1 - drop_rate`. The default value is `0`.

## Return Values

`torch.Tensor`

Self-attention computation result.
