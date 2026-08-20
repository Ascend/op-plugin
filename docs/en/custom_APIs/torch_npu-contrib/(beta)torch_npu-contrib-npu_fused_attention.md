# (beta) torch_npu.contrib.npu_fused_attention

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Fuses BERT self-attention computations.

## Prototype

```python
torch_npu.contrib.npu_fused_attention(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, scale=1, keep_prob=0)
```

## Parameters

- **`hidden_states`** (`Tensor`): Hidden states tensor of the last layer.
- **`attention_mask`** (`Tensor`): Attention mask.
- **`query_kernel`** (`Tensor`): Query weight.
- **`key_kernel`** (`Tensor`): Key weight.
- **`value_kernel`** (`Tensor`): Value weight.
- **`query_bias`** (`Tensor`): Query bias.
- **`key_bias`** (`Tensor`): Key bias.
- **`value_bias`** (`Tensor`): Value bias.
- **`scale`** (`float`): Scaling factor used to compute attention scores.
- **`keep_prob`** (`float`): Probability of retaining data elements during computation, equal to `1 - drop_rate`.

## Return Values

`Tensor`

Self-attention computation result.
