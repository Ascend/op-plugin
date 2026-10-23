# (beta) torch_npu.contrib.npu_fused_attention

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Fuses BERT self-attention computations.

## Prototype

```python
torch_npu.contrib.npu_fused_attention(hidden_states, attention_mask, query_kernel, key_kernel, value_kernel, query_bias, key_bias, value_bias, scale=1, keep_prob=0)
```

## Parameters

- **`hidden_states`** (`Tensor`): Required. Hidden states of the last layer. The data format supports FRACTAL_NZ. The shape is 2D, where dimension 0 must be an integer multiple of 32 and dimension 1 can be 768 or 1024.
- **`attention_mask`** (`Tensor`): Required. Mask tensor used to mask out token positions that do not need to participate in attention computation (such as padding positions). The data format supports FRACTAL_NZ. This parameter is 4D with shape `(B, 1, S, S)`, where B is the batch size and S is the sequence length.
- **`query_kernel`** (`Tensor`): Required. Query weight matrix. The data format supports FRACTAL_NZ. The shape is 2D, and dimension 0 can be 768 or 1024.
- **`key_kernel`** (`Tensor`): Required. Key weight matrix. The data format supports FRACTAL_NZ. The shape is 2D, and dimension 0 can be 768 or 1024.
- **`value_kernel`** (`Tensor`): Required. Value weight matrix. The data format supports FRACTAL_NZ. The shape is 2D, and dimension 0 can be 768 or 1024.
- **`query_bias`** (`Tensor`): Required. Query bias. The data format supports ND.
- **`key_bias`** (`Tensor`): Required. Key bias. The data format supports ND.
- **`value_bias`** (`Tensor`): Required. Value bias. The data format supports ND.
- **`scale`** (`double`): Optional. Scaling factor used to compute attention scores. The default value is `1`.
- **`keep_prob`** (`float`): Optional. Probability of retaining data during computation, equal to `1 - drop_rate`. The default value is `0`.

## Return Values

`Tensor`

Self-attention computation result.
