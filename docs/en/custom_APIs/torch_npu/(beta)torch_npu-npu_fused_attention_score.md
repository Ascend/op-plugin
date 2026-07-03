# (beta) torch_npu.npu_fused_attention_score

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. Use `torch_npu.npu_fusion_attention` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Implements the fused computation logic of Transformer attention scores, primarily fusing operations such as `matmul`, `transpose`, `add`, `softmax`, `dropout`, `batchmatmul`, and `permute`.

## Prototype

```python
torch_npu.npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob, query_transpose=False, key_transpose=False, bmm_score_transpose_a=False, bmm_score_transpose_b=False, value_transpose=False, dx_transpose=False) -> Tensor
```

## Parameters

- **`query_layer`** (`Tensor`): Required. Only `float16` is supported. If transpose parameters are not considered, the shape must be `(batch_size, num_heads, q_seq_len, head_dim)`.
- **`key_layer`** (`Tensor`): Required. Only `float16` is supported. If transpose parameters are not considered, the shape must be `(batch_size, num_heads, k_seq_len, head_dim)`.
- **`value_layer`** (`Tensor`): Required. Only `float16` is supported. If transpose parameters are not considered, the shape must be `(batch_size, num_heads, v_seq_len, head_dim)`.
- **`attention_mask`** (`Tensor`): Required. Only `float16` is supported. If transpose parameters are not considered, the shape must be `(batch_size, num_heads, q_seq_len, k_seq_len)`.
- **`scale`** (`Scalar`): Required. Scaling factor, which is a floating-point scalar.
- **`keep_prob`** (`float`): Required. Probability of not performing dropout. The value must be within the range `[0, 1]`, which is a floating-point number.
- **`query_transpose`** (`bool`): Optional. Specifies whether to transpose `query`. The default value is `False`.
- **`key_transpose`** (`bool`): Optional. Specifies whether to transpose `key`. The default value is `False`.
- **`bmm_score_transpose_a`** (`bool`): Optional. Specifies whether to transpose the left matrix in bmm computation. The default value is `False`.
- **`bmm_score_transpose_b`** (`bool`): Optional. Specifies whether to transpose the right matrix in bmm computation. The default value is `False`.
- **`value_transpose`** (`bool`): Optional. Specifies whether to transpose `value`. The default value is `False`.
- **`dx_transpose`** (`bool`): Optional. Specifies whether to transpose `dx` during backward computation. The default value is `False`.

## Return Values

`Tensor`

Output tensor after attention computation. The data type is `float16`. When default values are used, the shape of the output tensor must be identical to that of `query_layer`.

## Constraints

The format IDs of the input tensors must all be `29`, and the data type must be `float16`. The `num_heads` dimensions of `query_layer`, `key_layer`, and `value_layer` must be identical. The `head_dim` dimensions of `query_layer` and `key_layer` must be identical to compute attention scores, whereas the `head_dim` dimension of `value_layer` can be different and determines the output feature dimension.

## Example

```python
>>> import torch
>>> import torch_npu
>>> query_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> key_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> value_layer = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 64).npu(), 29).half()
>>> attention_mask = torch_npu.npu_format_cast(torch.rand(24, 16, 512, 512).npu(), 29).half()
>>> scale = 0.125
>>> keep_prob = 0.5
>>> context_layer = torch_npu.npu_fused_attention_score(query_layer, key_layer, value_layer, attention_mask, scale, keep_prob)
>>> print(context_layer)
tensor([[0.5010, 0.4709, 0.4841,  ..., 0.4321, 0.4448, 0.4834],
        [0.5107, 0.5049, 0.5239,  ..., 0.4436, 0.4375, 0.4651],
        [0.5308, 0.4944, 0.5005,  ..., 0.5010, 0.5103, 0.5303],
        ...,
        [0.5142, 0.5068, 0.5176,  ..., 0.5498, 0.4868, 0.4805],
        [0.4941, 0.4731, 0.4863,  ..., 0.5161, 0.5239, 0.5190],
        [0.5459, 0.5107, 0.5415,  ..., 0.4641, 0.4688, 0.4531]],
       device='npu:0', dtype=torch.float16)
```
