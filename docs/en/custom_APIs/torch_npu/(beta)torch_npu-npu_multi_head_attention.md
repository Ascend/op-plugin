# (beta) torch_npu.npu_multi_head_attention

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Implements the Multi-Head Attention (MHA) computation logic in the Transformer module.

## Prototype

```python
torch_npu.npu_multi_head_attention(query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`query`** (`Tensor`): Only the `float16` data type is supported.
- **`key`** (`Tensor`): Only the `float16` data type is supported.
- **`value`** (`Tensor`): Only the `float16` data type is supported.
- **`query_weight`** (`Tensor`): Only the `float16` data type is supported.
- **`key_weight`** (`Tensor`): Only the `float16` data type is supported.
- **`value_weight`** (`Tensor`): Only the `float16` data type is supported.
- **`attn_mask`** (`Tensor`): Only the `float16` data type is supported.
- **`out_proj_weight`** (`Tensor`): Only the `float16` data type is supported.
- **`query_bias`** (`Tensor`): Only the `float16` data type is supported.
- **`key_bias`** (`Tensor`): Only the `float16` data type is supported.
- **`value_bias`** (`Tensor`): Only the `float16` data type is supported.
- **`out_proj_bias`** (`Tensor`): Only the `float16` data type is supported.
- **`dropout_mask`** (`Tensor`): Only the `float16` data type is supported.
- **`attn_head_num`** (`int`): Attention head numbers.
- **`attn_dim_per_head`** (`int`): Attention dimension of a head.
- **`src_len`** (`int`): Source length.
- **`tgt_len`** (`int`): Target length.
- **`dropout_prob`** (`float`): Dropout keep probability.
- **`softmax_use_float`** (`bool`): SoftMax uses `float32` to keep precision.

## Return Values

- **`y`** (`Tensor`): Only the `float16` data type is supported.
- **`dropout_mask`** (`Tensor`): Only the `float16` data type is supported.
- **`query_res`** (`Tensor`): Only the `float16` data type is supported.
- **`key_res`** (`Tensor`): Only the `float16` data type is supported.
- **`value_res`** (`Tensor`): Only the `float16` data type is supported.
- **`attn_scores`** (`Tensor`): Only the `float16` data type is supported.
- **`attn_res`** (`Tensor`): Only the `float16` data type is supported.
- **`context`** (`Tensor`): Only the `float16` data type is supported.

## Constraints

`attn_head_num` must be aligned to a multiple of 16.
`attn_dim_per_head` must be aligned to a multiple of 16.
`src_len` must be aligned to a multiple of 16.
`tgt_len` must be aligned to a multiple of 16.

## Example

```python
import torch
import torch_npu
import numpy as np
 
batch = 8
attn_head_num = 16
attn_dim_per_head = 64
src_len = 64
tgt_len = 64
dropout_prob = 0.0
softmax_use_float = True
 
weight_col = attn_head_num * attn_dim_per_head
query = torch.from_numpy(np.random.uniform(-1, 1, (batch * tgt_len, weight_col)).astype("float16")).npu()
key = torch.from_numpy(np.random.uniform(-1, 1, (batch * src_len, weight_col)).astype("float16")).npu()
value = torch.from_numpy(np.random.uniform(-1, 1, (batch * tgt_len, weight_col)).astype("float16")).npu()
query_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
key_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
value_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
out_proj_weight = torch.from_numpy(np.random.uniform(-1, 1, (weight_col, weight_col)).astype("float16")).npu()
attn_mask = torch.from_numpy(np.random.uniform(-1, 1, (batch, attn_head_num, tgt_len, src_len)).astype("float16")).npu()
query_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
key_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
value_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
out_proj_bias = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
dropout_mask = torch.from_numpy(np.random.uniform(-1, 1, (weight_col,)).astype("float16")).npu()
            
npu_result, npu_dropout_mask, npu_query_res, npu_key_res, npu_value_res, npu_attn_scores, npu_attn_res, npu_context = torch_npu.npu_multi_head_attention (query, key, value, query_weight, key_weight, value_weight, attn_mask, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias,  dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float)
 
print(npu_result)
 
 
 
tensor([[ 623.5000,   75.5000,  307.0000,  ...,   25.3125, -418.7500,
           35.9688],
        [-254.2500, -165.6250,  176.2500,  ...,   87.3750,   78.0000,
           65.2500],
        [ 233.2500,  207.3750,  324.7500,  ...,   38.6250, -264.2500,
          153.7500],
        ...,
        [-110.2500,  -92.5000,  -74.0625,  ...,  -68.0625,  195.6250,
         -157.6250],
        [ 300.0000, -184.6250,   -6.0039,  ...,  -15.7969, -299.0000,
          -93.1875],
        [  -2.5996,   36.8750,  100.0625,  ...,  112.7500,  202.0000,
         -166.3750]], device='npu:0', dtype=torch.float16)
```
