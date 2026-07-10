# torch_npu.npu_sparse_flash_attention

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 inference products</term>  | √  |
|<term>Atlas A3 inference products</term>  | √  |

## Function

- Description: Provides highly efficient attention computations for long-sequence inference scenarios. `sparse_flash_attention` (SFA) reduces computational cost by computing only the critical portions of attention. However, it introduces a large amount of discrete memory access. This increases data transfer overhead and impacts overall performance.

- Formulas:

    $$
    \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
    $$

    $\tilde{K}$ and $\tilde{V}$ represent key and value tensors with higher importance obtained through a selection algorithm such as `lightning_indexer`. They typically feature sparse or block-sparse characteristics. $d_k$ represents the per-head dimension of $Q$ and $\tilde{K}$.
    The `sparse_flash_attention` operator is specifically designed for sparse attention scenarios, with detailed optimizations such as instruction reduction and aggregated data transfer to mitigate the overhead caused by irregular memory access.

## Prototype

```python
torch_npu.npu_sparse_flash_attention(query, key, value, sparse_indices, scale_value, *, block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None, query_rope=None, key_rope=None, sparse_block_size=1, layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, attention_mode=0, return_softmax_lse=False) -> (Tensor, Tensor, Tensor)
```

## Parameters

> [!NOTE]  
>
>- Dimension definitions for the `query`, `key`, and `value` parameters:<br>`B` (`Batch Size`) indicates the input sample batch size.<br>`S` (`Sequence Length`) indicates the input sample sequence length.<br>`H` (`Head Size`) indicates the hidden layer size.<br>`N` (`Head Num`) indicates the number of heads.<br>`D` (`Head Dim`) indicates the minimum unit size of the hidden layer, satisfying `D = H/N`.<br>`T` indicates the cumulative sum of the sequence lengths of all batch input samples.
>- `Q_S` or `S1` indicates the S dimension in the shape of `query`.<br>`KV_S` or `S2` indicates the S dimension in the shape of `key`.<br>`Q_N` or `N1` indicates `num_query_heads`.<br>`KV_N` or `N2` indicates `num_key_value_heads`.<br>`T1` indicates the T dimension in the shape of `query`.<br>`T2` indicates the accumulated sum of the input sample sequence lengths in the shape of `key`.
>
- **`query`** (`Tensor`): Required. $Q$ in the formulas. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16` or `float16`. When `layout_query` is `BSND`, the shape must be `[B, S1, N1, D]`. When `layout_query` is `TND`, the shape must be `[T1, N1, D]`. The value of `N1` can be `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`.
- **`key`** (`Tensor`): Required. $\tilde{K}$ in the formula. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16` or `float16`. When `layout_kv` is `PA_BSND`, the shape must be `[block_num, block_size, KV_N, D]`. The parameter `block_num` indicates the total number of blocks in PageAttention. The parameter `block_size` indicates the number of tokens in one block. The value of `block_size` must be a multiple of `16` and must be less than or equal to `1024`. When `layout_kv` is `BSND`, the shape must be `[B, S2, KV_N, D]`. When `layout_kv` is `TND`, the shape must be `[T2, KV_N, D]`. The value of `KV_N` must be `1`.

- **`value`** (`Tensor`): Required. $\tilde{V}$ in the formula. Non-contiguous tensors are not supported. The size of the dimension `N` must be `1`. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape must be identical to that of `key`.
    
- **`sparse_indices`** (`Tensor`): Required. Index tensor for discrete KV cache selection. Non-contiguous tensors are not supported. The data layout can only be ND. The data type can only `int32`. When `layout_query` is `BSND`, the shape of this parameter must be `[B, Q_S, KV_N, sparse_size]`. When `layout_query` is `TND`, the shape of this parameter must be `[Q_T, KV_N, sparse_size]`, where `sparse_size` indicates the block count selected per discrete operation. The valid values in each row must be positioned in the first half, the invalid values must be positioned in the second half, and the value of `sparse_size` must be greater than 0.

- **`scale_value`** (`double`): Required. Scaling factor applied as a scalar value after the matrix multiplication of query and key. The data type must be `double`.

- **`*`**: Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).

- **`block_table`** (`Tensor`): Optional. Block mapping table used for kvCache storage in PageAttention. The data layout can be ND. The data type can be `int32`. The shape of this parameter is 2D. The length of the first dimension must be identical to the batch size `B`, and the length of the second dimension must not be less than the block count corresponding to the maximum `S2` across all batches, computed through dividing `S2_max` by `block_size` and rounding the result up.

- **`actual_seq_lengths_query`** (`Tensor`): Optional. Valid token count of `query` in different batches. The data type can be `int32`. If not specified, this parameter can be set to `None` to indicate that its length is identical to the size of the S dimension in the shape of `query`. The valid token count for each batch must not exceed the size of the S dimension in `query` and must be greater than or equal to 0. This parameter must be a 1D tensor of length `B`.<br>When `layout_query` is `TND`, this parameter must be provided, where its element count determines the batch size `B`, and each element indicates the cumulative token count of the current batch and all preceding batches, representing a prefix sum. Therefore, the value of each element must be greater than or equal to that of the preceding element.

- **`actual_seq_lengths_kv`** (`Tensor`): Optional. Valid token count of `key` and `value` in different batches. The data type can be `int32`. If omitted, you can set it to `None`, indicating that its length is identical to the size of the `S` dimension in the shape of `key`. The valid token count for each batch must not exceed the size of the S dimension in `key` and `value` and must be greater than or equal to `0`. This parameter must be a 1D tensor of length `B`.<br>When `layout_kv` is set to `TND` or `PA_BSND`, this parameter must be provided. When `layout_kv` is set to `TND`, each element indicates the prefix sum of token counts across batches, and the value of each element must be greater than or equal to that of the preceding element.

- **`query_rope`** (`Tensor`): Optional. RoPE information for the query in the MLA structure. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16` or `float16`.
    
- **`key_rope`** (`Tensor`): Optional. RoPE information for the key in the MLA structure. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16` or `float16`.

- **`sparse_block_size`** (`int`): Optional. Block size used in the sparse phase during the importance score computation. The data type can be `int64`. The value must be a power of 2 in the range of [1, 128].
    - When `sparse_block_size` is set to `1`, token-wise sparsification is enabled. Each token acts as an independent unit. The configuration evaluates the independent correlation degree between each query token and each key-value token during importance score computations.
    - When `sparse_block_size` is greater than `1` and less than or equal to `128`, block-wise sparsification is enabled. The token sequences are partitioned into continuous blocks of a fixed size. Importance evaluations are performed based on blocks. Tokens within a block share the identical sparsification decision.

- `layout_query` (`str`): Optional. Data layout of the input `query`. Valid values are `BSND` or `TND`. If omitted, the default value `BSND` is used.

- **`layout_kv`** (`str`): Optional. Data layout of the input `key`. Valid values are `TND`, `BSND`, or `PA_BSND`. If omitted, the default value `BSND` is used. `PA_BSND` is used when PageAttention is enabled.

- **`sparse_mode`** (`int`): Optional. Sparse mode. The data type can be `int64`.
    - `0`: enables full computation.
    - `3`: enables the `rightDownCausal` mode mask, corresponding to lower triangular scenarios where the dividing line extends from the bottom-right vertex to the top-left vertex.

- **`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. Only the default value `2^63 - 1` is supported.

- **`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. Only the default value `2^63 - 1` is supported.

- **`attention_mode`**: Optional. Indicates the attention mode. The data type can be `int64`. Only the value `2` is supported, indicating the MLA-absorb mode. In this mode, computations concatenate the `nope` portion of the query or key with the rope portion of `query_rope` or `key_rope` along the head dimension `D`. This merges them to form the final query or key for subsequent computations. The key or value shares the identical underlying tensor data.

- **`return_softmax_lse`** (`bool`): Optional. Specifies whether to return `softmax_max` or `softmax_sum`. Valid values are `True` (enables the return but does not support graph capture mode) or `False` (disables the return). The default value is `False`. This parameter is supported only during training when `layout_kv` is not `PA_BSND`.

## Return Values

- **`attention_out`** (`Tensor`): Output in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. When `layout_query` is `BSND`, the shape must be `[B, S1, N1, D]`. When `layout_query` is `TND`, the shape must be `[T1, N1, D]`.
- **`softmax_max`** (`Tensor`): Optional output. Obtained by taking the maximum value of the query-key product in the attention computation. The data type can be `float`. When `layout_query` is `BSND`, the shape must be `[B, N2, S1, N1/N2]`. When `layout_query` is `TND`, the shape must be `[N2, T1, N1/N2]`.
- **`softmax_sum`** (`Tensor`): Optional output. Obtained by subtracting `softmax_max` from the query-key product in the attention computation, applying `exp`, and then calculating the sum. The data type can be `float`. When `layout_query` is `BSND`, the shape must be `[B, N2, S1, N1/N2]`. When `layout_query` is `TND`, the shape must be `[N2, T1, N1/N2]`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- The dimension `D` of `query` must be identical to that of `key` or `value`, and must be `512`. The dimension `D` of `query_rope` must be identical to that of `key_rope`, and must be `64`.
- The data types of `query`, `key`, or `value` must be identical.
- The parameter `sparse_block_size` must evenly divide `block_size`.
- When `layout_kv` is `PA_BSND`, `layout_query` or `layout_kv` are not required to be identical. When `layout_kv` is `BSND` or `TND`, `layout_query` or `layout_kv` must be identical.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import numpy as np
    import random
    
    query_type = torch.float16
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    sparse_block_count = 2048
    t = 10
    b = 4
    s1 = 1
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    dr = 64
    tile_size = 128
    block_size = 256
    s2_act = 4096
    attention_mode = 2
    return_softmax_lse = False

    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
    key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
    value = key.clone()
    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
        to(torch.int32)
    query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
    act_seq_q = [s1] * b
    act_seq_kv = [s2_act] * b

    query = query.npu()
    key = key.npu()
    value = value.npu()
    sparse_indices = sparse_indices.npu()
    query_rope = query_rope.npu()
    key_rope = key_rope.npu()
    act_seq_q = torch.tensor(act_seq_q).to(torch.int32).npu()
    act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).npu()

    attention_out, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        query, key, value, sparse_indices, scale_value, block_table=None, 
        actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
        layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1,
        attention_mode = attention_mode, return_softmax_lse = return_softmax_lse)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair
    import torch.nn as nn
    import numpy as np
    import random

    query_type = torch.float16
    scale_value = 0.041666666666666664
    sparse_block_size = 1
    sparse_block_count = 2048
    t = 10
    b = 4
    s1 = 1
    s2 = 8192
    n1 = 128
    n2 = 1
    dn = 512
    dr = 64
    tile_size = 128
    block_size = 256
    s2_act = 4096
    attention_mode = 2
    return_softmax_lse = False

    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dn))).to(query_type)
    key = torch.tensor(np.random.uniform(-5, 10, (b, s2, n2, dn))).to(query_type)
    value = key.clone()
    idxs = random.sample(range(s2_act - s1 + 1), sparse_block_count)
    sparse_indices = torch.tensor([idxs for _ in range(b * s1 * n2)]).reshape(b, s1, n2, sparse_block_count). \
        to(torch.int32)
    query_rope = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, dr))).to(query_type)
    key_rope = torch.tensor(np.random.uniform(-10, 10, (b, s2, n2, dr))).to(query_type)
    act_seq_q = [s1] * b
    act_seq_kv = [s2_act] * b

    query = query.npu()
    key = key.npu()
    value = value.npu()
    sparse_indices = sparse_indices.npu()
    query_rope = query_rope.npu()
    key_rope = key_rope.npu()
    act_seq_q = torch.tensor(act_seq_q).to(torch.int32).npu()
    act_seq_kv = torch.tensor(act_seq_kv).to(torch.int32).npu()

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)

    class Network(nn.Module):
        def __init__(self):
            super(Network, self).__init__()

        def forward(self, query, key, value, sparse_indices, scale_value, 
            block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
            query_rope, key_rope, sparse_block_size, layout_query, layout_kv,
            sparse_mode, pre_tokens, next_tokens, attention_mode, return_softmax_lse):
            
            attention_out, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
                query, key, value, sparse_indices, scale_value, block_table=None, 
                actual_seq_lengths_query=actual_seq_lengths_query, actual_seq_lengths_kv=actual_seq_lengths_kv,
                query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
                layout_query=layout_query, layout_kv=layout_kv, sparse_mode=sparse_mode, 
                pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1, attention_mode = attention_mode, 
                return_softmax_lse = return_softmax_lse)
            return attention_out, softmax_max, softmax_sum

    mod = torch.compile(Network().npu(), backend=npu_backend, fullgraph=True)

    attention_out, softmax_max, softmax_sum = mod(query, key, value, sparse_indices, 
        scale_value, block_table=None, actual_seq_lengths_query=act_seq_q, actual_seq_lengths_kv=act_seq_kv,
        query_rope=query_rope, key_rope=key_rope, sparse_block_size=sparse_block_size,
        layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=(1<<63)-1, next_tokens=(1<<63)-1,
        attention_mode = attention_mode, return_softmax_lse = return_softmax_lse)
    ```
