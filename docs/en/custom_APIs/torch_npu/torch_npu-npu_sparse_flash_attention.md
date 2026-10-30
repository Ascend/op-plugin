# torch_npu.npu_sparse_flash_attention

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 inference products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Supported
<!-- end id3 -->

## Function

- Function: sparse_flash_attention (SFA) is an efficient attention computation module for long-sequence inference scenarios. It greatly reduces the computation amount by "computing only the key parts", but it introduces a large number of discrete memory accesses, which increases the data transfer time and thus affects the overall performance.

- Formulas:

    $$
    \text{softmax}(\frac{Q@\tilde{K}^T}{\sqrt{d_k}})@\tilde{V}
    $$

    In the formula, $\tilde{K},\tilde{V}$ are the Key and Value of high importance obtained based on a certain selection algorithm (such as `lightning_indexer`) and generally have sparse or block-sparse characteristics, and $d_k$ is the dimension of each head of $Q,\tilde{K}$. It corresponds to the index value `b` in `sparse_indices`, indicating that a segment of consecutive tokens starting from `b * sparse_block_size` in the KV sequence is selected. After `key/value` are collected according to `sparse_indices`, $\tilde{K},\tilde{V}$ in the formulas are obtained.

    When `attention_mode = 2`, it indicates the MLA-absorb mode. In this case, $Q$ and $\tilde{K}$ in the original formulas are formed by concatenating the `nope` part and the `rope` part:

    $$
    Q = [Q_{nope}, Q_{rope}], \quad \tilde{K} = [\tilde{K}_{nope}, \tilde{K}_{rope}]
    $$

    In the formula, `query/key` provide $Q_{nope}, \tilde{K}_{nope}$, and `query_rope/key_rope` provide $Q_{rope}, \tilde{K}_{rope}$.

    Under different `sparse_mode` values, the "valid KV set" corresponding to $\tilde{K},\tilde{V}$ in the formulas differs:

    - When `sparse_mode = 0`, $\tilde{K},\tilde{V}$ are constructed only according to the KV positions selected by `sparse_indices`, that is, only sparse selection is performed and no additional causal mask is applied.
    - When `sparse_mode = 3`, the `rightDownCausal` constraint is further superimposed on the basis of the KV positions selected by `sparse_indices`, limiting the causal range that the current query can access. If the valid lengths of query and kv of the current batch are $L_q$ and $L_{kv}$ respectively, then for the i-th query position, the kv position $j$ that can be accessed is:

      $$
      j \le (L_{kv} - L_q) + i
      $$

    The `sparse_flash_attention` released this time is a brand-new operator for Sparse Attention, with detailed optimization of instruction reduction and transfer aggregation for discrete memory access.

    A typical usage process is as follows:
    1. Use a selection algorithm (such as `lightning_indexer`) to compute the indices of important tokens based on the Query and Key;
    2. Fill the indices into the format required by the `sparse_indices` parameter;
    3. Pass `sparse_indices` to this API to complete the sparse attention computation.
    > [!NOTE]
    >
    > `lightning_indexer` is a sparse KV selection algorithm used to identify the Key/Value blocks most relevant to the current query in long-sequence inference. Its output is the `sparse_indices` tensor required by this API. If `lightning_indexer` is not used, the user must implement the index selection logic to ensure that `sparse_indices` meets the format requirements in [Parameters](#parameters).

## Prototype

```python
torch_npu.npu_sparse_flash_attention(query, key, value, sparse_indices, scale_value, *, block_table=None, actual_seq_lengths_query=None, actual_seq_lengths_kv=None, query_rope=None, key_rope=None, sparse_block_size=1, layout_query='BSND', layout_kv='BSND', sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807, attention_mode=0, return_softmax_lse=False, sinks=None) -> (Tensor, Tensor, Tensor)
```

> [!NOTE]
>
> The `value` parameter is a positional parameter (required). Its type is `Tensor` (nullable), and `None` can be passed. `value=None` is supported only when CANN >= 9.2.0; lower versions raise a `NOT_SUPPORT` error.

## Parameters

> [!NOTE]
>
>- Meanings of the dimensions of the parameters query, key, and value: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, N (Head Num) indicates the number of heads, D (Head Dim) indicates the minimum unit size of the hidden layer and satisfies D=H/N, and T indicates the cumulative sum of the sequence lengths of input samples of all batches.
>- Q\_S and S1 indicate S in the query shape, KV\_S and S2 indicate S in the key shape, Q\_N and N1 indicate num\_query\_heads, KV\_N and N2 indicate num\_key\_value\_heads, T1 indicates T in the query shape, and T2 indicates the cumulative sum of the sequence lengths of input samples in the key shape.
>
- **`query`** (`Tensor`): Required. $Q$ in the formula. Non-contiguous tensors are not supported. The data layout can be `ND`. The data type can be `torch.bfloat16` or `torch.float16`. When `layout_query` is `BSND`, the shape must be `[B, S1, N1, D]`. When `layout_query` is `TND`, the shape must be `[T1, N1, D]`. For Atlas A3 inference products/Atlas A2 inference products, the value of `N1` can be `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`. For Ascend 950PR/Ascend 950DT, the value of `N1` ranges from `1` to `128`.
    <!-- npu="A3,910b" id6 -->
    - For Atlas A3 inference products/Atlas A2 inference products, N1 supports 1/2/4/8/16/32/64/128.
    <!-- end id6 -->
    <!-- npu="950" id7 -->
    - For Ascend 950PR&950DT products, N1 supports 1 to 128.
    <!-- end id7 -->
- **`key`** (`Tensor`): Required. $\tilde{K}$ in the formula. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `torch.bfloat16` or `torch.float16`. When `layout_kv` is `PA_BSND`, the shape must be `[block_num, block_size, KV_N, D]`. The parameter `block_num` indicates the total number of blocks in PagedAttention. The parameter `block_size` indicates the number of tokens in one block. The value of `block_size` must be a multiple of `16` and must be less than or equal to `1024`. When `layout_kv` is `BSND`, the shape must be `[B, S2, KV_N, D]`. When `layout_kv` is `TND`, the shape must be `[T2, KV_N, D]`. The value of `KV_N` must be `1`.

- **`value`** (`Tensor`): Required (nullable). $\tilde{V}$ in the formula. When a tensor is passed, non-contiguous values along axis 0 are supported when `layout_kv` is `PA_BSND`; dimension N supports only `1`; the data layout can be ND; the data type can be `torch.bfloat16` or `torch.float16`; and the shape must be identical to that of `key`. When `None` is passed, the underlying kernel processes D and V together to reduce redundant memory access and computation and improve performance; in this case, `d_value` in backward propagation is an empty tensor. `value=None` is supported only in CANN 9.2.0 or later. Earlier versions raise a `NOT_SUPPORT` error.

- **`sparse_indices`** (`Tensor`): Required. Index tensor for discrete KV cache selection. This index is typically generated by a sparse selection algorithm, such as `lightning_indexer`. For details on the generation process, see the typical usage process in [Function](#function). Non-contiguous tensors are not supported. The data layout can only be ND. The data type can only be `torch.int32`. When `layout_query` is `BSND`, the shape of this parameter must be `[B, Q_S, KV_N, sparse_size]`. When `layout_query` is `TND`, the shape of this parameter must be `[Q_T, KV_N, sparse_size]`, where `sparse_size` indicates the block count selected per discrete operation. The valid values in each row must be positioned in the first half, the invalid values must be positioned in the second half, and the value of `sparse_size` must be greater than 0.

- **`scale_value`** (`double`): Required. Scaling factor applied as a scalar value after the matrix multiplication of query and key. The data type must be `double`.

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`block_table`** (`Tensor`): Optional. Block mapping table used for kvCache storage in PagedAttention. The data layout can be ND. The data type can be `torch.int32`. The shape of this parameter is 2D. The length of the first dimension must be identical to the batch size `B`, and the length of the second dimension must not be less than the block count corresponding to the maximum `S2` across all batches, computed through dividing `S2_max` by `block_size` and rounding the result up.

- **`actual_seq_lengths_query`** (`Tensor`): Optional. Valid token count of `query` in different batches. The data type can be `torch.int32`. If not specified, this parameter can be set to `None`, indicating that its length is identical to the size of the S dimension in the shape of `query`. The valid token count for each batch must not exceed the size of the S dimension in `query` and must be greater than or equal to 0. This parameter must be a 1D tensor of length `B`.<br>When `layout_query` is `TND`, this parameter must be provided, where its element count determines the batch size `B`, and each element indicates the cumulative token count of the current batch and all preceding batches, representing a prefix sum. Therefore, the value of each element must be greater than or equal to that of the preceding element.

- **`actual_seq_lengths_kv`** (`Tensor`): Optional. Valid token count of `key` and `value` in different batches. The data type can be `torch.int32`. If omitted, you can set it to `None`, indicating that its length is identical to the size of the `S` dimension in the shape of `key`. The valid token count for each batch must not exceed the size of the S dimension in `key` and `value` and must be greater than or equal to `0`. This parameter must be a 1D tensor of length `B`.<br>When `layout_kv` is set to `TND` or `PA_BSND`, this parameter must be provided. When `layout_kv` is set to `TND`, each element indicates the prefix sum of token counts across batches, and the value of each element must be greater than or equal to that of the preceding element.

- **`query_rope`** (`Tensor`): Optional. RoPE information for the query in the MLA structure. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `torch.bfloat16` or `torch.float16`.

- **`key_rope`** (`Tensor`): Optional. RoPE information for the key in the MLA structure. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `torch.bfloat16` or `torch.float16`.

- **`sparse_block_size`** (`int`): Optional. Block size used in the sparse phase during the importance score computation. The data type can be `int64`. The value must be a power of 2 in the range of [1, 128].
    - When sparse_block_size is 1, it is the token-wise sparsification scenario, in which each token is treated as an independent unit, and the independent correlation between each query token and each key-value token is evaluated when computing the importance score.
    - When sparse_block_size is greater than 1 and less than or equal to 128, it is the block-wise sparsification scenario, in which the token sequence is divided into consecutive blocks of a fixed size, the importance is evaluated in units of blocks, and tokens in a block share the same sparsification decision.

- **`layout_query`** (`str`): Optional. Data layout of the input `query`. Valid values are `BSND` or `TND`. If omitted, the default value `BSND` is used.

- **`layout_kv`** (`str`): Optional. Data layout of the input `key`. Valid values are `TND`, `BSND`, or `PA_BSND`. If omitted, the default value `BSND` is used. `PA_BSND` is used when PagedAttention is enabled.

- **`sparse_mode`** (`int`): Optional. Sparse mode. The data type can be `int64`.
    - When sparse\_mode is 0, all computations are performed.
    - When sparse\_mode is 3, it indicates the mask of the rightDownCausal mode, corresponding to the lower-triangular scenario divided along the line from the bottom-right vertex to the top-left.

- **`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. Only the default value `9223372036854775807` is supported.

- **`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. Only the default value `9223372036854775807` is supported.

- **`attention_mode`** (`int`): Optional. Indicates the attention mode, passed as a Python `int`. Only the value `2` is supported, indicating the MLA-absorb mode. In this mode, computations concatenate the `nope` portion of the query or key with the rope portion of `query_rope` or `key_rope` along the head dimension `D`. This merges them to form the final query or key for subsequent computations. The key or value shares the identical underlying tensor data.

- **`return_softmax_lse`** (`bool`): Optional. Specifies whether to return `softmax_max` or `softmax_sum`. Valid values are `True` (enables the return but does not support graph capture mode) or `False` (disables the return). The default value is `False`. This parameter is supported only during training when `layout_kv` is not `PA_BSND`.

- **`sinks`** (`Tensor`): Optional. Sinks information in the attention structure. When this parameter is passed, the `aclnnSparseFlashAttentionV2` API is called; when it is not passed, the original `aclnnSparseFlashAttention` API is called. The data layout can be ND, the data type can be `torch.float32`, and the shape is [N1].

## Return Values

- **`attention_out`** (`Tensor`): Output in the formula. The data layout can be ND. The data type can be `torch.bfloat16` or `torch.float16`. When `layout_query` is `BSND`, the shape must be `[B, S1, N1, D]`. When `layout_query` is `TND`, the shape must be `[T1, N1, D]`.
- **`softmax_max`** (`Tensor`): Optional output. Obtained by taking the maximum value of the query-key product in the attention computation. The data type can be `torch.float`. When `layout_query` is `BSND`, the shape must be `[B, N2, S1, N1/N2]`. When `layout_query` is `TND`, the shape must be `[N2, T1, N1/N2]`.
- **`softmax_sum`** (`Tensor`): Optional output. Obtained by subtracting `softmax_max` from the query-key product in the attention computation, applying `exp`, and then calculating the sum. The data type can be `torch.float`. When `layout_query` is `BSND`, the shape must be `[B, N2, S1, N1/N2]`. When `layout_query` is `TND`, the shape must be `[N2, T1, N1/N2]`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- The D value in the parameter query is equal to that of key and value, which is 512, and the D value in the parameter query\_rope is equal to that of key\_rope, which is 64.
- The data types of the parameters query, key, and value must be the same.
- `value=None` is supported only when CANN >= 9.2.0; lower versions raise a `NOT_SUPPORT` error.
- sparse\_block\_size must divide block\_size.
- When `layout_kv` is PA_BSND, `layout_query` and `layout_kv` do not need to be the same; when `layout_kv` is BSND or TND, `layout_query` and `layout_kv` must be the same.

<!-- npu="950" id4 -->
- Ascend 950PR&950DT products:
  - Q\_N supports 1 to 128.
  - Only 1 is supported for sparse\_block\_size.
  - Only when `layout_kv` is PA_BSND do key, value, and key_rope support a non-contiguous 0th axis.
<!-- end id4 -->
<!-- npu="A3,910b" id5 -->
- Atlas A3 inference products/Atlas A2 inference products:
  - Q\_N supports 1/2/4/8/16/32/64/128.
  - The parameter sparse\_block\_size supports [1,128] and must be a power of 2. In the PagedAttention scenario, sparse\_block\_size must divide block\_size.
  - sinks is not supported.
  - Non-contiguous key, value, and key_rope are not supported.
<!-- end id5 -->

## Example

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
    # Note: random.sample is used here only for demonstration purposes. In a production environment, a sparse selection algorithm (such as lightning_indexer) should be used to generate sparse_indices.
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
