# torch_npu.npu_block_sparse_attention

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- Function: BlockSparseAttention, that is, sparse attention computation. It supports the block-level sparse mode, in which `block_sparse_mask` specifies the KV blocks selected by each Q block, implementing efficient sparse attention.

- Formulas: The sparse block size is blockShapeX × blockShapeY, and `block_sparse_mask` specifies the sparse mode:

$$
attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T) \cdot value_{sparse}
$$

## Prototype

```python
torch_npu.npu_block_sparse_attention(query, key, value, block_sparse_mask, block_shape, *, q_input_layout='TND', kv_input_layout='TND', num_key_value_heads=1, scale_value=0.0, inner_precise=1, actual_seq_lengths=None, actual_seq_lengths_kv=None, softmax_lse_flag=0) -> (Tensor, Tensor)
```

## Parameters

- **`query`** (`Tensor`): Required. Query in the attention, $query$ in the formula. The data layout can be ND. The data type can be `torch.float16` or `torch.bfloat16`.
  - TND: the shape is `[totalQTokens, headNum, headDim]`;
  - BNSD: the shape is `[batch, headNum, maxQSeqLength, headDim]`;

- **`key`** (`Tensor`): Required. Key in the attention, $key$ in the formula. The data layout can be ND. The data type must be identical to that of `query`.
  - TND: the shape is `[totalKTokens, numKeyValueHeads, headDim]`;
  - BNSD: the shape is `[batch, numKeyValueHeads, maxKvSeqLength, headDim]`;

- **`value`** (`Tensor`): Required. Value in the attention, $value$ in the formula. The shape and data type must be identical to those of `key`.

- **`block_sparse_mask`** (`Tensor`): Required. Block sparse mask. The shape of this parameter is `[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]`. It indicates the blocks that participate in the computation after block partitioning. A value of `1` indicates that the corresponding block participates in the attention computation, and a value of `0` indicates it is excluded from computation. The data type is `torch.int8`.

- **`block_shape`** (`list[int]`): Required. Block shape for sparse computation. It must contain at least two elements, such as `[blockShapeX, blockShapeY]`, and all values must be greater than 0. `blockShapeX` represents the block size along the Q dimension, and `blockShapeY` represents the block size along the KV dimension. **`blockShapeY` must be a multiple of 128.**
  <!-- npu="A3,910b" id4 -->
  - On <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: **blockShapeY must be a multiple of 128**.
  <!-- end id4 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950DT products</term>: **blockShapeY must be a multiple of 16**.
  <!-- end id8 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`q_input_layout`** (`str`): Optional. Layout of `query`. The default value is `"TND"`. Currently, only `"TND"` and `"BNSD"` are supported.

- **`kv_input_layout`** (`str`): Optional. Layout of `key` and `value`. The default value is `"TND"`. Currently, only `"TND"` and `"BNSD"` are supported, and it must match `q_input_layout`.

- **`num_key_value_heads`** (`int`): Optional. Number of `key` and `value` attention heads. The default value is `1`.

- **`scale_value`** (`float`): Optional. Scaling factor. The default value is `0.0`, and it is typically set to $D^{-0.5}$.

- **`inner_precise`** (`int`): Optional. Softmax precision mode. The default value is `1`. Valid values are `0` (high accuracy mode using `torch.float32` intermediate results) or `1` (high performance mode using `torch.float16` intermediate results). **When `query`, `key`, or `value` uses the `torch.bfloat16` data type, only `0` is supported.**
  <!-- npu="950" id5 -->
  - <term>Ascend 950DT products</term>: only `4` is supported.
  <!-- end id5 -->
  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: only `0` or `1` is supported.
  <!-- end id9 -->
- **`actual_seq_lengths`** (`list[int]`): Optional. Actual sequence lengths of `query` per batch, used in variable-length sequence scenarios.
  - **Required when `q_input_layout` is `"TND"`**: in TND, the shape of `query` is `[totalQTokens, headNum, headDim]`. If there is no batch dimension, the operator cannot infer the `query` length of each batch from the shape.
  - **Optional when `q_input_layout` is `"BNSD"`**: in BNSD, the shape of `query` is `[batch, headNum, maxQSeqLength, headDim]`. If this parameter is not passed, the operator uses S (maxQSeqLength) in the shape as the sequence length. If it is passed, the operator uses the actual length specified by this parameter.

- **`actual_seq_lengths_kv`** (`list[int]`): Optional. Actual sequence lengths of `key` and `value` per batch, used in variable-length sequence scenarios.
  - **Required when `kv_input_layout` is `"TND"`**: in TND, the shape of `key`/`value` is `[totalKvTokens, numKeyValueHeads, headDim]`. If there is no batch dimension, the operator cannot infer the kv length of each batch from the shape.
  - **Optional when `kv_input_layout` is `"BNSD"`**: in BNSD, the shape of `key`/`value` is `[batch, numKeyValueHeads, maxKvSeqLength, headDim]`. If this parameter is not passed, the operator uses S in the shape as the sequence length. If it is passed, the operator uses the actual length specified by this parameter.

- **`softmax_lse_flag`** (`int`): Optional. Specifies whether to output the `softmax_lse` tensor. The default value is `0`. Valid values are `0` (does not output `softmax_lse`) or `1` (outputs `softmax_lse`, which may cause performance degradation).

## Return Values

- **`attention_out`** (`Tensor`): $attentionOut$ in the formula. The data type and layout must be identical to those of `query`, and the last dimension matches the `headDim` of `value`.
- **`softmax_lse`** (`Tensor`): Log-sum-exp intermediate result of Softmax computation. The data type is `torch.float32`. This tensor is returned only when `softmax_lse_flag` is set to `1`.
  - For TND, the shape is `[totalQTokens, headNum, 1]`;
  - For BNSD, the shape is `[batch, headNum, maxQSeqLength, 1]`.

## Constraints

- The data types of `query`, `key`, and `value` must be the same, and must be `torch.float16` or `torch.bfloat16`.
- The number of heads $N1$ of `query` and the number of heads $N2$ of `key`/`value` must satisfy $N1 ≥ N2$ and $N1 \% N2 = 0$.
- `actual_seq_lengths` and `actual_seq_lengths_kv` must currently be configured together or not configured at all. Configuring only one of them is intercepted by the operator.
- The sequence length does not need to be divisible by `block_shape`. The number of blocks is calculated by rounding up.
- The forward path currently supports headDim=64 or 128. The backward path currently supports headDim=128.
- The backward path supports `q_input_layout` and `kv_input_layout` both being `"BNSD"` or both being `"TND"`, and supports MHA and GQA scenarios. In the MHA scenario, $N1 = N2$. In the GQA scenario, $N1 > N2$ and $N1 \% N2 = 0$ must be satisfied, where $N1$ is the number of heads of `query` and $N2$ is the number of heads of `key`/`value`.

## Example

Single-operator call:

- BNSD layout

    ```python
    import torch
    import torch_npu

    B, N, S, D = 2, 8, 32, 64
    num_kv_heads = 8
    scale_value = 1.0 / (D ** 0.5)
    block_shape = [128, 128]  # blockShapeY must be a multiple of 128
    ceil_q = (S + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (S + block_shape[1] - 1) // block_shape[1]

    query = torch.randn(B, N, S, D, dtype=torch.float16).npu()
    key = torch.randn(B, num_kv_heads, S, D, dtype=torch.float16).npu()
    value = torch.randn(B, num_kv_heads, S, D, dtype=torch.float16).npu()
    block_sparse_mask = torch.ones(B, N, ceil_q, ceil_kv, dtype=torch.int8).npu()

    attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
        query, key, value, block_sparse_mask, block_shape,
        q_input_layout="BNSD", kv_input_layout="BNSD",
        num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
    )
    print(attention_out.shape)  # (B, N, S, D)
    ```

- TND layout

    ```python
    import torch
    import torch_npu

    T, N, D = 32, 8, 64
    num_kv_heads = 8
    scale_value = 1.0 / (D ** 0.5)
    block_shape = [128, 128]  # blockShapeY must be a multiple of 128
    ceil_q = (T + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (T + block_shape[1] - 1) // block_shape[1]

    query = torch.randn(T, N, D, dtype=torch.float16).npu()
    key = torch.randn(T, num_kv_heads, D, dtype=torch.float16).npu()
    value = torch.randn(T, num_kv_heads, D, dtype=torch.float16).npu()
    block_sparse_mask = torch.ones(1, N, ceil_q, ceil_kv, dtype=torch.int8).npu()

    attention_out, softmax_lse = torch_npu.npu_block_sparse_attention(
        query, key, value, block_sparse_mask, block_shape,
        q_input_layout="TND", kv_input_layout="TND",
        num_key_value_heads=num_kv_heads, scale_value=scale_value, inner_precise=1,
        actual_seq_lengths=[T], actual_seq_lengths_kv=[T],
    )
    print(attention_out.shape)  # (T, N, D)
    ```
