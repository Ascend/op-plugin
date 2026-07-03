# torch_npu.npu_block_sparse_attention

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E).

## Supported Products

| Product                                                  | Supported|
|:-----------------------------------------------------|:----:|
| Atlas A3 training products/Atlas A3 inference products                     |  √   |
| Atlas A2 training products/Atlas 800I A2 inference products/Atlas 200I A2 Box heterogeneous components|  √   |

## Function

- Description: Computes `BlockSparseAttention`, a sparse attention mechanism that supports block-level sparsity. It uses `block_sparse_mask` to specify the KV blocks selected by each Q block to achieve efficient attention computation.

- Formula: The sparse block size is $blockShapeX \times blockShapeY$. `block_sparse_mask` defines the sparsity pattern.

$$
attentionOut = Softmax(scale \cdot query \cdot key_{sparse}^T) \cdot value_{sparse}
$$

## Prototype

```python
torch_npu.npu_block_sparse_attention(query, key, value, block_sparse_mask, block_shape, *, q_input_layout='TND', kv_input_layout='TND', num_key_value_heads=1, scale_value=0.0, inner_precise=1, actual_seq_lengths=None, actual_seq_lengths_kv=None, softmax_lse_flag=0) -> (Tensor, Tensor)
```

## Parameters

- **`query`** (`Tensor`): Required. Query in the attention, $query$ in the formula. The data layout can be ND. The data type can be `float16` or `bfloat16`.
  - In `TND` layout: The shape is `[totalQTokens, headNum, headDim]`.
  - In `BNSD` layout: The shape is `[batch, headNum, maxQSeqLength, headDim]`.

- **`key`** (`Tensor`): Required. Key in the attention, $key$ in the formula. The data layout can be ND. The data type must be identical to that of `query`.
  - `TND`: The shape is `[totalKTokens, numKeyValueHeads, headDim]`.
  - `BNSD`: The shape is `[batch, numKeyValueHeads, maxKvSeqLength, headDim]`.

- **`value`** (`Tensor`): Required. Value in the attention, $value$ in the formula. The shape and data type must be identical to those of `key`.

- **`block_sparse_mask`** (`Tensor`): Required. Block sparse mask. The shape of this parameter is `[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]`. It indicates the blocks that participate in the computation after block partitioning. A value of `1` indicates that the corresponding block participates in the attention computation, and a value of `0` indicates it is excluded from computation. The data type is `int8`.

- **`block_shape`** (`list[int]`): Required. Block shape for sparse computation. It must contain at least two elements, such as `[blockShapeX, blockShapeY]`, and all values must be greater than 0. `blockShapeX` represents the block size along the Q dimension, and `blockShapeY` represents the block size along the KV dimension. **`blockShapeY` must be a multiple of 128.**

- **`*`**: Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).

- **`q_input_layout`** (`str`): Optional. Layout of `query`. The default value is `"TND"`. Currently, only `"TND"` and `"BNSD"` are supported.

- **`kv_input_layout`** (`str`): Optional. Layout of `key` and `value`. The default value is `"TND"`. Currently, only `"TND"` and `"BNSD"` are supported, and it must match `q_input_layout`.

- **`num_key_value_heads`** (`int`): Optional. Number of `key` and `value` attention heads. The default value is `1`.

- **`scale_value`** (`float`): Optional. Scaling factor. The default value is `0.0`, and it is typically set to $D^{-0.5}$.

- **`inner_precise`** (`int`): Optional. Softmax precision mode. The default value is `1`. Valid values are `0` (high accuracy mode using `float32` intermediate results) or `1` (high performance mode using `float16` intermediate results). **When `query`, `key`, or `value` uses the `bfloat16` data type, only `0` is supported.**

- **`actual_seq_lengths`** (`list[int]`): Optional. Actual sequence lengths of `query` per batch, used in variable-length sequence scenarios.
  - **Required when `q_input_layout` is `"TND"`**: In `TND` layout, the `query` shape is `[totalQTokens, headNum, headDim]`. Without the batch dimension, the operator cannot infer the sequence length of each batch from the shape alone.
  - **Optional when `q_input_layout` is `"BNSD"`**: In `BNSD` layout, if this parameter is omitted, `maxQSeqLength` from the shape of `query` `[batch, headNum, maxQSeqLength, headDim]` is used as the sequence length; if provided, execution follows the actual lengths specified by this parameter.

- **`actual_seq_lengths_kv`** (`list[int]`): Optional. Actual sequence lengths of `key` and `value` per batch, used in variable-length sequence scenarios.
  - **Required when `kv_input_layout` is `"TND"`**: In `TND` layout, the `key`/`value` shape is `[totalKvTokens, numKeyValueHeads, headDim]`. Without the batch dimension, the operator cannot infer the sequence length of each batch from the shape alone.
  - **Optional when `kv_input_layout` is `"BNSD"`**: In BNSD layout, if this parameter is omitted, `maxKvSeqLength` from the shape of `key`/`value` is used as the sequence length; if provided, execution follows the actual lengths specified by this parameter.

- **`softmax_lse_flag`** (`int`): Optional. Specifies whether to output the `softmax_lse` tensor. The default value is `0`. Valid values are `0` (does not output `softmax_lse`) or `1` (outputs `softmax_lse`, which may cause performance degradation).

## Return Values

- **`attention_out`** (`Tensor`): $attentionOut$ in the formula. The data type and layout must be identical to those of `query`, and the last dimension matches the `headDim` of `value`.
- **`softmax_lse`** (`Tensor`): Log-sum-exp intermediate result of Softmax computation. The data type is `float32`. This tensor is returned only when `softmax_lse_flag` is set to `1`.
  - In `TND` layout, the shape is `[totalQTokens, headNum, 1]`.
  - In `BNSD` layout, the shape is `[batch, headNum, maxQSeqLength, 1]`.

## Constraints

- The data types of `query`, `key`, and `value` must be identical, and the data type can be `float16` or `bfloat16`.
- The number of heads for `query` ($N1$) and the number of heads for `key`/`value` ($N2$) must satisfy the conditions $N1 \ge N2$ and $N1 \% N2 = 0$.
- `actual_seq_lengths` and `actual_seq_lengths_kv` must be either both provided or both omitted. Providing only one of these parameters will be rejected by the operator.
- The sequence length does not need to be perfectly divisible by `block_shape`. The total number of blocks is calculated using ceiling division.
- The forward path supports `headDim = 64` or `128`. The backward path supports `headDim = 128` only.
- The backward path supports both "BNSD" and "TND" layouts for `q_input_layout` and `kv_input_layout`, and supports both MHA and GQA scenarios. In MHA scenarios, $N1 = N2$. In GQA scenarios, $N1 > N2$ and $N1 \bmod N2 = 0$, where $N1$ denotes the number of heads in `query`, and $N2$ denotes the number of heads in `key`/`value`.

## Examples

Single-operator call

- `BNSD` layout

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

- `TND` layout

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
