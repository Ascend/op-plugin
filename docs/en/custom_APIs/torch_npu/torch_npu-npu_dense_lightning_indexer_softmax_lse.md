# torch_npu.npu_dense_lightning_indexer_softmax_lse

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                          | Supported|
|------------------------------| :------: |
| <term>Atlas A3 training products</term>| √  |
| <term>Atlas A2 training products</term>| √  |

## Function

- Description: Optimizes device memory usage as a frontend API of `npu_dense_lightning_indexer_grad_kl_loss` by precomputing the maximum and sum values used in the Softmax computation of the Lightning Indexer component.

- Formulas:

$$
res=\text{AttentionMask}\left(\text{ReduceSum}\left(W\odot\text{ReLU}\left(\tilde{Q}@\tilde{K}^T\right)\right)\right)
$$

$$
maxIndex=\text{max}\left(res\right)
$$

$$
sumIndex=\text{ReduceSum}\left(\text{exp}\left(res-maxIndex\right)\right)
$$

The output tensors $maxIndex$ and $sumIndex$ are passed to the downstream `npu_dense_lightning_indexer_grad_kl_loss` API as inputs for Softmax computation.

## Prototype

```python
npu_dense_lightning_indexer_softmax_lse(query_index, key_index, weights, *, actual_seq_qlen=None, actual_seq_klen=None, layout='BSND', sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1) -> (Tensor, Tensor)
```

## Parameters

**`query_index`** (`Tensor`): Required. Forward input query of the Lightning Indexer, $\tilde{Q}$ in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S1, N1index, D)` or `(T1, N1index, D)`.

**`key_index`** (`Tensor`): Required. Forward input key of the Lightning Indexer, $\tilde{K}$ in the formula. The data layout can be ND. The data type can be `bfloat16` or `float16`. The shape can be `(B, S2, N2index, D)` or `(T2, N2index, D)`.

**`weights`** (`Tensor`): Required. Weight coefficients of the Lightning Indexer, $W$ in the formula. The data layout can be ND. The data type can be `bfloat16`, `float16`, or `float32`. The shape can be `(B, S1, N1index)` or `(T1, N1index)`.

**`actual_seq_qlen`** (`list[int]`): Optional. Accumulated sum of sequence lengths for each $S$ in the query tensor. This parameter is required in the `TND` scenario. The data type can be `int64`. The data layout can be ND. The default value is `None`.

**`actual_seq_klen`** (`list[int]`): Optional. Accumulated sum of sequence lengths for each $S$ in the key tensor. This parameter is required in the TND scenario. The data type can be `int64`. The data layout can be ND. The default value is `None`.

**`layout`** (`str`): Optional. Data layout format of the input `query_index`. Valid values are `BSND` or `TND`. The default value is `BSND`.

**`sparse_mode`** (`int`): Optional. Sparse mode. The data type can be `int32`. The default value is `3`. Currently, only mode `3` is supported.

**`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2^{63}-1`.

**`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2^{63}-1`.

## Return Values

- **`softmax_max_index`** (`Tensor`): Maximum value used in Softmax computation, $maxIndex$ in the formulas. The data layout can be ND. The data type can be `float32`.
- **`softmax_sum_index`** (`Tensor`): Sum value used in Softmax computation, $sumIndex$ in the formula. The data layout can be ND. The data type can be `float32`.
The data layout can be ND. The data type can be `float32`.

## Constraints

- The data types of `query_index` and `key_index` must be identical.
- When the data type of `weights` is not `float32`, the data types of `query_index`, `key_index`, and `weights` must be identical.
- Shape variable constraints:

| Item   | Value      | Description        |
|-----------|------------|-----------------|
| B         | 1 to 256     | -               |
| S1, S2   | 1 to 128K    | `S1` and `S2` can have different lengths. When `layout` is `BSND`, the condition $S1 \le S2$ must be satisfied. When `layout` is `TND`, `actual_seq_qlen[i]` must be less than or equal to `actual_seq_klen[i]` at each corresponding index, and $S1 \le S2$ must also be satisfied.|
| N1index   | 8, 16, 32, 64| -               |
| N2index   | 1          | -               |
| D         | 128        | -               |

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    def gen_inputs(isTnd=False):
        B = 20
        N1 = 32
        N2 = 1
        S1 = 511
        S2 = 2049
        D = 128

        output_dtype = torch.float16
        query_index = torch.randn(B, S1, N1, D, dtype=output_dtype, device=torch.device('npu'))
        key_index = torch.randn(B, S2, N2, D, dtype=output_dtype, device=torch.device('npu'))
        weights = torch.randn(B, S1, N1, dtype=output_dtype, device=torch.device('npu'))
        if isTnd:
            query_index = query_index.reshape(B*S1, N1, D)
            key_index = key_index.reshape(B*S2, N2, D)
            weights = weights.reshape(B*S1, N1)
            layout = 'TND'
            actual_seq_qlen = [S1*(i+1) for i in range(B)]
            actual_seq_klen = [S2*(i+1) for i in range(B)]
        else :
            layout = 'BSND'
            actual_seq_qlen = None
            actual_seq_klen = None

        sparse_mode = 3
        pre_tokens = 9223372036854775807
        next_tokens = 9223372036854775807

        return query_index, key_index, weights, actual_seq_qlen, actual_seq_klen, layout, sparse_mode, pre_tokens, next_tokens

    query_index, key_index, weights, actual_seq_qlen, actual_seq_klen, layout, sparse_mode, pre_tokens, next_tokens = gen_inputs(isTnd=False)
    softmax_max_index, softmax_sum_index = torch_npu.npu_dense_lightning_indexer_softmax_lse(
        query_index, 
        key_index, 
        weights, 
        actual_seq_qlen=actual_seq_qlen, 
        actual_seq_klen=actual_seq_klen, 
        layout=layout, 
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens)
    ```
