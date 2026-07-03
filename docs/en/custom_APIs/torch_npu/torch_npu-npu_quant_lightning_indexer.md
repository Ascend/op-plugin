# torch_npu.npu_quant_lightning_indexer

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>   |  √  |
|<term>Atlas A3 inference products</term>  | √  |
|<term>Atlas A2 inference products</term>  | √  |

## Function

- Description: Performs preprocessing computation for SparseFlashAttention (SFA) in inference scenarios. This API selects key sparse tokens and quantizes the input `query` and `key` to implement INT8 storage and INT8 computation to maximize performance gains.

- Formula:
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{INT8}@{\left(K_{index}^{INT8}\right)}^T\right)\right)\right]\right\}$$
    The main computation process is as follows:
    1. Multiply the input parameter `query` ($Q_{index}^{INT8}\in\R^{g\times d}$) corresponding to a token by the given context `key` ($K_{index}^{INT8}\in\R^{S_{k}\times d}$) to obtain the correlation.
    2. Multiply the correlation results by the dequantization coefficients `query_dequant_scale` ($Scale_Q$) and `key_dequant_scale` ($Scale_K^T$) corresponding to `query` and `key`. Invalid negative correlation signals are filtered by the $\text{ReLU}$ activation function to obtain a correlation score vector between the current token and all preceding tokens.
    3. Multiply the vector result by the weight coefficient `weights` ($W$). Then, select the Top-K indices along the `g` dimension to obtain the output `out`, which serves as the input to SparseFlashAttention.

## Prototype

```python
torch_npu.npu_quant_lightning_indexer(query, key, weights, query_dequant_scale, key_dequant_scale, query_quant_mode, key_quant_mode, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query='BSND', layout_key='BSND', sparse_count=2048, sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1,
query_dtype=None, key_dtype=None) -> Tensor
```

## Parameters
>
> [!NOTE]
>
> - Dimension definitions for the `query`, `key`, `weights`, `query_dequant_scale`, and `key_dequant_scale`:<br>`B` (`Batch Size`) indicates the input sample batch size.<br>`S` (`Sequence Length`) indicates the input sample sequence length.<br>`H` (`Head Size`) indicates the hidden layer size.<br>`N` (`Head Num`) indicates the number of heads.<br>`D` (`Head Dim`) indicates the minimum unit size of the hidden layer, satisfying `D = H/N`.<br>`T` indicates the cumulative sum of the sequence lengths of all batch input samples.
> - `S1` and `S2` indicate the input sequence lengths of `query` and `key`, respectively.<br>`N1` and `N2` indicate the head counts corresponding to `query` and `key`, respectively.<br>`k` indicates the number of finally selected indices. The size of the `D` dimension in both `query` and `key` must be identical and equal to `128`. `T1` and `T2` indicate the cumulative sums of input sequence lengths for `query` and `key`, respectively.
>
- **`query`** (`Tensor`): Required. Input index query, $Q_{index}^{INT8}\in\R^{g\times d}$ in the formula. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `int8`, `float8_e4m3fn`, or `hifloat8`. When `layout_query` is `"BSND"`, the shape of this parameter is `(B, S1, N1, D)`. When `layout_query` is `"TND"`, the shape is `(T1, N1, D)`. The value range of `N1` is [1, 64].

- **`key`** (`Tensor`): Required. Input index key, $K_{index}^{INT8}\in\R^{S_{k}\times d}$ in the formula. Non-contiguous tensors are supported. The data layout can be ND. The data type can be `int8`, `float8_e4m3fn`, or `hifloat8`. When `layout_key` is `"PA_BSND"`, the shape of this parameter is `(block_count, block_size, N2, D)`, where `block_count` represents the total block count in PageAttention, and `block_size` represents the token count within a single block. The value of `block_size` must be divisible by `16`, and must be less than or equal to `1024`. When `layout_key` is `BSND`, the shape of this parameter is `[B, S2, N2, D]`. When `layout_key` is `TND`, the shape of this parameter is `[T2, N2, D]`. The value of `N2` must be `1`.

- **`weights`** (`Tensor`): Required. Weight coefficient, $W$ in the formula. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `float16` or `bfloat16`. The shape can be `(B, S1, N1)` or `(T, N1)`.

- **`query_dequant_scale`** (`Tensor`): Required. Dequantization coefficient of the index query, $Scale_Q$ in the formula. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `float16` or `float32`. The shape can be `(B, S1, N1)` or `(T, N1)`.

- **`key_dequant_scale`** (`Tensor`): Required. Dequantization coefficient of the index key, $Scale_K^T$ in the formula. Non-contiguous tensors are supported. The data layout can be ND. The data type can be `float16` or `float32`. When `layout_key` is `"PA_BSND"`, the shape of this parameter is `(block_count, block_size, N2)`, where `block_count` represents the total block count in PageAttention, and `block_size` represents the token count within a single block.

- **`query_quant_mode`** (`int`): Optional. Quantization mode for `query`. Currently, only the value `0` (`pertoken-head` mode) is supported.

- **`key_quant_mode`** (`int`): Optional. Quantization mode for `key`. Currently, only the value `0` (`pertoken-head` mode) is supported.

- **`*`**: Positional argument separator. Arguments preceding it are positional and must be provided in order. Arguments following it are optional. If omitted, default values are used.

- **`actual_seq_lengths_query`** (`Tensor`): Optional. Valid token count of `query` in different batches. The data type can be `int32`. If not specified, this parameter can be set to `None` to indicate that its length is identical to the size of the S dimension in the shape of `query`. The valid token count for each batch must not exceed the size of the S dimension in `query` and must be greater than or equal to 0. This parameter must be a 1D tensor of length `B`. When `layout_query` is `TND`, this parameter must be provided, where its element count determines the batch size `B`, and each element indicates the cumulative token count of the current batch and all preceding batches, representing a prefix sum. Therefore, the value of each element must be greater than or equal to that of the preceding element. Negative values are not allowed.

- **`actual_seq_lengths_key`** (`Tensor`): Optional. Valid token count of `key` in different batches. The data type can be `int32`. If this parameter is not specified or is set to `None`, its length is identical to the size of the `S` dimension in the shape of `key`. The valid token count for each batch must not exceed the size of the S dimension in `key` and `value` and must be greater than or equal to `0`. This parameter must be a 1D tensor of length `B`. When `layout_key` is set to `TND` or `PA_BSND`, this parameter must be provided. When `layout_key` is set to `TND`, each element indicates the prefix sum of token counts across batches, and the value of each element must be greater than or equal to that of the preceding element.

- **`block_table`** (`Tensor`): Optional. Block mapping table used for KV storage in PageAttention. The data layout is ND. The data type can be `int32`. In PageAttention scenarios, `block_table` must be a 2D tensor. Its first dimension must equal `B`, and its second dimension must be no less than `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actual_seq_lengths_key` across batches). The value of `block_size` must be divisible by `16`, and must be less than or equal to `1024`.

- **`layout_query`** (`str`): Optional. Layout format for the data arrangement of the input `query`. Valid values are `BSND` or `TND`. The default value is `BSND`.

- **`layout_key`** (`str`): Optional. Data layout configuration of the input `key`. Valid values are `"PA_BSND"`, `"BSND"`, or `"TND"`. The default value is `"BSND"`. In non-PageAttention scenarios, the value of this parameter must be identical to that of `layout_query`.

- **`sparse_count`** (`int`): Optional. Number of blocks retained during the Top-K stage. The value range is [1, 2048]. The data type can be `int32`.

- **`sparse_mode`** (`int`): Optional. Sparsification mode. The data type can be `int32`. Valid values: `0`: enables `defaultMask` mode. `3`: enables `rightDownCausal` mode mask, corresponding to lower triangular scenarios where the dividing line extends from the right vertex.

- **`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. Only the default value `2^63 - 1` is supported.

- **`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. Only the default value `2^63 - 1` is supported.

- **`query_dtype`** (`int`): Optional. Actual data type of `query`. Valid values are `int8`, `float8_e4m3fn`, or `hifloat8`.

- **`key_dtype`** (`int`): Optional. Actual data type of `key`. Valid values are `int8`, `float8_e4m3fn`, or `hifloat8`.

## Return Values

`Tensor`

$out$ in the formula. The data layout can be ND. The data type can be `int32`. The output shape can be `(B, S1, N2, k)` or `(T, N2, k)`.

## Constraints

- This API supports graph mode.
- The result of $W \odot Scale_Q$ must be within the representable range of `float16`.
- Sorting NaN values during the TopK process is undefined behavior.
- Atlas A3 inference products:
    - The data types of `query` and `key` can be `int8`.
    - Only the data type combination of (`float16`, `float16`, `float16`) is supported for `weights`, `query_dequant_scale`, and `key_dequant_scale`.
- Ascend 950PR/Ascend 950DT:
    - Valid values for `N1` in `query` are `8`, `16`, `24`, `32`, or `64`.
    - The data types of `query` and `key` can be `float8_e4m3fn`, `hifloat8`, or `int8`.
    - When the data types of `query` and `key` are `float8_e4m3fn`, data types of `weights`, `query_dequant_scale`, and `key_dequant_scale` can be (`bfloat16`, `float`, `float`) or (`float16`, `float16`, `float16`).
    - When the data types of `query` and `key` are `hifloat8`, only the data types (`bfloat16`, `float`, `float`) are supported for `weights`, `query_dequant_scale`, and `key_dequant_scale`.
    - When the data types of `query` and `key` are `int8`, only the data types (`float16`, `float16`, `float16`) are supported for `weights`, `query_dequant_scale`, and `key_dequant_scale`.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math

    n1 = 64
    n2 = 1
    d = 128
    block_size = 128
    layout_key = "PA_BSND"
    layout_query = "BSND"
    query_quant_mode = 0
    key_quant_mode = 0
    np.random.seed(0)
    # -------------
    b = 24
    t = None
    s1 = 4
    s2 = 512
    act_seq_q = None
    act_seq_k = None
    sparse_mode = 0
    sparse_count = 2048
    max_block_table_num = (s2 + block_size - 1) // block_size
    block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
    key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
    key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
    key_dequant_scale = key_dequant_scale.to(torch.float16)
    query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
    query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
    weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)

    npu_out = torch_npu.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),
                                                    key_dequant_scale.npu(),
                                                    actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                    actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                    block_table=block_table.npu(),
                                                    query_quant_mode=query_quant_mode,
                                                    key_quant_mode=key_quant_mode,
                                                    layout_query=layout_query,
                                                    layout_key=layout_key, sparse_count=sparse_count,
                                                    sparse_mode=sparse_mode)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import numpy as np
    import torch.nn as nn
    import math

    n1 = 64
    n2 = 1
    d = 128
    block_size = 128
    layout_key = "PA_BSND"
    layout_query = "BSND"
    query_quant_mode = 0
    key_quant_mode = 0
    np.random.seed(0)
    # -------------
    b = 24
    t = None
    s1 = 4
    s2 = 512
    act_seq_q = None
    act_seq_k = None
    sparse_mode = 0
    sparse_count = 2048
    max_block_table_num = (s2 + block_size - 1) // block_size
    block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
    key = torch.tensor(np.random.uniform(-128, 127, (b * max_block_table_num, block_size, n2, d))).to(torch.int8)
    key_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b * max_block_table_num, block_size, n2)))
    key_dequant_scale = key_dequant_scale.to(torch.float16)
    query = torch.tensor(np.random.uniform(-128, 127, (b, s1, n1, d))).to(torch.int8)
    query_dequant_scale = torch.tensor(np.random.uniform(0, 10, (b, s1, n1))).to(torch.float16)
    weights = torch.tensor(np.random.uniform(0, 0.01, (b, s1, n1))).to(torch.float16)
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32) \
                                if act_seq_q is None else torch.tensor(act_seq_q).to(torch.int32)
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32) \
                                if act_seq_k is None else torch.tensor(act_seq_k).to(torch.int32)

    class LIQuantNetwork(nn.Module):
        def __init__(self):
            super(LIQuantNetwork, self).__init__()

        def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query=None,
                    actual_seq_lengths_key=None, block_table=None, query_quant_mode=0, key_quant_mode=0,
                    layout_query='BSND', layout_key='BSND', sparse_count=2048, sparse_mode=3):

            out = torch_npu.npu_quant_lightning_indexer(query.npu(), key.npu(), weights.npu(), query_dequant_scale.npu(),
                                                        key_dequant_scale.npu(),
                                                        actual_seq_lengths_query=actual_seq_lengths_query.npu(),
                                                        actual_seq_lengths_key=actual_seq_lengths_key.npu(),
                                                        block_table=block_table.npu(),
                                                        query_quant_mode=query_quant_mode,
                                                        key_quant_mode=key_quant_mode,
                                                        layout_query=layout_query,
                                                        layout_key=layout_key, sparse_count=sparse_count,
                                                        sparse_mode=sparse_mode)
            return out

    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(LIQuantNetwork().npu(), fullgraph=True, backend=npu_backend, dynamic=False)
    npu_out = npu_mode(query, key, weights, query_dequant_scale, key_dequant_scale,
                        actual_seq_lengths_query=actual_seq_lengths_query,
                        actual_seq_lengths_key=actual_seq_lengths_key,
                        block_table=block_table, query_quant_mode=query_quant_mode,
                        key_quant_mode=key_quant_mode, layout_query=layout_query,
                        layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
    ```
