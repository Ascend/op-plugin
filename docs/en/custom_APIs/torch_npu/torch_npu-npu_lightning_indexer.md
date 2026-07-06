# torch_npu-npu_lightning_indexer

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 inference products</term>        | √  |
|<term>Atlas A3 inference products</term>        | √  |

## Function

- Description: Obtains the Top-$k$ positions for each token through a series of operations.

- Formula:
     $$
     Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
     $$
     For an index query $Q_{index} \in \mathbb{R}^{g \times d}$ corresponding to a token, given the context index key $K_{index} \in \mathbb{R}^{S_{k} \times d}$ and $W \in \mathbb{R}^{g \times 1}$. $g$ indicates the group size in Grouped-Query Attention (GQA). $d$ indicates the dimension of each head. $S_k$ indicates the context length.

## Prototype

```python
torch_npu.npu_lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3, pre_tokens=2^63-1, next_tokens=2^63-1, return_value=False) -> (Tensor, Tensor)
```

## Parameters

> [!NOTE]   
>
> - Dimension definitions for the `query`, `key`, and `weights` parameters:<br>`B` (`Batch Size`) indicates the input sample batch size.<br>`S` (`Sequence Length`) indicates the input sample sequence length.<br>`H` (`Head Size`) indicates the hidden layer size.<br>`N` (`Head Num`) indicates the number of heads.<br>`D` (`Head Dim`) indicates the minimum unit size of the hidden layer, satisfying `D = H/N`.<br>`T` indicates the cumulative sum of the sequence lengths of all batch input samples.
> - `S1` indicates the `S` dimension in the shape of `query`.<br>`S2` indicates the `S` dimension in the shape of `key`.<br>`T1` indicates the `T` dimension in the shape of `query`.<br>`T2` indicates the `T` dimension in the shape of `key`.<br>`N1` indicates the `N` dimension in the shape of `query`.<br>`N2` indicates the `N` dimension in the shape of `key`.

- **`query`** (`Tensor`): Required. Non-contiguous tensors are not supported. The data layout is ND. The data type can be `bfloat16` or `float16`. When `layout_query` is `BSND`, the shape of this parameter is `[B, S1, N1, D]`. When `layout_query` is `TND`, the shape of this parameter is `[T1, N1, D]`. The value of `N1` must be less than or equal to `64`.
    
- **`key`** (`Tensor`): Required. Non-contiguous tensors are not supported. The data layout is ND. The data type can be `bfloat16` or `float16`. When `layout_key` is `PA_BSND`, the shape of this parameter is `[block_count, block_size, N2, D]`, where `block_count` indicates the total number of blocks in PageAttention, and `block_size` represents the token count per block, which must be an integer multiple of 16 and supports a maximum size of `1024`. When `layout_key` is `BSND`, the shape of this parameter is `[B, S2, N2, D]`. When `layout_key` is `TND`, the shape of this parameter is `[T2, N2, D]`. The value of `N2` must be `1`.
    
- **`weights`** (`Tensor`): Required. Non-contiguous tensors are not supported. The data layout is ND. The data type can be `bfloat16`, `float16`, or `float32`. The shape of this parameter can be `[B, S1, N1]` or `[T, N1]`.
    
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).

- **`actual_seq_lengths_query`** (`Tensor`): Optional. Valid token count of `query` in different batches. The data type can be `int32`. If not specified, this parameter can be set to `None` to indicate that its length is identical to the size of the S dimension in the shape of `query`.
    - The valid token count for each batch must not exceed the size of the S dimension in `query` and must be greater than or equal to 0. This parameter must be a 1D tensor of length `B`. When `layout_query` is `TND`, this parameter must be provided, where its element count determines the batch size `B`, and each element indicates the cumulative token count of the current batch and all preceding batches, representing a prefix sum. Therefore, the value of each element must be greater than or equal to that of the preceding element.

- **`actual_seq_lengths_key`** (`Tensor`): Optional. Valid token count of `key` in different batches. The data type can be `int32`. If this parameter is not specified or is set to `None`, its length is identical to the size of the `S` dimension in the shape of `key`.
    - The valid token count for each batch must not exceed the size of the `S` dimension in `key` and must be greater than or equal to 0. This parameter must be a 1D tensor of length `B`. When `layout_key` is set to `TND` or `PA_BSND`, this parameter must be provided. When `layout_key` is set to `TND`, each element indicates the prefix sum of token counts across batches, and the value of each element must be greater than or equal to that of the preceding element.

- **`block_table`** (`Tensor`): Optional. Block mapping table used for KV storage in PageAttention. The data layout is ND. The data type can be `int32`.
    - In page attention scenarios, `block_table` must be a 2D tensor. Its first dimension must equal `B`, and its second dimension must be no less than `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actual_seq_lengths_key` across batches).

- **`layout_query`** (`str`): Optional. Data layout configuration of the input `query`. Valid values are `BSND` or `TND`. The default value is `BSND`.

- **`layout_key`** (`str`): Optional. Data layout configuration of the input `key`. Valid values are `PA_BSND`, `BSND`, or `TND`. The default value is `"BSND"`. In non-PageAttention scenarios, the value of this parameter must be identical to that of `layout_query`.

- **`sparse_count`** (`int`): Optional. Block count retained in the Top-K phase. Valid values are in the range `[1, 2048]`, or can be `3072`, `4096`, `5120`, `6144`, `7168`, or `8192`. The data type can be `int32`.

- **`sparse_mode`** (`int`): Optional. Sparsification mode. The data type can be `int32`. Valid values:
    
    - `0`: enables `defaultMask` mode.
    - `3`: enables rightDownCausal` mode mask, corresponding to lower triangular scenarios where the dividing line extends from the right vertex.

- **`pre_tokens`** (`int`): Optional. Used for sparse computation, indicating the number of preceding tokens with which the attention is associated. The data type can be `int64`. Only the default value `2^63-1` is supported.

- **`next_tokens`** (`int`): Optional. Used for sparse computation, indicating the number of subsequent tokens with which the attention is associated. The data type can be `int64`. Only the default value `2^63-1` is supported.

- **`return_value`** (`bool`): Optional. Specifies whether to output `sparse_values`. `True` outputs `sparse_values`, and `False` disables it. The default value is `False`. This parameter is supported only during training and when `layout_key` is not set to `PA_BSND`.

## Return Values

- **`sparse_indices`** (`Tensor`): Output $Indices$ in the formula. The data type can be `int32`. The data layout is ND. When `layout_query` is `"BSND"`, the shape of this parameter is `[B, S1, N2, sparse_count]`. When `layout_query` is `"TND"`, the shape of this parameter is `[T1, N2, sparse_count]`.

- **`sparse_values`** (`Tensor`): Value data corresponding to the $Indices$ output in the formula. The data type can be `bfloat16` or `float16`. The data layout is ND. The shape of this parameter is identical to that of `sparse_indices`.

## Constraints

- This API supports graph mode.
- In the `query` parameter, the value of `N` must be less than or equal to `64`. In the `key` parameter, only the value `1` is supported for `N`.
- The size of the `D` dimension in both `query` and `key` must be identical and equal to `128`.
- The data types of `query` and `key` must be identical.
- When the data type of `weights` is not `float32`, the data types of `query`, `key`, and `weights` must be identical.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import math
    import numpy as np
    # Generate random data and send it to the NPU.
    b = 1
    s1 = 1
    s2 = 8192
    n1 = 64
    n2 = 1
    d = 128
    block_size = 256
    
    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, d))).to(torch.bfloat16).npu()
    key = torch.tensor(np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d))).to(torch.bfloat16).npu()
    weights = torch.tensor(np.random.uniform(-1, 1, (b, s1, n1))).to(torch.bfloat16).npu()
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32).npu()
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32).npu()
    block_table = torch.tensor([range(b*s2//block_size)], dtype=torch.int32).reshape(b, -1).npu()
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3

    # Call the lightning_indexer operator.
    sparse_indices, sparse_values = torch_npu.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
            layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)

    print(sparse_indices)
    # Expected sparse_indices output of the preceding code sample:
    tensor([[[[4488, 3926, 1154, ..., 3535, 8031, 8180]]]],
            device='npu:0', dtype=torch.int32)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import numpy as np
    import math
    import torchair as tng

    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"

    # Configure logging and debug settings for graph capture
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # Generate data
    b = 1
    s1 = 1
    s2 = 8192
    n1 = 64
    n2 = 1
    d = 128
    block_size = 256
    
    query = torch.tensor(np.random.uniform(-10, 10, (b, s1, n1, d))).to(torch.bfloat16).npu()
    key = torch.tensor(np.random.uniform(-10, 10, (b*(s2//block_size), block_size, n2, d))).to(torch.bfloat16).npu()
    weights = torch.tensor(np.random.uniform(-1, 1, (b, s1, n1))).to(torch.bfloat16).npu()
    actual_seq_lengths_query = torch.tensor(np.random.uniform(s1, s1, (b))).to(torch.int32).npu()
    actual_seq_lengths_key = torch.tensor(np.random.uniform(s2, s2, (b))).to(torch.int32).npu()
    block_table = torch.tensor([range(b*s2//block_size)], dtype=torch.int32).reshape(b, -1).npu()
    layout_query = 'BSND'
    layout_key = 'PA_BSND'
    sparse_count = 2048
    sparse_mode = 3

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_lightning_indexer(query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
                actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
                layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_lightning_indexer(query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
            layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)
        print("single op output:", single_op[0], single_op[0].shape)
        print("graph output:", graph_output[0], graph_output[0].shape)
    if __name__ == "__main__":
        MetaInfershape()

    # Expected output of the preceding code sample:
    single op output: tensor([[[[4488, 3926, 1154, ..., 3535, 8031, 8180]]]],
            device='npu:0', dtype=torch.int32) torch.Size([1, 1, 1, 2048])

    graph output: tensor([[[[4488, 3926, 1154, ..., 3535, 8031, 8180]]]],
            device='npu:0', dtype=torch.int32) torch.Size([1, 1, 1, 2048])
    ```
