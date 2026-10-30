# torch_npu-npu_lightning_indexer

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 inference products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Supported
<!-- end id3 -->

## Function

- Function: `lightning_indexer` obtains the Top-$k$ positions corresponding to each token through a series of operations.

- Formulas:
     $$
     Indices=\text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(Q_{index}@K_{index}^T\right)\right]\right\}
     $$
     For the Index Query $Q_{index}\in\R^{g\times d}$ corresponding to a token, given the context Index Key $K_{index}\in\R^{S_{k}\times d},W\in\R^{g\times 1}$, where $g$ is the group size corresponding to GQA, $d$ is the dimension of each head, and $S_{k}$ is the length of the context.

## Prototype

```python
torch_npu.npu_lightning_indexer(query, key, weights, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3, pre_tokens=2**63-1, next_tokens=2**63-1, return_value=False) -> (Tensor, Tensor)
```

## Parameters

> [!NOTE]   
>
> - Meanings of the dimensions of the parameters query, key, and weights: B (Batch Size) indicates the batch size of input samples, S (Sequence Length) indicates the sequence length of input samples, H (Head Size) indicates the size of the hidden layer, N (Head Num) indicates the number of heads, D (Head Dim) indicates the minimum unit size of the hidden layer and satisfies D=H/N, and T indicates the cumulative sum of the sequence lengths of input samples of all batches.
> - S1 indicates S in the query shape, S2 indicates S in the key shape, T1 indicates T in the query shape, T2 indicates T in the key shape, N1 indicates N in the query shape, and N2 indicates N in the key shape.

- **`query`** (`Tensor`): Required. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `torch.bfloat16` or `torch.float16`. When `layout_query` is `BSND`, the shape of this parameter is `[B, S1, N1, D]`. When `layout_query` is `TND`, the shape of this parameter is `[T1, N1, D]`. The value of `N1` must be less than or equal to `64`.
    
- **`key`** (`Tensor`): Required. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `torch.bfloat16` or `torch.float16`. When `layout_key` is `PA_BSND`, the shape of this parameter is `[block_count, block_size, N2, D]`, where `block_count` indicates the total number of blocks in PagedAttention, and `block_size` represents the token count per block, which must be an integer multiple of 16 and supports a maximum size of `1024`. When `layout_key` is `BSND`, the shape of this parameter is `[B, S2, N2, D]`. When `layout_key` is `TND`, the shape of this parameter is `[T2, N2, D]`. The value of `N2` must be `1`.
    
- **`weights`** (`Tensor`): Required. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `torch.bfloat16`, `torch.float16`, or `torch.float32`. The shape of this parameter can be `[B, S1, N1]` or `[T, N1]`.
    
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`actual_seq_lengths_query`** (`Tensor`): Optional. Valid token count of `query` in different batches. The data type can be `torch.int32`. If not specified, this parameter can be set to `None`, indicating that its length is identical to the size of the S dimension in the shape of `query`.
    - The number of valid tokens of each batch in this input parameter must not exceed the size of dimension S in `query` and must not be less than 0. A 1D tensor with the length B is supported. When `layout_query` is TND, this input parameter must be passed, the number of its elements is used as the value of B, and the value of each element in this input parameter indicates the total number of tokens of the current batch and all previous batches, that is, the prefix sum. Therefore, the value of the latter element must be greater than or equal to that of the former element.

- **`actual_seq_lengths_key`** (`Tensor`): Optional. Valid token count of `key` in different batches. The data type can be `torch.int32`. If this parameter is not specified or is set to `None`, its length is identical to the size of the `S` dimension in the shape of `key`.
    - The number of valid tokens of each batch in this parameter must not exceed the size of dimension S in `key` and must not be less than 0. A 1D tensor with the length B is supported. When `layout_key` is TND or PA_BSND, this input parameter must be passed. When `layout_key` is TND, the value of each element in this parameter indicates the total number of tokens of the current batch and all previous batches, that is, the prefix sum. Therefore, the value of the latter element must be greater than or equal to that of the former element.

- **`block_table`** (`Tensor`): Optional. Block mapping table used for KV storage in PagedAttention. The data layout can be ND. The data type can be `torch.int32`.
    - In the PagedAttention scenario, block\_table must be 2D, the length of the first dimension must be equal to B, and the length of the second dimension must not be less than maxBlockNumPerSeq (maxBlockNumPerSeq is the number of blocks corresponding to the maximum actual\_seq\_lengths\_key in each batch)

- **`layout_query`** (`str`): Optional. Data layout configuration of the input `query`. Valid values are `BSND` or `TND`. The default value is `BSND`.

- **`layout_key`** (`str`): Optional. Data layout configuration of the input `key`. Valid values are `PA_BSND`, `BSND`, or `TND`. The default value is `"BSND"`. In non-PagedAttention scenarios, the value of this parameter must be identical to that of `layout_query`.

- **`sparse_count`** (`int`): Optional. Block count retained in the Top-K phase. Valid values are in the range `[1, 2048]`, or can be `3072`, `4096`, `5120`, `6144`, `7168`, or `8192`. The data type can be `int32`.

- **`sparse_mode`** (`int`): Optional. Sparsification mode. The values 0 and 3 are supported. The data type can be `int32`.
    
    - When sparse\_mode is 0, it indicates the defaultMask mode.
    - When sparse\_mode is 3, it indicates the mask of the rightDownCausal mode, corresponding to the lower-triangular scenario divided by the right vertex.

- **`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. Only the default value `2^63-1` is supported.

- **`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. Only the default value `2^63-1` is supported.

- **`return_value`** (`bool`): Optional. Specifies whether to output `sparse_values`. Valid values are `True` (enables the output but does not support graph capture mode) or `False` (disables the output). The default value is `False`. This parameter is supported only during training and when `layout_key` is not set to `PA_BSND`.

## Return Values

- **`sparse_indices`** (`Tensor`): Output $Indices$ in the formula. The data type can be `torch.int32`. The data layout can be ND. When `layout_query` is `"BSND"`, the shape of this parameter is `[B, S1, N2, sparse_count]`. When `layout_query` is `"TND"`, the shape of this parameter is `[T1, N2, sparse_count]`.

- **`sparse_values`** (`Tensor`): Value data corresponding to the $Indices$ output in the formula. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. The shape of this parameter is identical to that of `sparse_indices`.

## Constraints

- This API supports graph mode.
- Only 1 is supported for N in the parameter key.
- The D value in the parameter query is equal to that in the parameter key, which is 128.
- The data types of the parameters query and key must be the same.
- When the parameter weights is not `torch.float32`, the data types of the parameters query, key, and weights must be the same.

<!-- npu="950" id4 -->
- return_value supports graph mode only on <term>Ascend 950DT products</term>.
<!-- end id4 -->
<!-- npu="950" id5 -->
- <term>Ascend 950DT products</term>:
    - Only 8, 16, 24, 32, and 64 are supported for N in query.
    - The parameter weights does not support the `torch.float32` type.
    - Only when layout_key is PA_BSND does key support a non-contiguous 0th axis.
<!-- end id5 -->
<!-- npu="A3,910b" id6 -->
- <term>Atlas A2 inference products</term> and <term>Atlas A3 inference products</term>:
    - N in query must be less than or equal to 64.
    - Non-contiguous key is not supported.
<!-- end id6 -->
- Determinism description:
    - The forward computation is deterministic by default.
    - The backward gradient is non-deterministic by default, and enabling determinism is not supported.

## Example

- Single-operator call

    ```python
    import torch
    import torch_npu
    import math
    import numpy as np
    # Generate random data and send it to the npu
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

    # Call the lightning_indexer operator
    sparse_indices, sparse_values = torch_npu.npu_lightning_indexer(
            query, key, weights, actual_seq_lengths_query=actual_seq_lengths_query, 
            actual_seq_lengths_key=actual_seq_lengths_key, block_table=block_table, layout_query=layout_query, 
            layout_key=layout_key, sparse_count=sparse_count, sparse_mode=sparse_mode)

    print(sparse_indices)
    # The output sparse_indices of the preceding code is similar to the following
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

    # Print macro that supports graph mode
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # Data generation
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
