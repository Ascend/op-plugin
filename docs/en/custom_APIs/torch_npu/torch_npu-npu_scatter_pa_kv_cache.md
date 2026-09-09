# torch_npu.npu_scatter_pa_kv_cache

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |

## Function

Updates the `key` and `value` at the specified positions in the KV cache.

The input and output support the following scenarios:

- Scenario 1:

    ```python
    key:[batch, num_head, k_head_size]
    value:[batch, num_head, v_head_size]
    key_cache:[num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]
    value_cache:[num_blocks, num_head * v_head_size // last_dim_k, block_size, last_dim_k]
    slot_mapping:[batch]
    ```    

- Scenario 2:

    ```python    
    key:[batch, seq_len, num_head, k_head_size]
    value:[batch, seq_len, num_head, v_head_size]
    key_cache:[num_blocks, block_size, 1, k_head_size]
    value_cache:[num_blocks, block_size, 1, k_head_size]
    slot_mapping:[batch, num_head]
    compress_lens:[batch, num_head]
    seq_lens:[batch]
    compress_seq_offsets:[batch * num_head]
    ```    

The scenario is determined by the constructed parameters. Execution flows to Scenario 1 when the input parameters match the first construction, and to Scenario 2 when they match the second construction. Scenario 1 does not include the three optional parameters `compress_lens`, `seq_lens`, and `compress_seq_offsets`.

## Prototype

```python
torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping, *, compress_lens=None, compress_seq_offsets=None, seq_lens=None, cache_mode='PA_NZ') -> ()
```

## Parameters

- **`key`** (`Tensor`): Required. Key values to be updated, which are the keys of multiple tokens in the current step. The shape must have 3 or 4 dimensions. The data type can be `float16`, `float`, `bfloat16`, `int8`, `uint8`, `int16`, `uint16`, `int32`, `uint32`, `hifloat8`, `float8_e5m2`, or `float8_e4m3fn`. The data layout can be ND.
- **`value`** (`Tensor`): Required. Value values to be updated, which are the values of multiple tokens in the current step. The shape must have 3 or 4 dimensions. The data type and data layout must be identical to those of `key`.
- **`key_cache`** (`Tensor`): Required. Key cache to be updated, which is the key cache of the current layer. The shape must have 4 dimensions. The data type and data layout must be identical to those of `key`.
- **`value_cache`** (`Tensor`): Required. Value cache to be updated, which is the value cache of the current layer. The shape must have 4 dimensions. The data type and data layout must be identical to those of `key`.
- **`slot_mapping`** (`Tensor`): Required. Storage offsets of each token key or value in the cache. The data type can be `int32` or `int64`. The data layout can be ND.
- **`compress_lens`** (`Tensor`): Optional. Compression lengths. The data type must be identical to that of `slot_mapping`. The data layout can be ND. The default value is `None`.
- **`compress_seq_offsets`** (`Tensor`): Optional. Compression start points for each head in each batch. The data type must be identical to that of `slot_mapping`. The data layout can be ND. The default value is `None`.
- **`seq_lens`** (`Tensor`): Optional. Actual sequence lengths of each batch. The data type must be identical to that of `slot_mapping`. The data layout can be ND. The default value is `None`.
- **`cache_mode`** (`str`): Memory layout of `key_cache` and `value_cache`. When `None` or `'Norm'` is provided, only the ND memory layout is supported. When `'PA_NZ'` is provided, only the NZ memory layout is supported. The default value is `'PA_NZ'`.

## Return Values

None. `key_cache` and `value_cache` are updated in place.

## Constraints

- Non-contiguous input parameters are not supported.
- The data types of `key`, `value`, `key_cache`, and `value_cache` must be identical.
- The data types of `slot_mapping`, `compress_lens`, `compress_seq_offsets`, and `seq_lens` must be identical.
- The value range of `slot_mapping` must be [0, `num_blocks * block_size - 1`]. The element values within `slot_mapping` must be unique. Accuracy is not guaranteed if duplicate values exist. 
- When key and value are both 3D, their first two shape dimensions must be identical.
- When `key` and `value` are both 4D, their first three shape dimensions must be identical, and the third dimension of `key_cache` and `value_cache` must be `1`.
- When `key` and `value` are 4D, `compress_lens` and `seq_lens` are required. When `key` and `value` are 3D, `compress_lens`, `compress_seq_offsets`, and `seq_lens` are optional.
- When `key` and `value` are both 4D, `slot_mapping` is 2D. The first dimension of `slot_mapping` must be identical to the first dimension of `key` (`batch`), and the second dimension of `slot_mapping` must be identical to the third dimension of `key` (`num_head`) (corresponding to Scenario 2).
- When `key` and `value` are both 4D, `seq_lens` is 1D. The value of `seq_lens` must be identical to the first dimension of `key` (`batch`) (corresponding to Scenario 2).
- Each element value within `seq_lens` and `compress_lens` must satisfy the formula: `reduceSum(seq_lens[i] - compress_lens[i]) <= num_blocks * block_size` (corresponding to Scenario 2).

## Examples

- Single-operator call

    ```python
    >>> import numpy as np
    >>> import torch
    >>> import torch_npu
    >>>
    >>> bs = 16
    >>> num_head = 4
    >>> k_head_size = 32
    >>> v_head_size = 64
    >>> num_blocks = 2
    >>> lastDim_k = 16
    >>> block_size = 32
    >>>
    >>> key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
    >>> value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
    >>> key_cache = np.random.randn(
    >>>     num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
    >>> value_cache = np.zeros(
    >>>     (num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
    >>> slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)
    >>>
    >>> key_npu = torch.from_numpy(key).npu()
    >>> value_npu = torch.from_numpy(value).npu()
    >>> key_cache_npu = torch.from_numpy(key_cache).npu()
    >>> value_cache_npu = torch.from_numpy(value_cache).npu()
    >>> key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
    >>> value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
    >>> slot_mapping_npu = torch.from_numpy(slot_mapping).npu()
    >>>
    >>> torch_npu.npu_scatter_pa_kv_cache(key_npu, value_npu, key_cache_npu_cast, value_cache_npu_cast, slot_mapping_npu)
    >>> print(key_cache_npu_cast)
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ...,
        [[ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]],
        ...,
        [[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120],
        ...,
        [[ 0.1113,  0.6255,  0.7686,  ...,  0.0247,  0.2490,  0.6909],
        [ 0.4312,  0.7954,  0.7339,  ...,  0.1154,  0.6440,  0.3342],
        [ 0.9570,  0.2869,  0.6489,  ...,  0.7451,  0.0234,  0.8843]],
        ...,
        [[ 1.8271,  1.4551,  1.3154,  ...,  1.9854,  1.4365,  1.0732],
        [ 1.9492,  1.6455,  1.6504,  ...,  1.5957,  1.6201,  1.4385],
        [ 0.0742,  0.1982,  0.8945,  ...,  0.4912,  0.6753,  0.1120]]]]
        device='npu:0', dtype=torch.float16)
    ```

- Graph mode call

    ```python
    import numpy as np
    import torch
    import torch_npu

    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"

    # Configure logging and debug settings for graph capture
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.aoe_config.aoe_mode = "2"
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # Generate data
    bs = 16
    num_head = 4
    k_head_size = 32
    v_head_size = 64
    num_blocks = 2
    lastDim_k = 16
    block_size = 32

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, key, value, slot_mapping, key_cache, value_cache):
            torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping)

    if __name__ == "__main__":
        torch_npu.npu.set_device(0)

        key = np.random.randn(bs, num_head, k_head_size).astype(np.float16)
        value = np.random.randn(bs, num_head, v_head_size).astype(np.float16)
        key_cache = np.random.randn(
            num_blocks, num_head * k_head_size // lastDim_k, block_size, lastDim_k).astype(np.float16)
        value_cache = np.zeros(
            (num_blocks, num_head * v_head_size // lastDim_k, block_size, lastDim_k)).astype(np.float16)
        slot_mapping = np.random.choice(num_blocks * block_size, bs, replace=False).astype(np.int32)

        key_npu = torch.from_numpy(key).npu()
        value_npu = torch.from_numpy(value).npu()
        key_cache_npu = torch.from_numpy(key_cache).npu()
        value_cache_npu = torch.from_numpy(value_cache).npu()
        key_cache_npu_cast = torch_npu.npu_format_cast(key_cache_npu.contiguous(), 29)
        value_cache_npu_cast = torch_npu.npu_format_cast(value_cache_npu.contiguous(), 29)
        slot_mapping_npu = torch.from_numpy(slot_mapping).npu()

        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)

        model = Model().npu()
        # Graph mode call
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
        model(key_npu, value_npu, slot_mapping_npu, key_cache_npu_cast, value_cache_npu_cast)

        # The output here is the same as that of the single operator call.
    ```
