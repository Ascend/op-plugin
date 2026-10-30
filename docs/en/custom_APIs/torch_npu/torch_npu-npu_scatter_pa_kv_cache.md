# torch_npu.npu_scatter_pa_kv_cache

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

- **API function**: Updates the key and value at the specified position in the KvCache.

- **Supported input and output scenarios**:

    - Scenario 1:

        ```python
        key: [batch *seq_len, num_head, k_head_size]
        value: [batch *seq_len, num_head, v_head_size]
        key_cache: [num_blocks, num_head * k_head_size // last_dim_k, block_size, last_dim_k]
        value_cache: [num_blocks, num_head * v_head_size // last_dim_k, block_size, last_dim_k]
        slot_mapping: [batch *seq_len]
        ```   

        `k_head_size` and `v_head_size` can be different or the same.

    - Scenario 2:

        ```python    
        key: [batch, seq_len, num_head, k_head_size]
        value: [batch, seq_len, num_head, v_head_size]
        key_cache: [num_blocks, block_size, 1, k_head_size]
        value_cache: [num_blocks, block_size, 1, k_head_size]
        slot_mapping: [batch, num_head]
        compress_lens: [batch, num_head]
        seq_lens: [batch]
        compress_seq_offsets: [batch * num_head]
        ```    

    The preceding scenarios are distinguished by the constructed parameters. If the first parameter construction is used, scenario 1 applies. If the second construction is used, scenario 2 applies. Scenario 1 does not have the three optional parameters `compress_lens`, `seq_lens`, and `compress_seq_offsets`.

## Prototype

```python
torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping, *, compress_lens = None, compress_seq_offsets = None, seq_lens = None, cache_mode = 'PA_NZ') -> ()
```

## Parameters

- **`key`** (`Tensor`): Required. Key values to be updated, that is, the `key` of multiple tokens in the current step. 3D or 4D is supported. The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`, `torch.int8`, `torch.uint8`, `torch.int16`, `torch.uint16`, `torch.int32`, `torch.uint32`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The data layout can be ND.
- **`value`** (`Tensor`): Required. Value to be updated, that is, the `value` of multiple tokens in the current step. 3D or 4D is supported. The data type and data layout must be the same as those of `key`.
- **`key_cache`** (`Tensor`): Required. `key_cache` to be updated, that is, the `key_cache` of the current layer. Only 4D is supported. The data type and data layout must be the same as those of `key`.
- **`value_cache`** (`Tensor`): Required. `value_cache` to be updated, that is, the `value_cache` of the current layer. Only 4D is supported. The data type and data layout must be the same as those of `key`.
- **`slot_mapping`** (`Tensor`): Required. Storage offset of each token key or value in the cache. The data type can be `torch.int32` or `torch.int64`. The data layout can be ND. The value range is [0, num_blocks * block_size-1], and element values must not be repeated. Correctness is not guaranteed if values are repeated.
- `*`: Variables before it are position-dependent and must be passed in order; variables after it are optional parameters and are order-independent. They must be assigned using key-value pairs, and default values are used if they are not assigned.
- **`compress_lens`** (`Tensor`): Optional. Compression amount. The data type is the same as that of `slot_mapping`. The data layout can be ND. The default value is None.
- **`compress_seq_offsets`** (`Tensor`): Optional. Compression start position of each head in each batch. The data type is the same as that of `slot_mapping`. The data layout can be ND. The default value is None.
- **`seq_lens`** (`Tensor`): Optional. Actual seqLens of each batch. The data type is the same as that of `slot_mapping`. The data layout can be ND. The default value is None.
- **`cache_mode`** (`str`): Optional. Memory layout format of `key_cache` and `value_cache`. When None or 'Norm' is passed, only the ND memory layout is supported. When 'PA_NZ' is passed, only the NZ memory layout is supported. The default value is 'PA_NZ'.

## Return Values

No return value. `key_cache` and `value_cache` are updated in place.

## Constraints

- This API supports inference and training scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- Non-contiguous input parameters are not supported;
- The data types of `key`, `value`, `key_cache`, and `value_cache` must be the same;
- The data types of `slot_mapping`, `compress_lens`, `compress_seq_offsets`, and `seq_lens` must be the same;
- The value range of `slot_mapping` is [0, num_blocks * block_size-1], and element values in `slot_mapping` must not be repeated. Correctness is not guaranteed if values are repeated;
- When both `key` and `value` are 3D, the first two dimensions of their `shape` must be the same;
- When both `key` and `value` are 4D, the first three dimensions of their `shape` must be the same, and the third dimension of `key_cache` and `value_cache` must be 1;
- When `key` and `value` are 4D, `compress_lens` and `seq_lens` are required. When `key` and `value` are 3D, `compress_lens`, `compress_seq_offsets`, and `seq_lens` are optional;
- When both `key` and `value` are 4D, `slot_mapping` is 2D, and the value of the first dimension of `slot_mapping` is equal to the first dimension of `key`, which is `batch`. The value of the second dimension of `slot_mapping` is equal to the third dimension of `key`, which is num_head (corresponding to scenario 2);
- When both `key` and `value` are 4D, `seq_lens` is 1D, and its value is equal to the first dimension of `key`, which is batch (corresponding to scenario 2);
- Each element value in `seq_lens` and `compress_lens` must satisfy the formula: reduceSum(seq_lens[i] - compress_lens[i]) <= num_blocks * block_size (corresponding to scenario 2);

## Examples

Single-operator call

- Scenario 1

    ```python
    import numpy as np
    import torch
    import torch_npu
    
    bs = 16
    num_head = 4
    k_head_size = 32
    v_head_size = 64
    num_blocks = 2
    lastDim_k = 16
    block_size = 32
    
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
    
    torch_npu.npu_scatter_pa_kv_cache(key_npu, value_npu, key_cache_npu_cast, value_cache_npu_cast, slot_mapping_npu)
    print(key_cache_npu_cast)
    tensor([[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
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

- Scenario 2

    ```python
    import torch
    import torch_npu
    a1 = torch.rand(2, 2, 3, 7).to(torch.float32).npu()     # key
    a2 = torch.rand(2, 2, 1, 7).to(torch.float32).npu()    # key_cache
    a3 = torch.tensor([[0, 3, 6], [9, 12, 15]]).to(torch.int32).npu()               # slot_mapping
    a4 = torch.rand(2, 2, 3, 7).to(torch.float32).npu()     # value
    a5 = torch.rand(2, 2, 1, 7).to(torch.float32).npu()    # value_cache
    a6 = torch.tensor([[1, 0, 0],[0, 1, 0]]).to(torch.int32).npu()                # compress_lens
    a7 = torch.tensor([0, 0, 1, 0, 1, 1]).to(torch.int32).npu()                  # compress_seq_offsets
    a8 = torch.tensor([2,1]).to(torch.int32).npu()                   # seq_lens
    res = torch_npu.npu_scatter_pa_kv_cache(a1, a4, a2, a5, a3, compress_lens=a6,compress_seq_offsets=a7, seq_lens=a8, cache_mode="Norm")
    ```

Graph mode call

- Scenario 1

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

    # Print macros supported in graph mode
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.aoe_config.aoe_mode = "2"
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # Data generation
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
        model = torch.compile(model, backend=npu_backend, dynamic = False, fullgraph = True)
        model(key_npu, value_npu, slot_mapping_npu, key_cache_npu_cast, value_cache_npu_cast)

        # The output here is the same as that of the single-operator mode
    ```

- Scenario 2

    ```python
    import os
    import torch
    import torch_npu
    import torchair

    config = torchair.CompilerConfig()
    # Set Aten operator information in the exported graph
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = torchair.get_npu_backend(compiler_config = config)

    def generate_input_data():
      a1 = torch.rand(2, 2, 3, 7).to(torch.float32)     # key
      a2 = torch.rand(2, 2, 1, 7).to(torch.float32)    # key_cache
      a3 = torch.tensor([[0, 3, 6], [9, 12, 15]]).to(torch.int32)               # slot_mapping
      a4 = torch.rand(2, 2, 3, 7).to(torch.float32)     # value
      a5 = torch.rand(2, 2, 1, 7).to(torch.float32)    # value_cache
      a6 = torch.tensor([[1, 0, 0],[0, 1, 0]]).to(torch.int32)                # compress_lens
      a7 = torch.tensor([0, 0, 1, 0, 1, 1]).to(torch.int32)                   # compress_seq_offsets
      a8 = torch.tensor([2,1]).to(torch.int32)                            # seq_lens

      input_data = [a1, a2, a3, a4, a5, a6, a7, a8]
      return input_data

    class Model(torch.nn.Module):
      def __init__(self):
        super().__init__()
      def forward(self, key, key_cache, slot_mapping, value, value_cache, compress_lens = None, compress_seq_offsets = None, seq_lens = None):
        return torch_npu.npu_scatter_pa_kv_cache(key, value, key_cache, value_cache, slot_mapping, compress_lens = compress_lens, compress_seq_offsets = compress_seq_offsets, seq_lens = seq_lens, cache_mode = "Norm")

    input_data = generate_input_data()
    a1_npu = input_data[0].npu()
    a2_npu = input_data[1].npu()
    a3_npu = input_data[2].npu()
    a4_npu = input_data[3].npu()
    a5_npu = input_data[4].npu()
    a6_npu = input_data[5].npu()
    a7_npu = input_data[6].npu()
    a8_npu = input_data[7].npu()

    model = Model()
    model=model.to('npu')

    model = torch.compile(model, backend=npu_backend, dynamic=True)
    res = model(a1_npu, a2_npu, a3_npu, a4_npu, a5_npu, a6_npu, a7_npu, a8_npu)
    print('************** test npu_scatter_pa_kv_cache dtype is: ', a1_npu.dtype, a3_npu.dtype)

    print("key_cache==========",a2_npu)
    print("value_cache=====:",a5_npu)
    print("success")
    ```
