# torch_npu.npu_scatter_pa_cache

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: Updates the key at the specified position in KCache.

- **Formulas**:

  Depending on the optional parameter combination, there are three scenarios:

  - Scenario 1: `compress_lens`, `seq_lens`, and `compress_seq_offset` are not used.

    $$\mathit{keyCache} = \mathit{slotMapping}(\mathit{key})$$

  - Scenario 2:

    $$
    \begin{aligned}
    \mathit{keyCache} &= \mathit{slotMapping}(\mathit{key}[: \mathit{compressSeqOffset}], \\
    &\quad \mathit{ReduceMean}(\mathit{key}[\mathit{compressSeqOffset}: \mathit{compressSeqOffset} + \mathit{compressLens}]), \\
    &\quad \text{key}[\text{compressSeqOffset} + \mathit{compressLens}: \mathit{seqLens}])
    \end{aligned}
    $$

  - Scenario 3: `compress_seq_offset` is not used.

    $$\mathit{keyCache} = \mathit{slotMapping}(\mathit{key}[\mathit{seqLens} - \mathit{compressLens}: \mathit{seqLens}])$$

## Prototype

```python
torch_npu.npu_scatter_pa_cache(key, slot_mapping, *, compress_lens = None, compress_seq_offset = None, seq_lens = None, key_cache) -> ()
```

## Parameters

- **`key`** (`Tensor`): Required. Key values to be updated. The data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`, `torch.int8`, `torch.uint8`, `torch.int16`, `torch.uint16`, `torch.int32`, `torch.uint32`, `torch_npu.hifloat8`, `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch_npu.float4_e2m1fn_x2`, or `torch_npu.float4_e1m2fn_x2`. The Tensor can be 3D or 4D. The data types `torch_npu.hifloat8`, `torch.float8_e5m2`, `torch.float8_e4m3fn`, `torch_npu.float4_e2m1fn_x2`, and `torch_npu.float4_e1m2fn_x2` are supported only when the Tensor is 3D. The shape is `(batch * seq_len, num_head, k_head_size)` or `(batch, seq_len, num_head, k_head_size)`. For `torch_npu.float4_e2m1fn_x2` and `torch_npu.float4_e1m2fn_x2`, `k_head_size` must be even. The data layout can be ND.
- **`slot_mapping`** (`Tensor`): Required. Storage offset of each token of `key` in the cache. The data type can be `torch.int32` or `torch.int64`. The Tensor can be 1D or 2D. When `key` is 3D, the shape of `slot_mapping` is `(batch * seq_len)`. When `key` is 4D, the shape of `slot_mapping` is `(batch, num_head)`. The value range is `[0, num_blocks * block_size - 1]`, and element values must not be repeated. Correctness is not guaranteed when values are repeated. The data layout can be ND.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`compress_lens`** (`Tensor`): Optional. Compression length. The data type is the same as that of `slot_mapping`. The Tensor can be 1D or 2D. When `key` is 4D and `compress_seq_offset` is not None, the shape is `(batch, num_head)`. When `key` is 4D and `compress_seq_offset` is None, the shape is `(batch * num_head)`. The data layout can be ND.
- **`compress_seq_offset`** (`Tensor`): Optional. Compression start point of each head in each batch. The data type is the same as that of `slot_mapping`. The Tensor is 1D with shape `(batch * num_head)`. The data layout can be ND.
- **`seq_lens`** (`Tensor`): Optional. Actual seqLens of each batch. The data type is the same as that of `slot_mapping`. The Tensor is 1D with shape `(batch)`. The data layout can be ND.
- **`key_cache`** (`Tensor`): Required. Output Tensor. The `key_cache` to be updated. The data type is the same as that of `key`. When `key` is 3D, the shape is `(num_blocks, block_size, num_head, k_head_size)`. When `key` is 4D, the shape is `(num_blocks, block_size, 1, k_head_size)`. The data layout can be ND.

## Return Values

No return value. The input `key_cache` is modified in place.

## Constraints

- This API supports training and inference scenarios.
- This API supports only single-operator mode.
- Non-contiguous Tensors are not supported. Empty Tensors are supported.
- Each element in `seq_lens` and `compress_lens` must satisfy: reduceSum(seq_lens[i] - compress_lens[i] + 1) <= num_blocks * block_size (corresponding to scenarios 2 and 3).
- Variables used in the shape description:
  - `batch`: Number of sequences in the current input (number of samples processed at a time). The value is a positive integer.
  - `seq_len`: Sequence length. The value is a positive integer.
  - `num_head`: Number of heads in multi-head attention. The value is a positive integer.
  - `k_head_size`: Feature dimension of the key in each attention head (length of a single-head key). The value is a positive integer.
  - `num_blocks`: Total number of pre-allocated blocks in keyCache, used to store key data of all sequences. The value is a positive integer.
  - `block_size`: Number of tokens contained in each cache block. The value is a positive integer.

## Examples

Single-operator call

- Scenario 1

    ```python
    import torch
    import torch_npu

    a1 = torch.randint(-1, 1, (256, 16, 16), dtype=torch.float32).npu() # key
    a2 = torch.randint(-1, 1, (16, 16, 16, 16), dtype=torch.float32).npu() # key_cache
    a3 = torch.arange(0, 256).view(256).to(torch.int32).npu() # slot_mapping
    torch_npu.npu_scatter_pa_cache(a1, a3, key_cache=a2)
    ```

- Scenario 2

    ```python
    import torch
    import torch_npu

    a1 = torch.randint(-1, 1, (16, 16, 16, 16), dtype=torch.float32).npu() # key
    a2 = torch.randint(-1, 1, (16, 256, 1, 16), dtype=torch.float32).npu() # key_cache
    a3 = torch.arange(0, 256).view(16, 16).to(torch.int32).npu() # slot_mapping
    a4 = torch.randint(0, 8, (16, 16), dtype=torch.int32).npu() # compress_lens
    a5 = torch.randint(0, 8, (256,), dtype=torch.int32).npu() # compress_seq_offsets
    a6 = torch.full((16,), 16).to(torch.int32).npu() # seq_len
    torch_npu.npu_scatter_pa_cache(a1, a3, compress_lens=a4, compress_seq_offsets=a5, seq_lens=a6, key_cache=a2)
    ```

- Scenario 3

    ```python
    import torch
    import torch_npu

    a1 = torch.randint(-1, 1, (16, 16, 16, 16), dtype=torch.float32).npu() # key
    a2 = torch.randint(-1, 1, (16, 256, 1, 16), dtype=torch.float32).npu() # key_cache
    a3 = torch.arange(0, 256).view(16, 16).to(torch.int32).npu() # slot_mapping
    a4 = torch.randint(0, 8, (256,), dtype=torch.int32).npu() # compress_lens
    a6 = torch.full((16,), 16).to(torch.int32).npu() # seq_len
    torch_npu.npu_scatter_pa_cache(a1, a3, compress_lens=a4, seq_lens=a6, key_cache=a2)
    ```
