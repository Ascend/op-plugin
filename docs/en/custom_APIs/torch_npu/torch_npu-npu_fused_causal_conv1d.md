# torch\_npu.npu\_fused\_causal\_conv1d

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product | Supported |
| --- | --- |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |

## Function

- **Description:**

    Performs causal 1D convolution on a sequence. Cached data (with a length equal to the convolution kernel width minus 1) is used to pad the beginning of each sequence along the sequence dimension, ensuring that the output depends on the current and historical inputs. After the convolution, the current sequence data is updated in the cache. The original input is added to the causal 1D convolution output to implement a residual connection. This API supports features such as automatic prefix caching (APC), MTP-based speculative decoding, and residual connections.

- **Formulas:**

  `K` is the convolution kernel width (fixed at 3), `L` is the original sequence length, and `dim` is the feature dimension.
  - Cache reading

      Cache line index:

      $$
      readCacheLine = \begin{cases}
      cacheIndices[batchId, \; initialStateIdx[batchId]], & \text{APC mode} \\
      cacheIndices[batchId], & \text{non-APC mode and cacheIndices exists} \\
      batchId, & \text{otherwise}
      \end{cases}
      $$

      Case 1: First computation (`numComputedTokens[batchId] == 0`)

      $$
      cachedState[i, dim] = 0, \quad 0 \leq i < K-1
      $$

      $$
      offset = 0
      $$

      Case 2: Speculative decoding mode (`numAcceptedTokens` exists)

      $$
      offset = numAcceptedTokens[batchId] - 1
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
      $$

      Case 3: Default mode

      $$
      offset = C - (K - 1)
      $$

      $$
      cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
      $$

  - Cache concatenation

      $$
      paddedInput[i, dim] =
      \begin{cases}
      cachedState[i, dim], & 0 \leq i < offset + K - 1 \\
      x[i - (offset + K - 1), dim], & offset + K - 1 \leq i < offset + K - 1 + L
      \end{cases}
      $$

  - Cache update

      $$
      Len = offset + K - 1 + L
      $$

      $$
      M = \min(C, \; Len)
      $$

      $$
      writeCacheLine = \begin{cases}
      cacheIndices[batchId, \; idxLast], & \text{APC mode} \\
      cacheIndices[batchId], & \text{non-APC mode and cacheIndices exists} \\
      batchId, & \text{otherwise}
      \end{cases}
      $$

      $$
      convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
      $$

  - Offset cropping

      $$
      x'[i, dim] = paddedInput[i + offset, dim], \quad 0 \leq i < K - 1 + L
      $$

  - APC cache filling (optional, in APC mode)

      $$
      seqCompletedOffsetToken = numComputedTokens[batchId] \mod B
      $$

      $$
      seqCompletedOffset = B - seqCompletedOffsetToken
      $$

      $$
      seqEndOffset = (L - seqCompletedOffset) \mod B
      $$

      $$
      lastFullBlockTokenIndex = \begin{cases}
      L - seqEndOffset - B, & seqEndOffset = 0 \\
      L - seqEndOffset, & \text{otherwise}
      \end{cases}
      $$

      $$
      nBlockToFill = idxLast - idxFirst
      $$

      For each `chunk = 0`, `1`, ..., `nBlockToFill - 1`:

      $$
      boundaryIdx = lastFullBlockTokenIndex - (nBlockToFill - chunk - 1) \times B
      $$

      $$
      convStates[cacheIndices[batchId, \; idxFirst + chunk]][C-(K-1)+j, \; dim] = x'[boundaryIdx + j, \; dim], \quad j = 0, \dots, K-2
      $$

  - Causal 1D convolution

      $$
      y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim], \quad i = 0, 1, \dots, L-1
      $$

  - Zero-padding reset (optional, when `convMode == 1` and `numComputedTokens` is not `None`)

      $$
      resetIdx = \min\!\Big(\max\!\big(K - 1 - numComputedTokens[batchId], \; 0\big), \; L\Big)
      $$

      $$
      y[i, dim] = 0, \quad 0 \leq i < resetIdx
      $$

  - Residual connection (optional)

      $$
      y[i, dim] = x[i, dim] + y[i, dim]
      $$

## Prototype

```python
torch_npu.npu_fused_causal_conv1d(x, weight, conv_states, *, query_start_loc=None, cache_indices=None,initial_state_mode=None, bias=None, num_accepted_tokens=None,activation="None", pad_slot_id=-1, run_mode=0, residual_connection=0, max_query_len=-1,num_computed_tokens=None, block_idx_first_scheduled_token=None,block_idx_last_scheduled_token=None, initial_state_idx=None, block_size=128, conv_mode="default") -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input sequence, $x$ in the formulas. The data type can be `float16` or `bfloat16`. The data layout must be `ND`. Non-contiguous tensors are supported. Empty tensors are not supported.
- **`weight`** (`Tensor`): Required. Causal 1D convolution kernel, $weight$ in the formulas. The data type and data layout are the same as those of `x`. Non-contiguous tensors are not supported. Empty tensors are not supported.
- **`conv_states`** (`Tensor`): Required. Cache state tensor for storing historical token data of each sequence, updated in place after computation for each sequence, $conv\_states$ in the formulas. The data type and data layout are the same as those of `x`. Non-contiguous tensors are supported. Empty tensors are not supported.
- **`*`**: Position delimiter. Variables before this delimiter are position-dependent and must be passed in order. Variables after this delimiter are optional keyword arguments and must be assigned using key-value pairs. If not specified, their default values are used.
- **`query_start_loc`** (`Tensor`): Optional. Sequence start position indices recording the starting positions of sequences in concatenated tensor `x`. The data type can be `int32`. The data layout must be `ND`. Non-contiguous tensors are not supported. This parameter cannot be omitted when `x` is a 2D tensor. The default value is `None`.
- **`cache_indices`** (`Tensor`): Optional. Cache indices specifying the index of the cache state for each sequence in `conv_states`. The data type can be `int32`. The data layout must be `ND`. Non-contiguous tensors are not supported. The default value is `None`.
- **`initial_state_mode`** (`Tensor`): Optional. Legacy parameter with no effect in the current API. The data type can be `int32`. The data layout must be `ND`. Non-contiguous tensors are not supported. The default value is `None`.
- **`bias`** (`Tensor`): Optional. Legacy parameter with no effect in the current API. The data type and data layout are the same as those of `x`. Non-contiguous tensors are not supported. The default value is `None`.
- **`num_accepted_tokens`** (`Tensor`): Optional. Number of accepted speculative tokens for each batch. The data type can be `int32`. The data layout must be `ND`. Non-contiguous tensors are not supported. The default value is `None`.
- **`activation`** (`str`): Optional. Activation function type. This parameter has no effect in the current API. The default value is `"None"`.
- **`pad_slot_id`** (`int`): Optional. Used to skip batches that do not need to participate in computation. The default value is `-1`.
- **`run_mode`** (`int`): Optional. This parameter has no effect in the current API. The default value is `0`.
- **`residual_connection`** (`int`): Optional. Determines whether to apply residual connection to the output. `0`: No residual connection. `1`: Output is the sum of the convolution result and input `x` (residual connection). The default value is `0`.
- **`max_query_len`** (`int`): Optional. Maximum `seq_len` across all batches. The default value is `-1`.
- **`num_computed_tokens`** (`Tensor`): Optional. Total number of tokens already processed in the current batch, used to determine the initial state. The data type can be `int32`. The data layout must be `ND`. Non-contiguous tensors are not supported. This parameter cannot be omitted when `conv_mode` is `"pangu"` or APC is enabled. The default value is `None`.
- **`block_idx_first_scheduled_token`** (`Tensor`): Optional. Block index corresponding to the first scheduled token of the current batch. The data type can be `int32`. The data layout must be `ND`. Non-contiguous tensors are not supported. This parameter cannot be omitted when APC is enabled. The default value is `None`.
- **`block_idx_last_scheduled_token`** (`Tensor`): Optional. Block index corresponding to the last scheduled token of the current batch. The data type can be `int32`. The data layout must be `ND`. Non-contiguous tensors are not supported. This parameter cannot be omitted when APC is enabled. The default value is `None`.
- **`initial_state_idx`** (`Tensor`): Optional. Index of the initial state block. The data type can be `int32`. The data layout must be `ND`. Non-contiguous tensors are not supported. This parameter cannot be omitted when APC is enabled. The default value is `None`.
- **`block_size`** (`int`): Optional. Block size. The default value is `128`.
- **`conv_mode`** (`str`): Optional. Supports two implementations: Qwen3-Next (`"default"`) and Pangu V2 (`"pangu"`). The default value is `"default"`.

## Return Values

**`y`** (`Tensor`): Computation result, $y$ in the formulas. The data type and data layout are the same as those of `x`. Non-contiguous tensors are not supported.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator and graph mode calls.

- Supported scenarios:
  - Prefill scenario:
    - `x`: `[cu_seq_len, dim]`
    - `weight`: `[K, dim]`, where `K = 3`
    - `conv_states`: `[-1, K-1, dim]`
    - `query_start_loc`: `[batch+1]`
    - `cache_indices`: `[batch]` or `None` when APC is disabled; `[batch, maxNumBlocks]` when APC is enabled
    - `initial_state_mode`: `[batch]` (no effect)
    - `bias`: `[dim]` (no effect)
    - `num_accepted_tokens`: `[batch]` (no effect)
    - `num_computed_tokens`: `[batch]`
    - `block_idx_first_scheduled_token`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `block_idx_last_scheduled_token`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `initial_state_idx`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `activation`: (no effect)
    - `pad_slot_id`: Default value `-1`
    - `run_mode`: (no effect)
    - `max_query_len`: Greater than `8`
    - `residual_connection`: `0` for no residual connection; `1` for residual connection
    - `block_size`: Typical value `128` or `256`
    - `conv_mode`: Qwen3-Next mode: `"default"`; Pangu V2 mode: `"pangu"`
    - `y`: `[cu_seq_len, dim]`

    Here, `cu_seq_len` is the total length obtained by concatenating all variable-length sequences in the batch.

    Input shape constraints:
    - `x` must be a 2D tensor with shape `[cu_seq_len, dim]`.
    - `weight` must be a 2D tensor with shape `[K, dim]`, where `K` is fixed at `3`.
    - `conv_states` must be a 3D tensor with shape `[..., K-1, dim]`. The size of the first dimension is not fixed and must be greater than or equal to `batch` and the total number of elements of `cache_indices`.
    - `cache_indices` must be a 1D tensor with shape `[batch]` or a 2D tensor with shape `[batch, maxNumBlocks]`. The 1D form indicates that APC is disabled, and the 2D form indicates that APC is enabled.
    - `cu_seq_len` must be in the range `[batch, 1024*1024]`. `dim` must be in the range `[64, 16384]` and must be a multiple of `16`. The product of `cu_seq_len` and `dim` must be in the range `[64*batch, 4G]`. `batch` must be in the range `[1, 256]`.
    - `maxNumBlocks` must be greater than or equal to `ceil(max_query_len, block_size)`.

  - Mixed prefill and decode scenario:
    - `x`: `[cu_seq_len, dim]`
    - `weight`: `[K, dim]`, where `K = 3`
    - `conv_states`: `[-1, K-1+m, dim]`
    - `query_start_loc`: `[batch+1]`
    - `cache_indices`: `[batch]` or `None` when APC is disabled; `[batch, maxNumBlocks]` when APC is enabled
    - `initial_state_mode`: `[batch]` (no effect)
    - `bias`: `[dim]` (no effect)
    - `num_accepted_tokens`: `[batch]`
    - `num_computed_tokens`: `[batch]`
    - `block_idx_first_scheduled_token`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `block_idx_last_scheduled_token`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `initial_state_idx`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `activation`: (no effect)
    - `pad_slot_id`: Default value `-1`
    - `run_mode`: (no effect)
    - `max_query_len`: Greater than `8`
    - `residual_connection`: `0` for no residual connection; `1` for residual connection
    - `block_size`: Typical value `128` or `256`
    - `conv_mode`: Qwen3-Next mode: `"default"`; Pangu V2 mode: `"pangu"`
    - `y`: `[cu_seq_len, dim]`

    Here, `cu_seq_len` is the total length obtained by concatenating all variable-length sequences in the batch.

    Input shape constraints:
    - `x` must be a 2D tensor with shape `[cu_seq_len, dim]`.
    - `weight` must be a 2D tensor with shape `[K, dim]`, where `K` is fixed at `3`.
    - `conv_states` must be a 3D tensor with shape `[..., K-1+m, dim]`. The size of the first dimension is not fixed and must be greater than or equal to `batch` and the total number of elements of `cache_indices`.
    - `cache_indices` must be a 1D tensor with shape `[batch]` or a 2D tensor with shape `[batch, maxNumBlocks]`. The 1D form indicates that APC is disabled, and the 2D form indicates that APC is enabled.
    - `cu_seq_len` must be in the range `[batch, 1024*1024]`. `dim` must be in the range `[64, 16384]` and must be a multiple of `16`. The product of `cu_seq_len` and `dim` must be in the range `[64*batch, 4G]`. `batch` must be in the range `[1, 256]`.
    - `maxNumBlocks` must be greater than or equal to `ceil(max_query_len, block_size)`.

  - Decode scenario (variable-length sequences):
    - `x`: `[cu_seq_len, dim]`
    - `weight`: `[K, dim]`, where `K = 3`
    - `conv_states`: `[-1, K-1+m, dim]`
    - `query_start_loc`: `[batch+1]`
    - `cache_indices`: `[batch]` or `None` when APC is disabled; `[batch, maxNumBlocks]` when APC is enabled
    - `initial_state_mode`: `[batch]` (no effect)
    - `bias`: `[dim]` (no effect)
    - `num_accepted_tokens`: `[batch]`
    - `num_computed_tokens`: `[batch]`
    - `block_idx_first_scheduled_token`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `block_idx_last_scheduled_token`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `initial_state_idx`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `activation`: (no effect)
    - `pad_slot_id`: Default value `-1`
    - `run_mode`: (no effect)
    - `max_query_len`: Default value `1`
    - `residual_connection`: `0` for no residual connection; `1` for residual connection
    - `block_size`: Typical value `128` or `256`
    - `conv_mode`: Qwen3-Next mode: `"default"`; Pangu V2 mode: `"pangu"`
    - `y`: `[cu_seq_len, dim]`

    Here, `state_len` must be greater than the maximum number of tokens among all batches plus `1`.

    Input shape constraints:
    - `x` must be a 2D tensor with shape `[cu_seq_len, dim]`.
    - `weight` must be a 2D tensor with shape `[K, dim]`, where `K` is fixed at `3`.
    - `conv_states` must be a 3D tensor with shape `[..., K-1+m, dim]`. The size of the first dimension is not fixed and must be greater than or equal to `batch` and the total number of elements of `cache_indices`.
    - `cache_indices` must be a 1D tensor with shape `[batch]` or a 2D tensor with shape `[batch, maxNumBlocks]`. The 1D form indicates that APC is disabled, and the 2D form indicates that APC is enabled.
    - `cu_seq_len` must be in the range `[batch, batch*8]`, and the number of tokens in each batch must be in the range `[1, 8]`. `dim` must be in the range `[64, 16384]` and must be a multiple of `16`. `batch` must be in the range `[1, 256]`.
    - `maxNumBlocks` must be greater than or equal to `ceil(max_query_len, block_size)`.

  - Decode scenario (fixed batch size):
    - `x`: `[batch, m+1, dim]`
    - `weight`: `[K, dim]`, where `K = 3`
    - `conv_states`: `[-1, K-1+m, dim]`
    - `query_start_loc`: `[batch+1]`
    - `cache_indices`: `[batch]` or `None` when APC is disabled; `[batch, maxNumBlocks]` when APC is enabled
    - `initial_state_mode`: `[batch]` (no effect)
    - `bias`: `[dim]` (no effect)
    - `num_accepted_tokens`: `[batch]`
    - `num_computed_tokens`: `[batch]`
    - `block_idx_first_scheduled_token`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `block_idx_last_scheduled_token`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `initial_state_idx`: `None` when APC is disabled; `[batch]` when APC is enabled
    - `activation`: (no effect)
    - `pad_slot_id`: Default value `-1`
    - `run_mode`: (no effect)
    - `max_query_len`: Default value `1`
    - `residual_connection`: `0` for no residual connection; `1` for residual connection
    - `block_size`: Typical value `128` or `256`
    - `conv_mode`: Qwen3-Next mode: `"default"`; Pangu V2 mode: `"pangu"`
    - `y`: `[batch, m+1, dim]`

    Input shape constraints:
    - `x` must be a 3D tensor with shape `[batch, m+1, dim]`.
    - `weight` must be a 2D tensor with shape `[K, dim]`, where `K` is fixed at `3`.
    - `conv_states` must be a 3D tensor with shape `[..., K-1+m, dim]`. The size of the first dimension is not fixed and must be greater than or equal to `batch` and the total number of elements of `cache_indices`.
    - `cache_indices` must be a 1D tensor with shape `[batch]` or a 2D tensor with shape `[batch, maxNumBlocks]`. The 1D form indicates that APC is disabled, and the 2D form indicates that APC is enabled.
    - `m` must be in the range `[0, 7]`. `dim` must be in the range `[64, 16384]` and must be a multiple of `16`. `batch` must be in the range `[1, 256]`.
    - `maxNumBlocks` must be greater than or equal to `ceil(max_query_len, block_size)`.

- Input value range constraints:
  - `query_start_loc` contains cumulative offsets, with values in the range `[0, cu_seq_len]` and length `batch+1`. `query_start_loc[i]` represents the starting offset of the `i`-th sequence, and `query_start_loc[batch+1]` represents the ending position of the last sequence.
  - `block_size` must be greater than or equal to `2`.
  - When APC is enabled, `block_idx_first_scheduled_token`, `block_idx_last_scheduled_token`, `initial_state_idx`, and `num_computed_tokens` must be provided and satisfy the following requirements, where `i` is the batch index:
    - `initial_state_idx[i]` must be less than or equal to `block_idx_first_scheduled_token[i] + 1`.
    - `initial_state_idx[i]` must be less than or equal to `block_idx_last_scheduled_token[i]`.
    - `block_idx_first_scheduled_token[i]` must be less than or equal to `block_idx_last_scheduled_token[i]`.
    - `block_idx_last_scheduled_token[i]` must be less than `maxNumBlocks`.
  - `num_accepted_tokens` can be `None` or non-`None`. When non-`None`, its length must be `batch`, and each element must be greater than `0` and no greater than the number of tokens in the current batch minus `1`.
  - Values of `cache_indices` must be in the range `[0, conv_states.dim[0]-1]`, and all elements must be distinct.
  - In Pangu V2 mode (`conv_mode = "pangu"`), `num_computed_tokens` cannot be `None`.
  - The operator inputs and intermediate computation results must remain within the value range of the corresponding data type (`float16` or `bfloat16`).
  - Operator inputs must not contain `±inf` or `nan`.

## Examples

- Single-operator call
  - Prefill scenario:

    ```python
    import torch
    import torch_npu

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [15, 12, 20, 10]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 15, 27, 47, 57], dtype=torch.int32).npu()

    num_slots = 8
    conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([10, 5, 0, 0], dtype=torch.int32).npu()

    max_query_len = max(seq_lens)
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    out = torch_npu.npu_fused_causal_conv1d(
      x,
      weight,
      conv_states,
      query_start_loc=query_start_loc,
      cache_indices=cache_indices,
      initial_state_mode=None,
      bias=None,
      num_accepted_tokens=None,
      num_computed_tokens=num_computed_tokens,
      block_idx_first_scheduled_token=None,
      block_idx_last_scheduled_token=None,
      initial_state_idx=None,
      activation="None",
      pad_slot_id=-1,
      run_mode=0,
      max_query_len=max_query_len,
      residual_connection=residual_connection,
      block_size=block_size,
      conv_mode=conv_mode,
    )
    print(f"output shape: {out.shape}")
    ```

  - Mixed prefill and decode scenario:

    ```python
    import torch
    import torch_npu

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [12, 9, 2, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 12, 21, 23, 24], dtype=torch.int32).npu()

    num_slots = 8
    m = 2
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([0, 20, 50, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([1, 1, 2, 1], dtype=torch.int32).npu()

    max_query_len = max(seq_lens)
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=None,
        bias=None,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        activation="None",
        pad_slot_id=-1,
        run_mode=0,
        max_query_len=max_query_len,
        residual_connection=residual_connection,
        block_size=block_size,
        conv_mode=conv_mode,
    )
    print(f"output shape: {out.shape}")
    ```

  - Decode scenario (variable-length sequences):

    ```python
    import torch
    import torch_npu

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [1, 1, 1, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32).npu()

    num_slots = 8
    m = 1
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([100, 50, 80, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([1, 1, 1, 1], dtype=torch.int32).npu()

    max_query_len = 1
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=None,
        bias=None,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        activation="None",
        pad_slot_id=-1,
        run_mode=0,
        max_query_len=max_query_len,
        residual_connection=residual_connection,
        block_size=block_size,
        conv_mode=conv_mode,
    )
    print(f"output shape: {out.shape}")
    ```

  - Decode scenario (fixed batch size):

    ```python
    import torch
    import torch_npu

    K = 3
    dim = 128
    batch = 4
    m = 2
    seq_len = m + 1
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    x = torch.randn(batch, seq_len, dim, dtype=dtype).npu()

    num_slots = 8
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    query_start_loc = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([100, 50, 80, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([2, 1, 3, 2], dtype=torch.int32).npu()

    max_query_len = seq_len
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    out = torch_npu.npu_fused_causal_conv1d(
        x,
        weight,
        conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        initial_state_mode=None,
        bias=None,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        activation="None",
        pad_slot_id=-1,
        run_mode=0,
        max_query_len=max_query_len,
        residual_connection=residual_connection,
        block_size=block_size,
        conv_mode=conv_mode,
    )
    print(f"output shape: {out.shape}")
    ```

- Graph mode call
  - Prefill scenario:

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [15, 12, 20, 10]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 15, 27, 47, 57], dtype=torch.int32).npu()

    num_slots = 8
    conv_states = torch.randn(num_slots, K - 1, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([10, 5, 0, 0], dtype=torch.int32).npu()

    max_query_len = max(seq_lens)
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def causal_conv1d_prefill(x, weight, conv_states, query_start_loc,
                              cache_indices, num_computed_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=None,
            bias=None,
            num_accepted_tokens=None,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="None",
            pad_slot_id=-1,
            run_mode=0,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
        )

    compiled_func = torch.compile(causal_conv1d_prefill, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc,
                        cache_indices, num_computed_tokens)
    print(f"output shape: {out.shape}")
    ```

  - Mixed prefill and decode scenario:

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [12, 9, 2, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 12, 21, 23, 24], dtype=torch.int32).npu()

    num_slots = 8
    m = 2
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 3, 1, 5], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([0, 20, 50, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([1, 1, 2, 1], dtype=torch.int32).npu()

    max_query_len = max(seq_lens)
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def causal_conv1d_mixed(x, weight, conv_states, query_start_loc,
                            cache_indices, num_computed_tokens, num_accepted_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=None,
            bias=None,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="None",
            pad_slot_id=-1,
            run_mode=0,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
        )

    compiled_func = torch.compile(causal_conv1d_mixed, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc,
                        cache_indices, num_computed_tokens, num_accepted_tokens)
    print(f"output shape: {out.shape}")
    ```

  - Decode scenario (variable-length sequences):

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    K = 3
    dim = 128
    batch = 4
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    seq_lens = [1, 1, 1, 1]
    cu_seq_len = sum(seq_lens)
    x = torch.randn(cu_seq_len, dim, dtype=dtype).npu()
    query_start_loc = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32).npu()

    num_slots = 8
    m = 1
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([100, 50, 80, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([1, 1, 1, 1], dtype=torch.int32).npu()

    max_query_len = 1
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def causal_conv1d_decode_2d(x, weight, conv_states, query_start_loc,
                                cache_indices, num_computed_tokens, num_accepted_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=None,
            bias=None,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="None",
            pad_slot_id=-1,
            run_mode=0,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
        )

    compiled_func = torch.compile(causal_conv1d_decode_2d, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc,
                        cache_indices, num_computed_tokens, num_accepted_tokens)
    print(f"output shape: {out.shape}")
    ```

  - Decode scenario (fixed batch size):

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    K = 3
    dim = 128
    batch = 4
    m = 2
    seq_len = m + 1
    dtype = torch.bfloat16

    weight = torch.randn(K, dim, dtype=dtype).npu()
    x = torch.randn(batch, seq_len, dim, dtype=dtype).npu()

    num_slots = 8
    state_len = K - 1 + m
    conv_states = torch.randn(num_slots, state_len, dim, dtype=dtype).npu()
    cache_indices = torch.tensor([0, 2, 4, 6], dtype=torch.int32).npu()
    query_start_loc = torch.tensor([0, 3, 6, 9, 12], dtype=torch.int32).npu()
    num_computed_tokens = torch.tensor([100, 50, 80, 30], dtype=torch.int32).npu()
    num_accepted_tokens = torch.tensor([2, 1, 3, 2], dtype=torch.int32).npu()

    max_query_len = seq_len
    block_size = 128
    conv_mode = "default"
    residual_connection = 0

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def causal_conv1d_decode_3d(x, weight, conv_states, query_start_loc,
                                cache_indices, num_computed_tokens, num_accepted_tokens):
        return torch_npu.npu_fused_causal_conv1d(
            x,
            weight,
            conv_states,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=None,
            bias=None,
            num_accepted_tokens=num_accepted_tokens,
            num_computed_tokens=num_computed_tokens,
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            activation="None",
            pad_slot_id=-1,
            run_mode=0,
            max_query_len=max_query_len,
            residual_connection=residual_connection,
            block_size=block_size,
            conv_mode=conv_mode,
        )

    compiled_func = torch.compile(causal_conv1d_decode_3d, backend=npu_backend)
    out = compiled_func(x, weight, conv_states, query_start_loc,
                        cache_indices, num_computed_tokens, num_accepted_tokens)
    print(f"output shape: {out.shape}")
    ```
