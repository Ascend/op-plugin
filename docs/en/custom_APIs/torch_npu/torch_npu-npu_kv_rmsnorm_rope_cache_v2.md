# torch_npu.npu_kv_rmsnorm_rope_cache_v2

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- **API function**: Fuses RMSNorm normalization, RoPE (Rotary Position Embedding) position encoding, and the ScatterUpdate operation for updating KVCache in the MLA (Multi-head Latent Attention) structure.

- **Formulas**:

  - **Split the input kv Tensor**: It is split into two parts, where B is the batch size and T is the sequence length. For example, if kv is 576, 512 is split out for the RMSNorm operation and 64 is split out for the RoPE operation.

    $rms\_in \in \mathbb{R}^{B \times 1 \times T \times 512}$, $rope\_in \in \mathbb{R}^{B \times 1 \times T \times 64}$

  - **RMS normalization**: Apply RMS normalization to rms_in.

    $\displaystyle y=\gamma\ \odot\ \left(\frac{rms\_in}{\sqrt{E_d[rms\_in^2]+\epsilon}}\right)$

    - $\gamma\in\mathbb{R}^{512}$ is a learnable scaling parameter.
    - $E_d[·]$ indicates the mean along the last dimension (dimension d=512).
    - $\epsilon$ is a small constant (such as 0.00001) used to prevent division by zero.
    - $\odot$ indicates element-wise multiplication.

  - **Rotary position embedding (RoPE)**

    - Reshape and transpose: Reshape and transpose rope_in to prepare for rotation.

      $k=reshape(rope\_in, [B, 1, T, 32, 2]) \rightarrow transpose(-1, -2) \rightarrow reshape([B, 1, T, 64])$

    - Rotation operation: Apply rotary position embedding.
    
      $k_{embed}=k \odot cos + RotateHalf(k) \odot sin$

    - cos and sin are precomputed rotation angle parameters.
    - RotateHalf(k) moves the second half of the elements of k to the first half and negates them, and uses the values of the first half for the second half. Specifically, for dimension d=64:

      $RotateHalf(k)_i=-k_{i+32},\ if\ i < 32\ else\ k_{i-32}$

## Prototype

```python
torch_npu.npu_kv_rmsnorm_rope_cache_v2(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, epsilon=1e-5, cache_mode='Norm', is_output_kv=False, k_cache_dtype=None, ckv_cache_dtype=None) -> (Tensor, Tensor)
```

## Parameters

- **`kv`** (`Tensor`): Required. Input feature Tensor. The data type can be `torch.bfloat16` or `torch.float16`. The data layout is BNSD. It must be a 4D Tensor with shape [batch_size, 1, seq_len, hidden_size], where hidden_size=rms_size (RMS)+rope_size (RoPE).
- **`gamma`** (`Tensor`): Required. Scaling parameter for RMS normalization. The data type can be `torch.bfloat16` or `torch.float16`. The data layout is ND. It must be a 1D Tensor with shape [rms_size].
- **`cos`** (`Tensor`): Required. Cosine component of the RoPE rotary position embedding. The data type can be `torch.bfloat16` or `torch.float16`. The data layout is ND. It must be a 4D Tensor with shape [batch_size, 1, seq_len, rope_size].
- **`sin`** (`Tensor`): Required. Sine component of the RoPE rotary position embedding. The data type can be `torch.bfloat16` or `torch.float16`. The data layout is ND. It must be a 4D Tensor with shape [batch_size, 1, seq_len, rope_size].
- **`index`** (`Tensor`): Required. Cache index Tensor used to locate the write positions of `k_cache` and `ckv_cache`. The data type can be `torch.int64`. The data layout is ND. The shape depends on `cache_mode`.
- **`k_cache`** (`Tensor`): Required. Used to store quantized/non-quantized k vectors. The data type can be `torch.bfloat16`, `torch.float16`, `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The data layout is ND. The shape depends on `cache_mode`. The input and output reuse the same address.
- **`ckv_cache`** (`Tensor`): Required. Used to store quantized/non-quantized compressed kv vectors. The data type can be `torch.bfloat16`, `torch.float16`, `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The data layout is ND. The shape depends on `cache_mode`. The input and output reuse the same address.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.

- **`k_rope_scale`** (`Tensor`): Optional. The default value is None. Quantization scaling factor of the k rotary position embedding. The data type can be `torch.float32`. The data layout is ND. It must be a 1D Tensor with shape [rope_size]. It is required in quantization mode.
- **`c_kv_scale`** (`Tensor`): Optional. The default value is None. Quantization scaling factor of the compressed kv. The data type can be `torch.float32`. The data layout is ND. It must be a 1D Tensor with shape [rms_size]. It is required in quantization mode.
- **`k_rope_offset`** (`Tensor`): Optional. The default value is None. Quantization offset of the k rotary position embedding. The data type can be `torch.float32`. The data layout is ND. It must be a 1D Tensor with shape [rope_size]. Symmetric and asymmetric quantization modes are supported. In asymmetric quantization, this parameter is required.

- **`c_kv_offset`** (`Tensor`): Optional. The default value is None. Quantization offset of the compressed kv. The data type can be `torch.float32`. The data layout is ND. It must be a 1D Tensor with shape [rms_size]. Symmetric and asymmetric quantization modes are supported. In asymmetric quantization, this parameter is required.

- **`epsilon`** (`float`): Optional. The default value is 1e-5. It indicates the minimum value in RMS normalization to prevent division by zero.
- **`cache_mode`** (`str`): Optional. The default value is 'Norm'. It indicates the cache mode. The supported modes are as follows.

    | Enum value | Mode name | Description |
    | --- | --- | --- |
    | Norm | KV-Cache update mode | The shape of k_cache is [batch_size, 1, cache_length, rope_size], and the shape of ckv_cache is [batch_size, 1, cache_length, rms_size]. The shape of index is [batch_size, seq_len], and the value in index indicates the offset in each batch. |
    | PA/PA_BNSD | PagedAttention mode | The shape of k_cache is [block_num, block_size, 1, rope_size], and the shape of ckv_cache is [block_num, block_size, 1, rms_size]. The shape of index is [batch_size*seq_len], and the value in index indicates the offset of each token. |
    | PA_NZ | PagedAttention mode with cache data in FRACTAL_NZ format | The shape of k_cache is [block_num, block_size, 1, rope_size], and the shape of ckv_cache is [block_num, block_size, 1, rms_size]. The shape of index is [batch_size * seq_len], and the value in index indicates the offset of each token.<br>The data layout differs in different quantization modes:<br><li>In non-quantization mode: the data layout of k_cache is [block_num, rope_size//16, block_size, 1, 16], and the data layout of ckv_cache is [block_num, rms_size//16, block_size, 1, 16]</li><li>In quantization mode: the data layout of k_cache is [block_num, rope_size//32, block_size, 1, 32], and the data layout of ckv_cache is [block_num, rms_size//32, block_size, 1, 32]</li> |
    | PA_BLK_BNSD | Special PagedAttention mode | The shape of k_cache is [block_num, block_size, 1, rope_size], and the shape of ckv_cache is [block_num, block_size, 1, rms_size]. The shape of index is [batch_size*Ceil(seq_len/block_size)], and the value in index indicates the starting offset of each block and no longer corresponds to tokens one by one. |
    | PA_BLK_NZ | Special PagedAttention mode with cache data in FRACTAL_NZ format | The shape of k_cache is [block_num, block_size, 1, rope_size], and the shape of ckv_cache is [block_num, block_size, 1, rms_size]. The shape of index is [batch_size * Ceil(seq_len / block_size)], and the value in index indicates the starting offset of each block and no longer corresponds to tokens one by one.<br>The data layout differs in different quantization modes:<br><li>In non-quantization mode: the data layout of k_cache is [block_num, rope_size//16, block_size, 1, 16], and the data layout of ckv_cache is [block_num, rms_size//16, block_size, 1, 16]</li><li>In quantization mode: the data layout of k_cache is [block_num, rope_size//32, block_size, 1, 32], and the data layout of ckv_cache is [block_num, rms_size//32, block_size, 1, 32]</li> |

- **`is_output_kv`** (`bool`): Optional. Whether to output the processed `k_embed_out` and `y_out` (the original values before quantization). The default value is False, indicating no output. It takes effect only when `cache_mode` is (PA/PA_BNSD/PA_NZ/PA_BLK_BNSD/PA_BLK_NZ).
- **`k_cache_dtype`** (`int`): Optional. Data type of the output `k_cache`.
- **`ckv_cache_dtype`** (`int`): Optional. Data type of the output `ckv_cache`.

## Return Values

- **`k_cache`** (`Tensor`): Reuses the same address as the input `k_cache`. The data type, dimensions, and data layout are the same (essentially an in-place update).
- **`ckv_cache`** (`Tensor`): Reuses the same address as the input `ckv_cache`. The data type, dimensions, and data layout are the same (essentially an in-place update).
- **`k_embed_out`** (`Tensor`): Output only when `is_output_kv`=True. It indicates the value after RoPE processing. It must be a 4D Tensor with shape [batch_size, 1, seq_len, rope_size]. The data type and layout are the same as those of the input kv.
- **`y_out`** (`Tensor`): Output only when `is_output_kv`=True. It indicates the value after RMSNorm processing. It must be a 4D Tensor with shape [batch_size, 1, seq_len, rms_size]. The data type and layout are the same as those of the input kv.

## Constraints

- This API supports single-operator mode and TorchAir graph mode.
- Variables used in Tensor shapes:
  - batch_size: Batch size.
  - seq_len: Sequence length.
  - hidden_size: Length of the MLA input vector.
  - rms_size: Length of the RMSNorm branch vector.
  - rope_size: Length of the RoPE branch vector.
  - cache_length: Valid in Norm mode. Maximum length supported by KVCache.
  - block_num: Valid in PagedAttention mode. Number of blocks.
  - block_size: Valid in PagedAttention mode. Block size.
  - hidden_size, rms_size, and rope_size are determined by actual service scenarios and are set by users as required. rope_size must be even and must satisfy rms_size+rope_size=hidden_size.

- Quantization mode: When `k_rope_scale` and `c_kv_scale` are not empty, the dtype of `k_cache` and `ckv_cache` is `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The last dimension of the cache shape must be 32 (Cache data format is FRACTAL_NZ mode). k_rope_scale and c_kv_scale must both be non-empty, and k_rope_offset and c_kv_offset must both be None or both be non-empty.
- Non-quantization mode: When `k_rope_scale` and `c_kv_scale` are empty, the dtype of `k_cache` and `ckv_cache` is `torch.bfloat16` or `torch.float16`.
- Index mapping: In all `cache_mode` cache modes, the index values cannot be repeated. If the passed index values contain duplicates, the operator behavior is undefined and unpredictable.
  - Norm: The value in index indicates the offset in each batch.
  - PA/PA_BNSD/PA_NZ: The value in index indicates the global offset.
  - PA_BLK_BNSD/PA_BLK_NZ: The value in index indicates the global offset of each page. This scenario assumes that cache updates are continuous, and non-continuous cache updates are not supported.

- Shape association rules: Different `cache_mode` cache modes have different shape rules.
  - Norm: The shape of k_cache is [batch_size, 1, cache_length, rope_size], the shape of ckv_cache is [batch_size, 1, cache_length, rms_size], the shape of index is [batch_size, seq_len], and cache_length>=seq_len.
  - Non-Norm mode (PagedAttention-related modes): block_num>=Ceil(seq_len/block_size)\*batch_size is required.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    batch_size=8
    seq_len=1
    page_num=8
    page_size=128
    input_dtype = torch.float16

    kv = torch.randn(batch_size, 1, seq_len, 576, dtype = input_dtype).npu()
    gamma = torch.randn(512, dtype = input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()

    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = input_dtype).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    is_output_kv = True


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache_v2(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                                                k_rope_scale=k_rope_scale,
                                                                c_kv_scale=c_kv_scale,
                                                                k_rope_offset=k_rope_offset,
                                                                c_kv_offset=c_kv_offset,
                                                                epsilon=epsilon,
                                                                cache_mode=cache_mode,
                                                                is_output_kv=is_output_kv)
            return k_rope, c_kv

    model = Model().npu()
    k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    npu_backend = tng.get_npu_backend(compiler_config=config)

    batch_size=8
    seq_len=1
    page_num=8
    page_size=128
    input_dtype = torch.float16

    kv = torch.randn(batch_size, 1, seq_len, 576, dtype = input_dtype).npu()
    gamma = torch.randn(512, dtype = input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype = input_dtype).npu()

    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = input_dtype).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    is_output_kv = True


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache_v2(kv, gamma, cos, sin, index, k_cache, ckv_cache,
                                                                  k_rope_scale=k_rope_scale,
                                                                  c_kv_scale=c_kv_scale,
                                                                  k_rope_offset=k_rope_offset,
                                                                  c_kv_offset=c_kv_offset,
                                                                  epsilon=epsilon,
                                                                  cache_mode=cache_mode,
                                                                  is_output_kv=is_output_kv)
            return k_rope, c_kv

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)
    ```
