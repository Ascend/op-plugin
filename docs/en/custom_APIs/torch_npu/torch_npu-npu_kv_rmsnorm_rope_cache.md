# torch\_npu.npu\_kv\_rmsnorm\_rope\_cache<a name="ZH-CN_TOPIC_0000002343094197"></a>

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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

## Function<a name="en-us_topic_0000002236535552_section1023311522369"></a>

- Fuses Root Mean Square Normalization (RMSNorm), Rotary Position Embedding (RoPE), and KV cache update operations (ScatterUpdate) within the Multi-Head Latent Attention (MLA) structure.
    - **MLA (Multi-head Latent Attention)**:
      An efficient attention mechanism proposed in DeepSeek-V2. It compresses Key/Value states into low-rank latent representations to reduce KV cache footprint and computational overhead while maintaining long-sequence modeling capabilities.

    - **RoPE (Rotary Position Embedding)**:
      A positional encoding method that injects positional information by applying a rotational transformation to vectors. It can model relative positional relationships and is suitable for long-context scenarios, with vector rotation implemented via the `RotateHalf` operation.

    - **RMSNorm (Root Mean Square Normalization)**:
      A normalization method that scales the input using the root mean square. It is computationally simple with low overhead and is commonly used in large language model training.

- Formulas:
    - **Input tensor `kv` splitting**: The input tensor `kv` is split into two parts, where `B` indicates the batch size and `T` indicates the sequence length. For example, when `kv` is `576`, `512` is split out for RMSNorm and `64` is split out for RoPE.

        $rms\_in \in \mathbb{R}^{B \times 1 \times T \times 512}$,  $rope\_in \in \mathbb{R}^{B \times 1 \times T \times 64}$

    - **Input tensor `kv` splitting**: The input tensor `kv` is split into two parts, where `B` indicates the batch size and `T` indicates the sequence length.

        $\displaystyle y=\gamma\ \odot\ \left(\frac{rms\_in}{\sqrt{E_d[rms\_in^2]+\epsilon}}\right)$

        - $γ∈R^{512}$ is a learnable scaling parameter.
        - $Ed[·]$ indicates the mean along the last dimension where the size of the D dimension is `512`.
        - $ε$ is a small constant (such as `0.00001`) used to prevent division by zero.
        - $⊙$ indicates element-wise multiplication.

    - **RoPE**

        - Reshaping and transposition: Reshape and transpose `rope_in` to prepare for rotation.

          $k=reshape(rope\_in, [B, 1, T, 32, 2]) \rightarrow transpose(-1, -2) \rightarrow reshape([B, 1, T, 64])$

        - Rotation operation: Apply rotary position embedding.

          $k_{embed}=k \odot cos + RotateHalf(k) \odot sin$

          - `cos` and `sin` are precomputed rotation parameters.
          - `RotateHalf(k)` moves the elements in the second half of `k` to the first half and negates them, and fills the second half with the values of the first half. Specifically, when the size of the `D` dimension is `64`:

            $RotateHalf(k)_i=-k_{i+32},\ if\ i < 32\ else\ k_{i-32}$

## Prototype<a name="en-us_topic_0000002236535552_section123412524369"></a>

```python
torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, epsilon=1e-5, cache_mode='Norm', is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000002236535552_section1723416525369"></a>

- **`kv`** (`Tensor`): Required. Input feature tensor. The data type can be `torch.bfloat16` or `torch.float16`. The data layout is `BNSD`. This parameter must be 4D with shape `[batch_size, 1, seq_len, hidden_size]`, where `hidden_size` = `rms_size` (RMS) + `rope_size` (RoPE).
- **`gamma`** (`Tensor`): Required. Scaling parameter for RMSNorm. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. This parameter must be 1D with shape `[rms_size]`.
- **`cos`** (`Tensor`): Required. Cosine component of RoPE. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. This parameter must be 4D with shape `[batch_size, 1, seq_len, rope_size]`.
- **`sin`** (`Tensor`): Required. Sine component of RoPE. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. This parameter must be 4D with shape `[batch_size, 1, seq_len, rope_size]`.
- **`index`** (`Tensor`): Required. Cache index tensor used to locate the write positions in `k_cache` and `ckv_cache`. The data type can be `torch.int64`. The data layout can be ND. The shape depends on `cache_mode`.
- **`k_cache`** (`Tensor`): Required. Storage tensor for quantized or non-quantized key vectors. The data type can be `torch.bfloat16`, `torch.float16`, `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The data layout can be ND. The shape depends on `cache_mode`.

  <!-- npu="A3,910b" id4 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: the data types `torch_npu.hifloat8`, `torch.float8_e5m2`, and `torch.float8_e4m3fn` are not supported.
  <!-- end id4 -->
  <!-- npu="950" id5 -->
  - <term>Ascend 950PR&950DT products</term>: the data types `torch.bfloat16`, `torch.float16`, `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, and `torch.float8_e4m3fn` are supported.
  <!-- end id5 -->

- **`ckv_cache`** (`Tensor`): Required. Storage tensor for quantized or non-quantized compressed KV vectors. The data type can be `torch.bfloat16`, `torch.float16`, `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The data layout can be ND. The shape depends on `cache_mode`.

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: the data types `torch_npu.hifloat8`, `torch.float8_e5m2`, and `torch.float8_e4m3fn` are not supported.
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: the data types `torch.bfloat16`, `torch.float16`, `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, and `torch.float8_e4m3fn` are supported.
  <!-- end id7 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`k_rope_scale`** (`Tensor`): Optional. The default value is `None`. Quantization scaling factor for the key Rotary Position Embedding (RoPE). The data type can be `torch.float32`. The data layout can be ND. This parameter must be 1D with shape `[rope_size]`. This parameter must be provided in quantization mode.
- **`c_kv_scale`** (`Tensor`): Optional. The default value is `None`. Quantization scaling factor for the compressed KV. The data type can be `torch.float32`. The data layout can be ND. This parameter must be 1D with shape `[rms_size]`. This parameter must be provided in quantization mode.
- **`k_rope_offset`** (`Tensor`): Optional. The default value is `None`. Quantization offset for the key Rotary Position Embedding (RoPE). The data type can be `torch.float32`. The data layout can be ND. This parameter must be 1D with shape `[rope_size]`.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: this parameter is required in symmetric quantization mode. Asymmetric quantization is not supported currently.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: symmetric and asymmetric quantization modes are supported. In asymmetric quantization mode, this parameter is required.
  <!-- end id9 -->

- **`c_kv_offset`** (`Tensor`): Optional. The default value is `None`. Quantization offset for the compressed KV. The data type can be `torch.float32`. The data layout can be ND. This parameter must be 1D with shape `[rms_size]`.

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: this parameter is required in symmetric quantization mode. Asymmetric quantization is not supported currently.
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: symmetric and asymmetric quantization modes are supported. In asymmetric quantization mode, this parameter is required.
  <!-- end id11 -->

- **`epsilon`** (`float`): Optional. The default value is `1e-5`. Small constant used in RMS normalization to prevent division by zero.
- **`cache_mode`** (`str`): Optional. The default value is 'Norm'. Cache mode. The supported modes are described in the following table.

    | Value | Mode Name | Description |
    | --- | --- | --- |
    | Norm | KV cache update mode | The shape of `k_cache` is `[batch_size, 1, cache_length, rope_size]`, and the shape of `ckv_cache` is `[batch_size, 1, cache_length, rms_size]`. The shape of `index` is `[batch_size, seq_len]`, where the values inside `index` indicate the offset for each batch. |
    | PA/PA_BNSD | PagedAttention mode | The shape of `k_cache` is `[block_num, block_size, 1, rope_size]`, and the shape of `ckv_cache` is `[block_num, block_size, 1, rms_size]`. The shape of `index` is `[batch_size*seq_len]`, where the values inside `index` indicate the offset for each token. |
    | PA_NZ | PagedAttention mode in which the cache data format is FRACTAL_NZ | The shape of `k_cache` is `[block_num, block_size, 1, rope_size]`, and the shape of `ckv_cache` is `[block_num, block_size, 1, rms_size]`. The shape of `index` is `[batch_size * seq_len]`, where the values inside `index` indicate the offset for each token.<br>The data layout differs in different quantization modes:<br><li>In non-quantization mode: the data layout of `k_cache` is `[block_num, rope_size//16, block_size, 1, 16]`, and the data layout of `ckv_cache` is `[block_num, rms_size//16, block_size, 1, 16]`.</li><li>In quantization mode: the data layout of `k_cache` is `[block_num, rope_size//32, block_size, 1, 32]`, and the data layout of `ckv_cache` is `[block_num, rms_size//32, block_size, 1, 32]`.</li> |
    | PA_BLK_BNSD | Special PagedAttention mode | The shape of `k_cache` is `[block_num, block_size, 1, rope_size]`, and the shape of `ckv_cache` is `[block_num, block_size, 1, rms_size]`. The shape of `index` is `[batch_size*Ceil(seq_len/block_size)]`, where the values inside `index` indicate the start offset of each block and no longer correspond to tokens one by one. |
    | PA_BLK_NZ | Special PagedAttention mode in which the cache data format is FRACTAL_NZ | The shape of `k_cache` is `[block_num, block_size, 1, rope_size]`, and the shape of `ckv_cache` is `[block_num, block_size, 1, rms_size]`. The shape of `index` is `[batch_size * Ceil(seq_len / block_size)]`, where the values inside `index` indicate the start offset of each block and no longer correspond to tokens one by one.<br>The data layout differs in different quantization modes:<br><li>In non-quantization mode: the data layout of `k_cache` is `[block_num, rope_size//16, block_size, 1, 16]`, and the data layout of `ckv_cache` is `[block_num, rms_size//16, block_size, 1, 16]`.</li><li>In quantization mode: the data layout of `k_cache` is `[block_num, rope_size//32, block_size, 1, 32]`, and the data layout of `ckv_cache` is `[block_num, rms_size//32, block_size, 1, 32]`.</li> |

- **`is_output_kv`** (`bool`): Optional. Specifies whether to output the processed `k_embed_out` and `y_out` (unquantized original values). The default value is `False`, which means no output. This parameter is valid only when `cache_mode` is `(PA/PA_BNSD/PA_NZ/PA_BLK_BNSD/PA_BLK_NZ)`.

## Return Values<a name="en-us_topic_0000002236535552_section3234185215368"></a>

- **`k_cache`** (`Tensor`): Storage tensor for key vectors. The data type, shape, and data layout are identical to those of the input `k_cache` (updated in place).
- **`ckv_cache`** (`Tensor`): Storage tensor for compressed KV vectors. The data type, shape, and data layout are identical to those of the input `ckv_cache` (updated in place).
- **`k_embed_out`** (`Tensor`): Value after RoPE processing, which is valid only when `is_output_kv` is set to `True`. This parameter must be 4D with shape `[batch_size, 1, seq_len, 64]`. The data type and data layout are identical to those of the input `kv`.
- **`y_out`** (`Tensor`): Value after RMSNorm processing, which is valid only when `is_output_kv` is set to `True`. This parameter must be 4D with shape `[batch_size, 1, seq_len, 512]`. The data type and data layout are identical to those of the input `kv`.

## Constraints<a name="en-us_topic_0000002236535552_section1523425283618"></a>

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- This API supports graph mode.
  - `batch_size`: batch size.
  - `seq_len`: sequence length.
  - `hidden_size`: length of the MLA input vector.
  - `rms_size`: length of the RMSNorm branch vector.
  - `rope_size`: length of the RoPE branch vector.
  - `cache_length`: maximum KV cache length, which is valid only in `Norm` mode.
  - `block_num`: number of blocks, which is valid only in PagedAttention mode.
  - `block_size`: block size, which is valid only in PagedAttention mode.

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: only `576` is supported for `hidden_size`, only `512` for `rms_size`, and only `64` for `rope_size`.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>: `hidden_size`, `rms_size`, and `rope_size` are determined by the actual service scenario and can be set as required. `rope_size` must be an even number, and `rms_size` + `rope_size` = `hidden_size` must be satisfied.
  <!-- end id13 -->

- Quantization mode: When `k_rope_scale` and `c_kv_scale` are provided, the data type of `k_cache` and `ckv_cache` is `torch.int8`. The size of the last dimension of the cache shape must be `32` (applicable when the cache data layout is FRACTAL_NZ mode). `k_rope_scale` and `c_kv_scale` must both be provided.
- Non-quantization mode: When `k_rope_scale` and `c_kv_scale` are not provided, the data type of `k_cache` and `ckv_cache` is `torch.bfloat16` or `torch.float16`.
- Asymmetric quantization parameters: `k_rope_offset` and `c_kv_offset` are currently not supported.
- Index mapping: In all `cache_mode` configurations, the values inside `index` must be unique. If duplicate values are provided, the operator behavior is undefined and unpredictable.
    - `Norm`: The values inside `index` indicate the offset within each batch.
    - `PA/PA_BNSD/PA_NZ`: The values inside `index` indicate the global offset.
    - `PA_BLK_BNSD/PA_BLK_NZ`: The values inside `index` indicate the global offset of each page. This scenario assumes that cache updates are continuous, and cache configurations with non-continuous updates are unsupported.

- Shape association rules: Different `cache_mode` configurations have different shape rules.
    - Norm: the shape of `k_cache` is `[batch_size, 1, cache_length, rope_size]`, the shape of `ckv_cache` is `[batch_size, 1, cache_length, rms_size]`, and the shape of `index` is `[batch_size, seq_len]`, where `cache_length` >= `seq_len`.
    - Non-`Norm` modes (PagedAttention-related modes): The size of `block_num` must be greater than or equal to `Ceil(seq_len/block_size) * batch_size`.

## Examples<a name="en-us_topic_0000002236535552_section3235105212365"></a>

- Single-operator call

    ```python
    # Example 1: Basic example (non-quantized, cache_mode="PA_BNSD")
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
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    is_output_kv = True
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_cache, v_cache, k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                 k_cache, ckv_cache,
                                                                                 k_rope_scale=k_rope_scale,
                                                                                 c_kv_scale=c_kv_scale,
                                                                                 k_rope_offset=k_rope_offset,
                                                                                 c_kv_offset=c_kv_offset,
                                                                                 epsilon=epsilon,
                                                                                 cache_mode=cache_mode,
                                                                                 is_output_kv=is_output_kv)
            return k_cache, v_cache, k_rope, c_kv
    
    model = Model().npu()
    _, _, k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)
    
    # Example 2: Quantization mode
    ## 1. Create an int8 cache (quantized)
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = torch.int8).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    ## 2. Prepare quantization parameters
    k_rope_scale = torch.randn([64], dtype=torch.float32).npu()  # Scale for the RoPE portion of K
    c_kv_scale = torch.randn([512], dtype=torch.float32).npu()  # Scale for compressed KV
    # The remaining code remains unchanged

    # Example 3: Norm
    cache_mode = "Norm"
    index_shape = (batch_size, seq_len)
    index = torch.arange(start=0, end=batch_size*seq_len, step=1, dtype=torch.int64).reshape(index_shape).npu()
    is_output_kv = False
    # The remaining code remains unchanged

    # Example 4: PA_NZ
    cache_mode = "PA_NZ"
    # The remaining code remains unchanged

    # Example 5: PA_BLK_BNSD
    cache_mode = "PA_BLK_BNSD"
    index_shape = [batch_size * (seq_len + page_size - 1 ) // page_size]
    # The remaining code remains unchanged

    # Example 6: PA_BLK_NZ
    cache_mode = "PA_BLK_NZ"
    index_shape = [batch_size * (seq_len + page_size - 1 ) // page_size]
    # The remaining code remains unchanged
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
    k_rope_scale = None
    c_kv_scale = None
    cache_mode="PA_BNSD"
    index_shape = (batch_size * seq_len,)
    index = torch.arange(start=0, end=index_shape[0], step=1, dtype=torch.int64).npu()
    is_output_kv = True
    
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                    k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
                    epsilon=1e-05, cache_mode="Norm", is_output_kv=False):
            k_cache, v_cache, k_rope, c_kv = torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index,
                                                                                 k_cache, ckv_cache,
                                                                                 k_rope_scale=k_rope_scale,
                                                                                 c_kv_scale=c_kv_scale,
                                                                                 k_rope_offset=k_rope_offset,
                                                                                 c_kv_offset=c_kv_offset,
                                                                                 epsilon=epsilon,
                                                                                 cache_mode=cache_mode,
                                                                                 is_output_kv=is_output_kv)
            return k_cache, v_cache, k_rope, c_kv
    
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    _, _, k_rope, c_kv = model(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale, c_kv_scale, None, None, 1e-5, cache_mode, is_output_kv)

    # Example 1: Quantization mode
    ## 1. Create an int8 cache (quantized)
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = torch.int8).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    ## 2. Prepare quantization parameters
    k_rope_scale = torch.randn([64], dtype=torch.float32).npu() # Scale for the RoPE portion of K
    c_kv_scale = torch.randn([512], dtype=torch.float32).npu() # Scale for compressed KV
    # The remaining code remains unchanged

    # Example 2: Norm
    cache_mode = "Norm"
    index_shape = (batch_size, seq_len)
    index = torch.arange(start=0, end=batch_size*seq_len, step=1, dtype=torch.int64).reshape(index_shape).npu()
    is_output_kv = False
    # The remaining code remains unchanged

    # Example 3: PA_NZ
    cache_mode = "PA_NZ"
    # The remaining code remains unchanged

    # Example 4: PA_BLK_BNSD
    cache_mode = "PA_BLK_BNSD"
    index_shape = [batch_size * (seq_len + page_size - 1 ) // page_size]
    # The remaining code remains unchanged

    # Example 5: PA_BLK_NZ
    cache_mode = "PA_BLK_NZ"
    index_shape = [batch_size * (seq_len + page_size - 1 ) // page_size]
    # The remaining code remains unchanged
    ```
