# torch\_npu.npu\_kv\_rmsnorm\_rope\_cache<a name="en-us_topic_0000002343094197"></a>

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function<a name="en-us_topic_0000002236535552_section1023311522369"></a>

- Description: Fuses Root Mean Square Normalization (RMSNorm), Rotary Position Embedding (RoPE), and KV cache update operations (ScatterUpdate) within the Multi-head Latent Attention (MLA) structure.
    - **MLA**
      An efficient attention mechanism proposed in DeepSeek-V2 that compresses keys and values using low-rank latent representations. This mechanism reduces KV cache usage and computational overhead while maintaining long-sequence modeling capability.

    - **RoPE**
      A positional encoding method that injects positional information by applying rotational transformations to vectors. This method models relative positional relationships, makes it well-suited for long-context scenarios, and implements vector rotation through the RotateHalf operation.

    - **RMSNorm**
      A normalization method that scales inputs using the root mean square. This method is computationally efficient with low overhead and is widely used in foundation model training.

- Formulas:
    - **Input tensor `kv` splitting**: The input tensor `kv` is split into two parts, where `B` indicates the batch size and `T` indicates the sequence length.

        ![](../../figures/en-us_formulaimage_0000002239561238.png)

    - **RMS normalization**: Applied to `rms_in`.

        ![](../../figures/en-us_formulaimage_0000002239721038.png)

        - $γ∈R^{512}$ is a learnable scaling parameter.
        - $Ed[·]$ indicates the mean along the last dimension where the size of the D dimension is `512`.
        - $ε$ is a small constant (such as `0.00001`) used to prevent division by zero.
        - $⊙$ indicates element-wise multiplication.

    - **RoPE**
        1. Reshaping and transposition: Reshape and transpose `rope_in` to prepare for rotation.

            ![](../../figures/en-us_formulaimage_0000002239561242.png)

        2. Rotation operation: Apply rotary positional encoding.

            ![](../../figures/en-us_formulaimage_0000002239721042.png)

            - `cos` and `sin` are precomputed rotation parameters.
            - `RotateHalf(k)` moves the second half of elements in `k` to the first half with negation, and moves the first half to the second half while keeping original values. Specifically, when the size of the D dimension is `64`:

            ![](../../figures/en-us_formulaimage_0000002242091560.png)

## Prototype<a name="en-us_topic_0000002236535552_section123412524369"></a>

```python
torch_npu.npu_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache, *, k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, epsilon=1e-5, cache_mode='Norm', is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000002236535552_section1723416525369"></a>

> [!NOTE]   
> Variables used in tensor shapes:
>
> - `batch_size`: batch size.
> - `seq_len`: sequence length.
> - `hidden_size`: length of the MLA input vector. Only the value `576` is supported.
> - `rms_size`: length of the RMSNorm branch vector. Only the value `512` is supported.
> - `rope_size`: length of the RoPE branch vector. Only the value `64` is supported.
> - `cache_length`: maximum KV cache length, which is valid only in `Norm` mode.
> - `block_num`: number of blocks, which is valid only in PageAttention mode.
> - `block_size`: block size, which is valid only in PageAttention mode.

- **`kv`** (`Tensor`): Required. Input feature tensor. The data type can be `bfloat16` or `float16`. The data layout is `BNSD`. This parameter must be 4D with shape `[batch_size, 1, seq_len, hidden_size]`, where `hidden_size` = `rms_size` (RMS) + `rope_size` (RoPE).
- `gamma` (`Tensor`): Required. Scaling parameter for RMSNorm. The data type can be `bfloat16` or `float16`. The data layout is ND. This parameter must be 1D with shape `[rms_size]`.
- **`cos`** (`Tensor`): Required. Cosine component of RoPE. The data type can be `bfloat16` or `float16`. The data layout is ND. This parameter must be 4D with shape `[batch_size, 1, seq_len, rope_size]`.
- **`sin`** (`Tensor`): Required. Sine component of RoPE. The data type can be `bfloat16` or `float16`. The data layout is ND. This parameter must be 4D with shape `[batch_size, 1, seq_len, rope_size]`.
- **`index`** (`Tensor`): Required. Cache index tensor used to locate the write positions in `k_cache` and `ckv_cache`. The data type can be `int64`. The data layout is ND. The shape depends on `cache_mode`.
- **`k_cache`** (`Tensor`): Required. Storage tensor for quantized or non-quantized key vectors. The data type can be `bfloat16`, `float16`, or `int8`. The data layout is ND. The shape depends on `cache_mode`.
- **`ckv_cache`** (`Tensor`): Required. Storage tensor for quantized or non-quantized compressed KV vectors. The data type can be `bfloat16`, `float16`, or `int8`. The data layout is ND. The shape depends on `cache_mode`.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`k_rope_scale`** (`Tensor`): Optional. Quantization scaling factor for `key` RoPE. The default value is `None`. The data type can be `float32`. The data layout is ND. This parameter must be 1D with shape `[rope_size]`. This parameter must be provided in quantization mode.
- **`c_kv_scale`** (`Tensor`): Optional. Quantization scaling factor for compressed KV. The default value is `None`. The data type can be `float32`. The data layout is ND. This parameter must be 1D with shape `[rms_size]`. This parameter must be provided in quantization mode.
- **`k_rope_offset`** (`Tensor`): Optional. Quantization offset for the key Rotary Position Embedding (RoPE). The default value is `None`. The data type can be `float32`. The data layout is ND. This parameter must be 1D with shape `[rope_size]`. This parameter must be provided in quantization mode.
- **`c_kv_offset`** (`Tensor`): Optional. Quantization offset for compressed KV. The default value is `None`. The data type can be `float32`. The data layout is ND. This parameter must be 1D with shape `[rms_size]`. This parameter must be provided in quantization mode.
- `epsilon` (`float`): Optional. Small constant used in Root Mean Square Normalization (RMSNorm) to prevent division by zero. The default value is `1e-5`.
- `cache_mode` (`str`): Optional. Cache mode. The supported modes are described in the following table. The default value is `'Norm'`.

    <a name="en-us_topic_0000002236535552_table16997195773911"></a>
    <table><thead align="left"><tr id="en-us_topic_0000002236535552_row12998195743918"><th class="cellrowborder" valign="top" width="10.34%" id="mcps1.1.4.1.1"><p id="en-us_topic_0000002236535552_p1299819576394"><a name="en-us_topic_0000002236535552_p1299819576394"></a><a name="en-us_topic_0000002236535552_p1299819576394"></a>Value</p>
    </th>
    <th class="cellrowborder" valign="top" width="19.84%" id="mcps1.1.4.1.2"><p id="en-us_topic_0000002236535552_p93491748114015"><a name="en-us_topic_0000002236535552_p93491748114015"></a><a name="en-us_topic_0000002236535552_p93491748114015"></a>Mode Name</p>
    </th>
    <th class="cellrowborder" valign="top" width="69.82000000000001%" id="mcps1.1.4.1.3"><p id="en-us_topic_0000002236535552_p099810576395"><a name="en-us_topic_0000002236535552_p099810576395"></a><a name="en-us_topic_0000002236535552_p099810576395"></a>Description</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="en-us_topic_0000002236535552_row499835715398"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002236535552_p09984572391"><a name="en-us_topic_0000002236535552_p09984572391"></a><a name="en-us_topic_0000002236535552_p09984572391"></a>Norm</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="en-us_topic_0000002236535552_p23491348164018"><a name="en-us_topic_0000002236535552_p23491348164018"></a><a name="en-us_topic_0000002236535552_p23491348164018"></a>KV cache update mode</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="en-us_topic_0000002236535552_p1499812579399"><a name="en-us_topic_0000002236535552_p1499812579399"></a><a name="en-us_topic_0000002236535552_p1499812579399"></a>The shape of <code>k_cache</code> is <code>[batch_size, 1, cache_length, rope_size]</code>, and the shape of <code>ckv_cache</code> is <code>[batch_size, 1, cache_length, rms_size]</code>. The shape of <code>index</code> is <code>[batch_size, seq_len]</code>, where the values inside <code>index</code> indicate the offset for each batch.</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002236535552_row1699810579398"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002236535552_p1699810576392"><a name="en-us_topic_0000002236535552_p1699810576392"></a><a name="en-us_topic_0000002236535552_p1699810576392"></a>PA/PA_BNSD</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="en-us_topic_0000002236535552_p1934914484406"><a name="en-us_topic_0000002236535552_p1934914484406"></a><a name="en-us_topic_0000002236535552_p1934914484406"></a>PagedAttention mode</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="en-us_topic_0000002236535552_p19457162584119"><a name="en-us_topic_0000002236535552_p19457162584119"></a><a name="en-us_topic_0000002236535552_p19457162584119"></a>The shape of <code>k_cache</code> is <code>[block_num, block_size, 1, rope_size]</code>, and the shape of <code>ckv_cache</code> is <code>[block_num, block_size, 1, rms_size]</code>. The shape of <code>index</code> is <code>[batch_size * seq_len]</code>, where the values inside <code>index</code> indicate the offset for each token.</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002236535552_row179981757193914"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002236535552_p99981257113913"><a name="en-us_topic_0000002236535552_p99981257113913"></a><a name="en-us_topic_0000002236535552_p99981257113913"></a>PA_NZ</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="en-us_topic_0000002236535552_p6349174814019"><a name="en-us_topic_0000002236535552_p6349174814019"></a><a name="en-us_topic_0000002236535552_p6349174814019"></a>PageAttention mode with cache data layout of FRACTAL_NZ</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="en-us_topic_0000002236535552_p15265174112368"><a name="en-us_topic_0000002236535552_p15265174112368"></a><a name="en-us_topic_0000002236535552_p15265174112368"></a>The shape of <code>k_cache</code> is <code>[block_num, block_size, 1, rope_size]</code>, and the shape of <code>ckv_cache</code> is <code>[block_num, block_size, 1, rms_size]</code>. The shape of <code>index</code> is <code>[batch_size * seq_len]</code>, where the values inside <code>index</code> indicate the offset for each token.</p>
    <p id="en-us_topic_0000002236535552_p10572135917168"><a name="en-us_topic_0000002236535552_p10572135917168"></a><a name="en-us_topic_0000002236535552_p10572135917168"></a><br>The data layout varies across different quantization modes:</p>
    <a name="en-us_topic_0000002236535552_ul183421286166"></a><a name="en-us_topic_0000002236535552_ul183421286166"></a><ul id="en-us_topic_0000002236535552_ul183421286166"><li>Non-quantization mode: The data layout of <code>k_cache</code> is <code>[block_num, rope_size//16, block_size, 1, 16]</code>, and the data layout of <code>ckv_cache</code> is <code>[block_num, rms_size//16, block_size, 1, 16]</code>.</li><li>Quantization mode: The data layout of <code>k_cache</code> is <code>[block_num, rope_size//32, block_size, 1, 32]</code>, and the data layout of <code>ckv_cache</code> is <code>[block_num, rms_size//32, block_size, 1, 32]</code>.</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000002236535552_row428921319407"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002236535552_p18290111311409"><a name="en-us_topic_0000002236535552_p18290111311409"></a><a name="en-us_topic_0000002236535552_p18290111311409"></a>PA_BLK_BNSD</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="en-us_topic_0000002236535552_p12349144844010"><a name="en-us_topic_0000002236535552_p12349144844010"></a><a name="en-us_topic_0000002236535552_p12349144844010"></a>Special PageAttention mode</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="en-us_topic_0000002236535552_p329011315406"><a name="en-us_topic_0000002236535552_p329011315406"></a><a name="en-us_topic_0000002236535552_p329011315406"></a>The shape of <code>k_cache</code> is <code>[block_num, block_size, 1, rope_size]</code>, and the shape of <code>ckv_cache</code> is <code>[block_num, block_size, 1, rms_size]</code>. The shape of <code>index</code> is <code>[batch_size * Ceil(seq_len/block_size)]</code>, where the values inside <code>index</code> indicate the starting offset of each block and no longer correspond to tokens on a one-to-one basis.</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002236535552_row182905133408"><td class="cellrowborder" valign="top" width="10.34%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002236535552_p13290313134017"><a name="en-us_topic_0000002236535552_p13290313134017"></a><a name="en-us_topic_0000002236535552_p13290313134017"></a>PA_BLK_NZ</p>
    </td>
    <td class="cellrowborder" valign="top" width="19.84%" headers="mcps1.1.4.1.2 "><p id="en-us_topic_0000002236535552_p173491486403"><a name="en-us_topic_0000002236535552_p173491486403"></a><a name="en-us_topic_0000002236535552_p173491486403"></a>Special PageAttention mode with cache data layout of FRACTAL_NZ</p>
    </td>
    <td class="cellrowborder" valign="top" width="69.82000000000001%" headers="mcps1.1.4.1.3 "><p id="en-us_topic_0000002236535552_p637504423711"><a name="en-us_topic_0000002236535552_p637504423711"></a><a name="en-us_topic_0000002236535552_p637504423711"></a>The shape of <code>k_cache</code> is <code>[block_num, block_size, 1, rope_size]</code>, and the shape of <code>ckv_cache</code> is <code>[block_num, block_size, 1, rms_size]</code>. The shape of <code>index</code> is <code>[batch_size * Ceil(seq_len/block_size)]</code>, where the values inside <code>index</code> indicate the starting offset of each block and no longer correspond to tokens on a one-to-one basis.</p>
    <p id="en-us_topic_0000002236535552_p916702812303"><a name="en-us_topic_0000002236535552_p916702812303"></a><a name="en-us_topic_0000002236535552_p916702812303"></a><br>The data layout varies across different quantization modes:</p>
    <a name="en-us_topic_0000002236535552_ul20871102344210"></a><a name="en-us_topic_0000002236535552_ul20871102344210"></a><ul id="en-us_topic_0000002236535552_ul20871102344210"><li><br>Non-quantization mode: The data layout of <code>k_cache</code> is <code>[block_num, rope_size // 16, block_size, 1, 16]</code>, and the data layout of <code>ckv_cache</code> is <code>[block_num, rms_size // 16, block_size, 1, 16]</code>.</li><li>Quantization mode: The data layout of <code>k_cache</code> is <code>[block_num, rope_size // 32, block_size, 1, 32]</code>, and the data layout of <code>ckv_cache</code> is <code>[block_num, rms_size // 32, block_size, 1, 32]</code>.</li></ul>
    </td>
    </tr>
    </tbody>
    </table>

- **`is_output_kv`** (`bool`): Optional. Specifies whether to output the processed `k_embed_out` and `y_out` (unquantized original values). The default value is `False`. This parameter is valid only when `cache_mode` is PA, PA_BNSD, PA_NZ, PA_BLK_BNSD, or PA_BLK_NZ.

## Return Values<a name="en-us_topic_0000002236535552_section3234185215368"></a>

- **`k_cache`** (`Tensor`): Storage tensor for key vectors. The data type, shape, and data layout are identical to those of the input `k_cache` (updated in place).
- **`ckv_cache`** (`Tensor`): Storage tensor for compressed KV vectors. The data type, shape, and data layout are identical to those of the input `ckv_cache` (updated in place).
- **`k_embed_out`** (`Tensor`): Value after RoPE processing, which is valid only when `is_output_kv` is set to `True`. This parameter must be 4D with shape `[batch_size, 1, seq_len, 64]`. The data type and data layout are identical to those of the input `kv`.
- **`y_out`** (`Tensor`): Value after RMSNorm processing, which is valid only when `is_output_kv` is set to `True`. This parameter must be 4D with shape `[batch_size, 1, seq_len, 512]`. The data type and data layout are identical to those of the input `kv`.

## Constraints<a name="en-us_topic_0000002236535552_section1523425283618"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- Quantization mode: When `k_rope_scale` and `c_kv_scale` are provided, the data type of `k_cache` and `ckv_cache` is `int8`. The size of the last dimension of the cache shape must be `32` (applicable when the cache data layout is FRACTAL_NZ mode). `k_rope_scale` and `c_kv_scale` must both be provided.
- Non-quantization mode: When `k_rope_scale` and `c_kv_scale` are not provided, the data type of `k_cache` and `ckv_cache` is `bfloat16` or `float16`.
- Asymmetric quantization parameters: `k_rope_offset` and `c_kv_offset` are currently not supported.
- Index mapping: In all `cache_mode` configurations, the values inside `index` must be unique. If duplicate values are provided, the operator behavior is undefined and unpredictable.
    - `Norm`: The values inside `index` indicate the offset within each batch.
    - `PA/PA_BNSD/PA_NZ`: The values inside `index` indicate the global offset.
    - `PA_BLK_BNSD/PA_BLK_NZ`: The values inside `index` indicate the global offset of each page. This scenario assumes that cache updates are continuous, and cache configurations with non-continuous updates are unsupported.

- Shape association rules: Different `cache_mode` configurations have different shape rules.
    - `Norm`: The shape of `k_cache` is `[batch_size, 1, cache_length, rope_size]`, the shape of `ckv_cache` is `[batch_size, 1, cache_length, rms_size]`, and the shape of `index` is `[batch_size, seq_len]`, where `cache_length` must be greater than or equal to `seq_len`.
    - Non-`Norm` modes (PageAttention-related modes): The size of `block_num` must be greater than or equal to `Ceil(seq_len/block_size) * batch_size`.

## Examples<a name="en-us_topic_0000002236535552_section3235105212365"></a>

- Single-operator call

    ```python
    # Example 1: Basic example (non-quantization, cache_mode="PA_BNSD")
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
    ## 1. Create an int8 cache (quantization).
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = torch.int8).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    ## 2. Prepare quantization parameters.
    k_rope_scale = torch.randn([64], dtype=torch.float32).npu() # Quantization scaling factor for key RoPE
    c_kv_scale = torch.randn([512], dtype=torch.float32).npu() # Quantization scaling factor for compressed KV
    # Other code remains unchanged.

    # Example 3: Norm
    cache_mode = "Norm"
    index_shape = (batch_size, seq_len)
    index = torch.arange(start=0, end=batch_size*seq_len, step=1, dtype=torch.int64).reshape(index_shape).npu()
    is_output_kv = False
    # Other code remains unchanged.

    # Example 4: PA_NZ
    cache_mode = "PA_NZ"
    # Other code remains unchanged.

    # Example 5: PA_BLK_BNSD
    cache_mode = "PA_BLK_BNSD"
    index_shape = (batch_size * (seq_len + page_size - 1 ) // page_size)
    # Other code remains unchanged.

    # Example 5: PA_BLK_NZ
    cache_mode = "PA_BLK_NZ"
    index_shape = (batch_size * (seq_len + page_size - 1 ) // page_size)
    # Other code remains unchanged.
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

    # Example 2: Quantization mode
    ## 1. Create an int8 cache (quantization).
    k_cache = torch.ones(page_num, page_size, 1, 64, dtype = torch.int8).npu()
    ckv_cache = torch.ones(page_num, page_size, 1, 512, dtype = input_dtype).npu()
    ## 2. Prepare quantization parameters.
    k_rope_scale = torch.randn([64], dtype=torch.float32).npu() # Quantization scaling factor for key RoPE
    c_kv_scale = torch.randn([512], dtype=torch.float32).npu() # Quantization scaling factor for compressed KV
    # Other code remains unchanged.

    # Example 3: Norm
    cache_mode = "Norm"
    index_shape = (batch_size, seq_len)
    index = torch.arange(start=0, end=batch_size*seq_len, step=1, dtype=torch.int64).reshape(index_shape).npu()
    is_output_kv = False
    # Other code remains unchanged.

    # Example 4: PA_NZ
    cache_mode = "PA_NZ"
    # Other code remains unchanged.

    # Example 5: PA_BLK_BNSD
    cache_mode = "PA_BLK_BNSD"
    index_shape = (batch_size * (seq_len + page_size - 1 ) // page_size)
    # Other code remains unchanged.

    # Example 5: PA_BLK_NZ
    cache_mode = "PA_BLK_NZ"
    index_shape = (batch_size * (seq_len + page_size - 1 ) // page_size)
    # Other code remains unchanged.
    ```
