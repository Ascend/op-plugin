# torch_npu.npu_mla_prolog_v3

## Supported Products

| Product     | Supported|
|:----------------------------|:-----------:|
|<term>Atlas A3 inference products</term>|      √     |

## Function

- Description: Performs computation during Multi-Head Latent Attention (MLA) preprocessing in inference scenarios. This operator implements four parallel computation paths:
    1. Standard query path: Input $x$ → Downsampling through $W^{DQ}$ → `RmsNorm` → Upsampling through $W^{UQ}$ → Upsampling through $W^{UK}$ → $q^N$
    2. Positional encoding query path: Input $x$ → Downsampling through $W^{DQ}$ → `RmsNorm` → $W^{QR}$ → Rotary positional encoding (`ROPE`) → $q^R$
    3. Standard key path: Input $x$ → Downsampling through $W^{DKV}$ → `RmsNorm` → Committed to a cache layer → $k^C$
    4. Positional encoding key path: Input $x$ → $W^{KR}$ → Rotary positional encoding (`ROPE`) → Committed to a cache layer → $k^R$

- The main changes compared with `torch_npu.npu_mla_prolog_v2` are as follows:
    - New outputs `query_norm` and `dequant_scale_q_norm` are provided to support the DeepSeek V3.2 network architecture.
    - A `pertile` quantization mode is added for `kv_cache`.
    - Scale correction factors for the query and key are added. They correspond to `qc_qr_scale` ($\alpha_q$) and `kc_scale` ($\alpha_{kv}$), respectively.
    - `cache_mode` is updated to support `"PA_BLK_BSND"`, `"PA_BLK_NZ"`, `"BSND"`, and `"TND"` data layouts.
    - Optional parameters `weight_quant_mode`, `kv_cache_quant_mode`, `query_quant_mode`, `ckvkr_repo_mode`, and `quant_scale_repo_mode` are added to configure quantization scenarios.
    - `cache_index` is modified to function as an optional parameter.

- Formulas:
    - `RmsNorm` formula:
        $$
        \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
        $$

        $$
        \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
        $$

    - Path 1: Standard query computation

        This path includes downsampling, `RmsNorm`, and two upsampling steps.

        $$
        c^Q = RmsNorm(x \cdot W^{DQ})
        $$

        $$
        q^C = c^Q \cdot W^{UQ}
        $$

        $$
        q^N = q^C \cdot W^{UK}
        $$

    - Path 2: Positional encoding query computation

        This path applies `ROPE` rotary positional encoding to the query.

        $$
        q^R = ROPE(c^Q \cdot W^{QR})
        $$

    - Path 3: Standard key computation

        This path includes downsampling and `RmsNorm`. It commits the computation results to a cache layer.

        $$
        c^{KV} = RmsNorm(x \cdot W^{DKV})
        $$

        $$
        k^C = Cache(c^{KV})
        $$

    - Path 4: Positional encoding key computation

        This path applies `ROPE` rotary positional encoding to the key. It commits the computation results to a cache layer.

        $$
        k^R = Cache(ROPE(x \cdot W^{KR}))
        $$

## Prototype

```python
torch_npu.npu_mla_prolog_v3(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=None, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None, quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None, actual_seq_len=None, k_nope_clip_alpha=None, rmsnorm_epsilon_cq=1e-05, rmsnorm_epsilon_ckv=1e-05, cache_mode='PA_BSND', query_norm_flag=False, weight_quant_mode=0, kv_cache_quant_mode=0, query_quant_mode=0, ckvkr_repo_mode=0, quant_scale_repo_mode=0, tile_size=128, qc_qr_scale=1.0, kc_scale=1.0) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

> [!NOTE]
> `B` (Batch Size) indicates the input sample batch size.<br>`S` (Sequence Length) indicates the input sample sequence length.<br>`He` (Head Size) indicates the hidden layer size.<br>`N` (Head Num) indicates the attention head count.<br>`Hcq` indicates the dimension of the low-rank query matrix.<br>`Hckv` indicates the dimension of the low-rank KV matrix.<br>`Dtile` indicates the `kv_cache` D-axis dimension.<br>`D` indicates the query and key dimension excluding positional encoding.<br>`Dr` indicates the query and key positional encoding dimension.<br>`Nkv` indicates the attention head count for key and value.<br>`BlockNum` indicates the number of blocks in the PagedAttention scenario.<br>`BlockSize` indicates the block size in the PagedAttention scenario.<br>`T` indicates the size after the fusion of the `B` and `S` axes.

- **`token_x`** (`Tensor`): Required. Input tensor used to compute the query and key in the formulas. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16` or `int8`. When `B` and `S` axes are fused, the shape of this parameter is `[T, He]`. When `B` and `S` axes are not fused, the shape is `[B, S, He]`.

- **`weight_dq`** (`Tensor`): Required. Downsampling weight matrix for query computation, $W^{DQ}$ in the formulas. Non-contiguous tensors are not supported. The data layout can be FRACTAL_NZ. The data type can be `bfloat16` or `int8`. The shape of this parameter is `[He, Hcq]`.

- **`weight_uq_qr`** (`Tensor`): Required. Combined upsampling weight matrix and positional encoding weight matrix for query computation, $W^{UQ}$ and $W^{QR}$ in the formulas. Non-contiguous tensors are not supported. The data layout can be FRACTAL_NZ. The data type can be `bfloat16` or `int8`. The shape of this parameter is `[Hcq, N * (D + Dr)]`.

- **`weight_uk`** (`Tensor`): Required. Upsampling weight matrix for key computation, $W^{UK}$ in the formulas. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16`. The shape of this parameter is `[N, D, Hckv]`.

- **`weight_dkv_kr`** (`Tensor`): Required. Combined downsampling weight matrix and positional encoding weight matrix for key computation, $W^{DKV}$ and $W^{KR}$ in the formulas. Non-contiguous tensors are not supported. The data layout can be FRACTAL_NZ. The data type can be `bfloat16` or `int8`. The shape of this parameter is `[He, Hckv+Dr]`.

- **`rmsnorm_gamma_cq`** (`Tensor`): Required. The $\gamma$ parameter in the `RmsNorm` formula for computing $c^Q$. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16`. The shape of this parameter is `[Hcq]`.

- **`rmsnorm_gamma_ckv`** (`Tensor`): Required. The $\gamma$ parameter in the `RmsNorm` formula for computing $c^{KV}$. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16`. The shape of this parameter is `[Hckv]`.

- **`rope_sin`** (`Tensor`): Required. Sine parameter matrix used to compute rotary positional encodings (`ROPE`). Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16`. When `B` and `S` axes are fused, the shape of this parameter is `[T, Dr]`. When `B` and `S` axes are not fused, the shape is `[B, S, Dr]`. Empty tensors are supported when `B=0`, `S=0`, and `T=0`.

- **`rope_cos`** (`Tensor`): Required. Cosine parameter matrix used to compute rotary positional encodings (`ROPE`). Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16`. When `B` and `S` axes are fused, the shape of this parameter is `[T, Dr]`. When `B` and `S` axes are not fused, the shape is `[B, S, Dr]`. Empty tensors are supported when `B=0`, `S=0`, and `T=0`.

- **`kv_cache`** (`Tensor`): Required. Cache tensor for storing key states, updated in place, $k^C$ in the formulas. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16` or `int8`. When `cache_mode` is set to `"PA_BSND"`, `"PA_NZ"`, `"PA_BLK_BSND"`, or `"PA_BLK_NZ"`, the shape of this parameter is `[BlockNum, BlockSize, Nkv, Dtile]`. That is, empty tensors are supported when `B=0` and `Skv=0`. When `cache_mode` is set to `"BSND"`, the shape is `[B, S, Nkv, Dtile]`. That is, empty tensors are not supported. When `cache_mode` is set to `"TND"`, the shape is `[T, Nkv, Dtile]`. That is, empty tensors are not supported. `Nkv` is associated with `N`. Here, `N` indicates a hyperparameter. Therefore, configuring `Nkv=0` is not supported.

- **`kr_cache`** (`Tensor`): Required. Cache tensor for key rotary positional encoding, updated in place, $k^R$ in the formulas. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `bfloat16` or `int8`. When `cache_mode` is set to `"PA_BSND"`, `"PA_NZ"`, `"PA_BLK_BSND"`, or `"PA_BLK_NZ"`, the shape of this parameter is `[BlockNum, BlockSize, Nkv, Dr]`. That is, empty tensors are supported when `B=0` and `Skv=0`. When `cache_mode` is set to `"BSND"`, the shape is `[B, S, Nkv, Dr]`. That is, empty tensors are not supported. When `cache_mode` is set to `"TND"`, the shape is `[T, Nkv, Dr]`. That is, empty tensors are not supported. `Nkv` is associated with `N`. Here, `N` represents a hyperparameter. Therefore, configuring `Nkv=0` is not supported.

- **`*`**: Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).

- **`cache_index`** (`Tensor`): Optional. Index for storing `kv_cache` and `kr_cache`. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `int64`. When `cache_mode` is set to `"PA_BSND"` or `"PA_NZ"`, if `B` and `S` axes are fused, the shape is `[T]`; if `B` and `S` axes are not fused, the shape is `[B, S]`. The value range is `[0, BlockNum * BlockSize)`. When `cache_mode` is set to `"PA_BLK_BSND"` or `"PA_BLK_NZ"`, if `B` and `S` axes are fused, the shape is `[Sum(Ceil(S_i/BlockSize))]`, where `S_i` represents the sequence length of the $i$-th batch; if `B` and `S` axes are not fused, the shape is `[B, Ceil(S/BlockSize)]`. The value range is `[0, BlockNum)`. When `cache_mode` is set to `"BSND"` or `"TND"`, this parameter does not need to be provided. The validity of input values is not verified internally, and must be ensured by the user.

- **`dequant_scale_x`** (`Tensor`): Optional. Dequantization scale parameter for `token_x`. Non-contiguous tensors are not supported. The supported data layout is ND. The data type can be `float`. The shape is [T] or [B*S, 1]. Empty tensors where `B=0`, `S=0`, and `T=0` are supported.

- **`dequant_scale_w_dq`** (`Tensor`): Optional. Dequantization scale parameter for `weight_dq`. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `float`. The shape of this parameter is `[1, Hcq]`.

- **`dequant_scale_w_uq_qr`** (`Tensor`): Optional. Per-channel dequantization scale parameter used after the `MatmulQcQr` matrix multiplication operation. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `float`. The shape of this parameter is `[1, N*(D+Dr)]`.

- **`dequant_scale_w_dkv_kr`** (`Tensor`): Optional. Dequantization scale parameter for `weight_dkv_kr`. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `float`. The shape of this parameter is `[1, Hckv+Dr]`.

- **`quant_scale_ckv`** (`Tensor`): Optional. Parameter used for quantizing the `kv_cache` output data. Non-contiguous tensors are not supported. The data layout can be `ND`. The data type can be `float`. The shape is `[1, Hckv]` in *partial quantization* scenarios or `[1]` in *full quantization* scenarios. Non-empty tensors are supported. This parameter is required only when the data type of the `kv_cache` output is `int8`.

- **`quant_scale_ckr`** (`Tensor`): Optional. Parameter used for quantizing the `kr_cache` output data. Non-contiguous tensors are not supported. The data layout can be `ND`, the data type can be `float`. The shape is `(1, Dr)`. Non-empty tensors are supported. This parameter is required only for `int8` quantized output scenarios.

- **`smooth_scales_cq`** (`Tensor`): Optional. Scaling factors used for dynamic quantization of the `RmsNorm_cq` output. Non-contiguous tensors are not supported. The data layout can be `ND`, the data type can be `float`. The shape is `[1, Hcq]` or `[1]`. Non-empty tensors are supported. This parameter is optional for `int8` quantized output scenarios.

- **`actual_seq_len`** (`Tensor`): Optional. Sequence length of each batch, stored in prefix-sum form. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `int32`. The shape of this parameter is `[B]`. Non-empty tensors are required. This parameter must be provided only when `B` and `S` axes are fused and `cache_mode` is set to `"PA_BLK_BSND"` or `"PA_BLK_NZ"`. The validity of input values is not verified internally, and must be ensured by the user.

- **`k_nope_clip_alpha`** (`Tensor`): Optional. Scaling factor for clipping `kv_cache`. This parameter takes effect only in `pertile` `kv_cache` quantization scenarios. Non-contiguous tensors are not supported. The data layout can be ND. The data type can be `float`. The shape of this parameter is `[1]`.

- **`rmsnorm_epsilon_cq`** (`float`): Optional. The $\epsilon$ parameter in the `RmsNorm` formula for computing $c^Q$. The default value is `1e-05`.

- **`rmsnorm_epsilon_ckv`** (`float`): Optional. The $\epsilon$ parameter in the `RmsNorm` formula for computing $c^{KV}$. The default value is `1e-05`.

- `cache_mode` (`str`): Optional. Layout mode of `kv_cache`. The supported values are `"PA_BSND"`, `"PA_NZ"`, `"PA_BLK_BSND"`, `"PA_BLK_NZ"`, `"TND"` (corresponding to fused `B` and `S` axes), and `"BSND"` (corresponding to non-fused `B` and `S` axes). The default value is `"PA_BSND"`.

- **`query_norm_flag`** (`bool`): Optional. Specifies whether to output `query_norm`. Only the `bool` data type is supported. When set to `False`, `query_norm` is not output. When set to `True`, `query_norm` is output and is accompanied by the output of `dequant_scale_q_norm` in quantization scenarios. The default value is `False`.

- **`weight_quant_mode`** (`int`): Optional. Quantization mode for `weight_dq`, `weight_uq_qr`, `weight_uk`, and `weight_dkv_kr`. Valid values are `0` (non-quantization), `1` (`weight_uq_qr` quantization), or `2` (`weight_dq`, `weight_uq_qr`, and `weight_dkv_kr` quantization). The default value is `0`.

- **`kv_cache_quant_mode`** (`int`): Optional. Quantization mode of `kv_cache`. Valid values are `0` (non-quantization), `1` (`pertensor` quantization), `2` (`perchannel` quantization) or `3` (`pertile` quantization). The default value is `0`.

- **`query_quant_mode`** (`int`): Optional. Quantization mode of the query. `0` specifies non-quantization. `1` specifies `per-token-head` quantization. The default value is `0`.

- **`ckvkr_repo_mode`** (`int`): Optional. Storage mode of `kv_cache` and `kr_cache`. `0` specifies that `kv_cache` and `kr_cache` are stored separately. `1` specifies that `kv_cache` and `kr_cache` are stored together. The default value is `0`.

- **`quant_scale_repo_mode`** (`int`): Optional. Storage mode of the quantization scale. `0` specifies that the quantization scale and data are stored separately. `1` specifies that the quantization scale and data are stored together. The default value is `0`.

- **`tile_size`** (`int`): Optional. Size of each tile in `pertile` quantization. This parameter is valid only when `kv_cache_quant_mode` is set to `3`. The default value is `128`.

- **`qc_qr_scale`** (`float`): Optional. Scale correction factor for the query. The default value is `1.0`.

- **`kc_scale`** (`float`): Optional. Scale correction factor for the key. The default value is `1.0`.

## Return Values

- **`query`** (`Tensor`): Query output tensor, corresponding to q<sup>N</sup> in the formulas. The data layout can be ND. The data type can be `bfloat16` or `int8`. This parameter must be 3D or 4D with shape `(T, N, Hckv)` or `(B, S, N, Hckv)`.

- **`query_rope`** (`Tensor`): Output tensor for query positional encoding, q<sup>R</sup> in the formulas. The data layout can be ND. The data type can be `bfloat16`. This parameter must be 3D or 4D with shape `(T, N, Dr)` or `(B, S, N, Dr)`.

- **`dequant_scale_q_nope`** (`Tensor`): Dequantization parameter of the output tensor of the query. The data layout can be ND. The data type can be `float`. This parameter must be 1D or 3D. In full KV cache quantization scenarios, the shape is `(T, N, 1)` or `(B * S, N, 1)`. In other scenarios, the shape is `[0]`.

- **`query_norm`** (`Tensor`): Output tensor of the query after `RmsNorm_cq`, $q^C$ in the formulas. The data layout can be ND. The data type can be `bfloat16` or `int8`. This parameter must be a 2D or 3D tensor. It is valid when `query_norm_flag` is set to `True`, with shape `(T, Hcq)` or `(B, S, Hcq)`. This parameter is invalid when `query_norm_flag` is set to `False`, and the shape is `[0]`.

- **`dequant_scale_q_norm`** (`Tensor`): Dequantization parameter of the query after `RmsNorm_cq`. The data layout can be ND. The data type can be `float`. This parameter must be a 1D or 3D tensor. It is valid when `query_norm_flag` is set to `True` and `weight_quant_mode` is set to `1` or `2`, with shape `(T, 1)` or `(B * S, 1)`. In other scenarios, this parameter is invalid, and the shape is `[0]`.

## Constraints

- This API can be used in inference scenarios.

- This API supports both single-operator mode and graph mode.

- The following table describes the shape layout fields.

    | Field      | Full Spelling/Description                 | Value Rule and Description                                                                |
    |--------------|--------------------------------|------------------------------------------------------------------------------|
    | B            | Batch (batch size of input samples)     | Value range: 0 to 65536.                                                          |
    | S            | Seq-Length (sequence length of input samples)| Value range: not limited.                                                             |
    | He           | Head-Size (hidden layer size)       | The value is fixed at `7168` or `7680`.                                                           |
    | Hcq          | Dimension of the low-rank q matrix                | The value is fixed at `1536`.                                                          |
    | N            | Head-Num (number of heads)            | Valid values: `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`.                                      |
    | Hckv         | Dimension of the low-rank KV matrix               | The value is fixed at `512`.                                                            |
    | Dtile        | D-axis dimension of `kv_cache`             | The value is fixed at `656` in `pertile` mode and `512` in non-`pertensor` modes.                                                         |
    | D            | QK dimension without position encoding           | The value is fixed at `128`.                                                            |
    | Dr           | QK positional encoding dimension               | The value is fixed at `64`.                                                             |
    | Nkv          | Number of KV heads                 | The value is fixed at `1`.                                                              |
    | BlockNum     | Number of blocks per tile in the PagedAttention scenario   | The value is rounded up to the nearest integer after the result of `B*Skv/BlockSize` is calculated. (`Skv` indicates the sequence length of KV, which can be `0`.)|
    | BlockSize    | Block size in the PagedAttention scenario | Value range: a multiple of 16 ranging from 16 to 1024.                                                          |
    | T            | Size after the fusion of the `B` and `S` axes               | Value range: not limited. Note: When the `B` and `S` axes are fused, `token_x`, `rope_sin`, `rope_cos`, and `query_norm` are 2D, `query_out` and `query_rope_out` are 3D, and `cache_index` is 1D.|

- Supported scenarios:
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">Scenario</th>
      <th>Constraint</th>
    </tr>
    <tr>
      <td colspan="2"><em>Non-quantization</em></td>
      <td>
          - <code>weight_quant_mode=0</code>, <code>kv_cache_quant_mode=0</code>, <code>query_quant_mode=0</code><br>
          - Inputs: All inputs are non-quantized data.<br>
          - Outputs: All outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3"><em>Partial quantization</em></td>
      <td><code>kv_cache</code> non-quantized</td>
      <td>
          - <code>weight_quant_mode=1</code>, <code>kv_cache_quant_mode=0</code>, <code>query_quant_mode=0</code><br>
          - Inputs: <code>weight_uq_qr</code> must be provided as <code>pertoken</code> quantized data, and all other inputs must be non-quantized data. The <code>dequant_scale_w_uq_qr</code> field is required, and the <code>smooth_scale_cq</code> field is optional.<br>
          - Outputs: All outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td><code>kv_cache</code> quantized in <code>perchannel</code> mode</td>
      <td>
          - <code>weight_quant_mode=2</code>, <code>kv_cache_quant_mode=2</code>, <code>query_quant_mode=0</code><br>
          - Inputs: <code>weight_uq_qr</code> must be provided as <code>pertoken</code> quantized data, <code>kv_cache</code> and <code>kr_cache</code> must be provided as <code>perchannel</code> quantized data, and all other inputs must be non-quantized data.<br>
          The <code>dequant_scale_w_uq_qr</code>, <code>quant_scale_ckv</code>, and <code>quant_scale_ckr</code> fields are required, and the <code>smooth_scale_cq</code> field is optional.<br>
          - Outputs: <code>kv_cache</code> and <code>kr_cache</code> are <code>perchannel</code> quantized data, and all other outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td><code>kv_cache</code> quantized in <code>pertile</code> mode</td>
      <td>
          - weight_quant_mode=3, kv_cache_quant_mode=3, query_quant_mode=0<br>
          - Inputs: <code>weight_uq_qr</code> must be provided as <code>pertoken</code> quantized data, <code>kv_cache</code> must be provided as <code>pertile</code> quantized data, and all other inputs must be non-quantized data.<br>
          The <code>dequant_scale_w_uq_qr</code> and <code>quant_scale_ckv</code> fields are required, and the <code>smooth_scale_cq</code> field is optional.<br>
          - Outputs: <code>kv_cache_out</code> is <code>pertile</code> quantized data, and all other outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td rowspan="3"><em>Full quantization</em></td>
      <td><code>kv_cache</code> non-quantized</td>
      <td>
          - <code>weight_quant_mode=2</code>, <code>kv_cache_quant_mode=0</code>, <code>query_quant_mode=0</code><br>
          - Inputs: <code>token_x</code> must be provided as pertoken quantized data, <code>weight_dq</code>, <code>weight_uq_qr</code>, and <code>weight_dkv_kr</code> must be provided as <code>perchannel</code> quantized data, and all other inputs must be non-quantized data.<br>
          The <code>dequant_scale_x</code>, <code>dequant_scale_w_dq</code>, <code>dequant_scale_w_uq_qr</code>, and <code>dequant_scale_w_dkv_kr</code> fields are required, and the <code>smooth_scale_cq</code> field is optional.<br>
          - Outputs: All outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td><code>kv_cache</code> quantized in <code>pertensor</code> mode</td>
      <td>
          - <code>weight_quant_mode=2</code>, <code>kv_cache_quant_mode=1</code>, <code>query_quant_mode=1</code><br>
          - Inputs: <code>token_x</code> must be provided as pertoken quantized data, <code>weight_dq</code>, <code>weight_uq_qr</code>, and <code>weight_dkv_kr</code> must be provided as <code>perchannel</code> quantized data, <code>kv_cache</code> must be provided as <code>pertensor</code> quantized data, and all other inputs must be non-quantized data.<br>
          The <code>dequant_scale_x</code>, <code>dequant_scale_w_dq</code>, <code>dequant_scale_w_uq_qr</code>, <code>dequant_scale_w_dkv_kr</code>, and <code>quant_scale_ckv</code> fields are required, and the <code>smooth_scale_cq</code> field is optional<br>.<br>
          - Outputs: <code>query_out</code> is <code>pertoken_head</code> quantized data, the <code>kv_cache</code> output parameter is <code>pertensor</code> quantized data, and all other outputs are non-quantized data.
      </td>
    </tr>
    <tr>
      <td><code>kv_cache</code> quantized in <code>pertile</code> mode</td>
      <td>
          - <code>weight_quant_mode=3</code>, <code>kv_cache_quant_mode=3</code>, <code>query_quant_mode=1</code><br>
          - Inputs: <code>token_x</code> must be provided as pertoken quantized data, <code>weight_dq</code>, <code>weight_uq_qr</code>, and <code>weight_dkv_kr</code> must be provided as <code>perchannel</code> quantized data, and all other inputs must be non-quantized data.<br>
          The <code>dequant_scale_x</code>, <code>dequant_scale_w_dq</code>, <code>dequant_scale_w_uq_qr</code>, <code>dequant_scale_w_dkv_kr</code>, and <code>quant_scale_ckv</code> fields are required, and the <code>smooth_scale_cq</code> field is optional<br>.<br>
          - Outputs: <code>query_out</code> is <code>pertoken_head</code> quantized data, the <code>kv_cache</code> output parameter is <code>pertensor</code> quantized data, and all other outputs are non-quantized data.
      </td>
    </tr>
  </table>

- The following table describes the parameter dtype constraints in different quantization scenarios.
  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="2">Parameter</th>
      <th><em>Non-quantization</em></th>
      <th colspan="3"><em>Partial Quantization</em></th>
      <th colspan="3"><em>Full Quantization</em></th>
    </tr>
    <tr>
      <th>dtype</th>
      <th><code>kv_cache</code> non-quantized<br>dtype</th>
      <th><code>kv_cache</code> quantized<br>dtype</th>
      <th><code>kv_cache</code> quantized in <code>pertile</code> mode<br>dtype</th>
      <th><code>kv_cache</code> non-quantized<br>dtype</th>
      <th><code>kv_cache</code> quantized<br>dtype</th>
      <th><code>kv_cache</code> quantized in <code>pertile</code> mode<br>dtype</th>
    </tr>
    <tr>
      <td>token_x</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
    </tr>
    <tr>
      <td>weight_dq</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
    </tr>
    <tr>
      <td>weight_uq_qr</td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
    </tr>
    <tr>
      <td>weight_uk</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
    </tr>
    <tr>
      <td>weight_dkv_kr</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
    </tr>
    <tr>
      <td> rmsnorm_gamma_cq </td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
    </tr>
    <tr>
      <td> rmsnorm_gamma_ckv </td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
    </tr>
    <tr>
      <td> rope_sin </td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
    </tr>
    <tr>
      <td> rope_cos </td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
    </tr>
    <tr>
      <td> kv_cache </td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>int8</td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>int8</td>
    </tr>
    <tr>
      <td> kr_cache </td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
    </tr>
    <tr>
      <td> cache_index </td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
    </tr>
    <tr>
      <td> dequant_scale_x </td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
    </tr>
    <tr>
      <td> dequant_scale_w_dq </td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
    </tr>
    <tr>
      <td> dequant_scale_w_uq_qr </td>
      <td>Not required</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
    </tr>
    <tr>
      <td> dequant_scale_w_dkv_kr </td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
    </tr>
    <tr>
      <td> quant_scale_ckv </td>
      <td>Not required</td>
      <td>Not required</td>
      <td>float</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>float</td>
      <td>Not required</td>
    </tr>
    <tr>
      <td> quant_scale_ckr </td>
      <td>Not required</td>
      <td>Not required</td>
      <td>float</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
    </tr>
    <tr>
      <td> smooth_scales_cq </td>
      <td>Not required</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
    </tr>
    <tr>
      <td> actual_seq_len </td>
      <td>int32</td>
      <td>int32</td>
      <td>int32</td>
      <td>int32</td>
      <td>int32</td>
      <td>int32</td>
      <td>int32</td>
    </tr>
    <tr>
      <td> k_nope_clip_alpha </td>
      <td>Not required</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>float</td>
      <td>Not required</td>
      <td>Not required</td>
      <td>float</td>
    </tr>
    <tr>
      <td> query_out </td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>int8</td>
    </tr>
    <tr>
      <td> query_rope_out </td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
      <td>bfloat16</td>
    </tr>
    <tr>
      <td> dequant_scale_q_nope_out </td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
    </tr>
    <tr>
      <td> query_norm_out </td>
      <td>bfloat16</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
      <td>int8</td>
    </tr>
    <tr>
      <td> dequant_scale_q_norm_out </td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
      <td>float</td>
    </tr>
  </table>
  </div>

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu
    import math
    torch.npu.config.allow_internal_format = True
    # Generate random data and send it to the NPU.
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B, S).to(torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    
    # Call the MlaProlog operator.
    query_mla, query_rope_mla, dequant_scale_q_nope_mla, query_norm_mla, dequant_scale_q_norm_mla = torch_npu.npu_mla_prolog_v3(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    print(query_mla)
    # Expected output of the preceding code sample:
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ..
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
    ```

- Graph mode call

    ```python
    # Configure graph capture
    import torch
    import torch_npu
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
    torch.npu.config.allow_internal_format = True
    
    # Generate data
    B = 8
    He = 7168
    Hcq = 1536
    Hckv = 512
    N = 32
    D = 128
    Dr = 64
    Skv = 1024
    S = 2
    Nkv = 1
    BlockSize = 128
    BlockNum = 64
    token_x = torch.rand(B, S, He, dtype=torch.bfloat16).npu()
    w_dq = torch.rand(He, Hcq, dtype=torch.bfloat16).npu()
    w_dq_cast = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
    w_uq_qr = torch.rand(Hcq, N * (D + Dr), dtype=torch.bfloat16).npu()
    w_uq_qr_cast = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
    w_uk = torch.rand(N, D, Hckv, dtype=torch.bfloat16).npu()
    w_dkv_kr = torch.rand(He, Hckv + Dr, dtype=torch.bfloat16).npu()
    w_dkv_kr_cast = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
    rmsnorm_gamma_cq = torch.rand(Hcq, dtype=torch.bfloat16).npu()
    rmsnorm_gamma_ckv = torch.rand(Hckv, dtype=torch.bfloat16).npu()
    rope_sin = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    rope_cos = torch.rand(B, S, Dr, dtype=torch.bfloat16).npu()
    cache_index = torch.rand(B, S).to(torch.int64).npu()
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_mla_prolog_v3(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        query_mla, query_rope_mla, dequant_scale_q_nope_mla, query_norm_mla, dequant_scale_q_norm_mla = torch_npu.npu_mla_prolog_v3(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, kv_cache, kr_cache, cache_index=cache_index, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
        print("single op output:", query_mla)
        print("graph output:", graph_output)
        
    if __name__ == "__main__":
        MetaInfershape()
    
    # Expected output of the preceding code sample:
    single op output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16)
    
    graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.bfloat16) 
    ```
