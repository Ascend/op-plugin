# torch_npu.npu_mla_prolog_v2

> [!NOTICE]  
> In this API, `kv_cache` and `kr_cache` are updated in place. However, this API is not implemented as a standard in-place operator. Replacing this API with `torch_npu.npu_mla_prolog_v3` is recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |

## Function

- Description: Performs computation during Multi-Head Latent Attention (MLA) preprocessing in inference scenarios. The main computation flow consists of five separate paths.
    - First, input `x` is multiplied by W<sup>DQ</sup> for downsampling and processed through `RmsNorm`. This output is then split into two distinct paths. In the first path, the tensor is multiplied by W<sup>UQ</sup> and W<sup>UK</sup>, followed by two upsampling operations, to yield q<sup>N</sup>.
    - In the second path, the tensor is multiplied by W<sup>QR</sup> and processed through rotary positional encoding (`ROPE`) to yield q<sup>R</sup>.
    - In the third path, input `x` is multiplied by W<sup>DKV</sup> for downsampling, processed through `RmsNorm`, and committed to a cache layer to yield k<sup>C</sup>.
    - In the fourth path, input `x` is multiplied by W<sup>KR</sup>, processed through `ROPE`, and committed to a separate cache layer to yield k<sup>R</sup>.
    - In the fifth path, the output q<sup>N</sup> is processed through dynamic quantization (`DynamicQuant`) to generate quantization parameters.

- Formulas:
    - `RmsNorm` formula:
        $$RmsNorm(x) = \gamma \cdot \frac{x} {\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}}$$

    - Query computation formulas:
        $$c^Q = RmsNorm(x \cdot W^{DQ})$$
        $$q^C = c^Q \cdot W^{UQ}$$
        $$q^N = q^C \cdot W^{UK}$$

    - Query `ROPE` rotary positional encoding formula:

        $$q^R = ROPE(c^Q \cdot W^{QR})$$

    - Key computation formula:

        $$k^C = Cache(RmsNorm(x \cdot W^{DKV}))$$

    - Key `ROPE` rotary positional encoding formula:

        $$k^R = Cache(ROPE(x \cdot W^{KR}))$$

    - Dequantization scaling factor (`dequantScaleQNope`) computation formulas:
        $$\text{dequantScaleQNope} = \frac{\text{RowMax}(\text{abs}(q^N))}{127}$$
        $$q^N = \text{round}(\frac{q^N}{\text{dequantScaleQNope}})$$

## Prototype

```python
torch_npu.npu_mla_prolog_v2(token_x, weight_dq, weight_uq_qr, weight_uk, weight_dkv_kr, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, *, dequant_scale_x=None, dequant_scale_w_dq=None, dequant_scale_w_uq_qr=None, dequant_scale_w_dkv_kr=None, quant_scale_ckv=None, quant_scale_ckr=None, smooth_scales_cq=None, rmsnorm_epsilon_cq=1e-05, rmsnorm_epsilon_ckv=1e-05, cache_mode="PA_BSND") -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## Parameters

- **`token_x`** (`Tensor`): Required. `x` in the formulas. This parameter must be 2D or 3D, with shape `(T, He)` or `(B, S, He)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`weight_dq`** (`Tensor`): Required. Downsampling weight matrix for query computation, W<sup>DQ</sup> in the formulas. This parameter must be 2D with shape `(He, Hcq)`. The data type can be `bfloat16` or `int8`. The data layout can be `FRACTAL_NZ`. The layout can be converted from `ND` by using `torch_npu.npu_format_cast`.
- `weight_uq_qr` (`Tensor`): Required. Combined upsampling weight matrix and positional encoding weight matrix for query computation, W<sup>UQ</sup> and W<sup>QR</sup> in the formulas. This parameter must be 2D with shape `(Hcq, N * (D + Dr))`. The data type can be `bfloat16` or `int8`. The data layout can be FRACTAL\_NZ.
- **`weight_uk`** (`Tensor`): Required. Upsampling weight matrix for key computation, corresponding to W<sup>UK</sup> in the formulas. This parameter must be 3D with shape `(N, D, Hckv)`. The data type can be `bfloat16`. The data layout can be ND.
- **`weight_dkv_kr`** (`Tensor`): Required. Combined downsampling weight matrix and positional encoding weight matrix for key computation, W<sup>DKV</sup> and W<sup>KR</sup> in the formulas. This parameter must be 2D with shape `(He, Hckv + Dr)`. The data type can be `bfloat16` or `int8`. The data layout can be FRACTAL\_NZ.
- **`rmsnorm_gamma_cq`** (`Tensor`): Required. The $\gamma$ parameter in the `RmsNorm` formula for computing c<sup>Q</sup>. This parameter must be 1D with shape `(Hcq,)`. The data type can be `bfloat16`. The data layout can be ND.
- **`rmsnorm_gamma_ckv`** (`Tensor`): Required. The $\gamma$ parameter in the `RmsNorm` formula for computing c<sup>KV</sup>. This parameter must be 1D with shape `(Hckv,)`. The data type can be `bfloat16`. The data layout can be ND.
- **`rope_sin`** (`Tensor`): Required. Sine parameter matrix for rotary positional encoding (`ROPE`) computation. This parameter must be 2D or 3D, with shape `(T, Dr)` or `(B, S, Dr)`. The data type can be `bfloat16`. The data layout can be ND.
- **`rope_cos`** (`Tensor`): Required. Cosine parameter matrix for rotary positional encoding (`ROPE`) computation. This parameter must be 2D or 3D, with shape `(T, Dr)` or `(B, S, Dr)`. The data type can be `bfloat16`. The data layout can be ND.
- **`cache_index`** (`Tensor`): Required. Linear page slot index for writing `kv_cache` and `kr_cache`. This parameter must 1D or 2D, with shape `(T)` or `(B, S)`. The data type can be `int64`. The data layout can be ND. The value range of each element is `[0, L)`. Here, `L` represents the product of the lengths of the 0th and 1st dimensions of `kv_cache`. This product must be identical to the product of the first two dimensions of `kr_cache`.
- **`kv_cache`** (`Tensor`): Required. Cache tensor for key and value indexing. This parameter must be 4D with shape `(BlockNum, BlockSize, Nkv, Hckv)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`kr_cache`** (`Tensor`): Required. Cache tensor for key rotary positional encoding. This parameter must be 4D with shape `(BlockNum, BlockSize, Nkv, Dr)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`dequant_scale_x`** (`Tensor`): Optional. Dequantization scale parameter used after downsampling when the data type of input `token_x` is `int8`. The quantization mode for `token_x` is `pertoken`. This parameter must be 2D with shape `(T, 1)` or `(BS, 1)`. The data type can be `float`. The data layout can be ND.
- **`dequant_scale_w_dq`** (`Tensor`): Optional. Dequantization scale parameter used after downsampling when the input `token_x` is set to `int8`. The quantization mode for `token_x` is `perchannel`. This parameter must be 2D with shape `(1, Hcq)`. The data type can be `float`. The data layout can be ND.
- **`dequant_scale_w_uq_qr`** (`Tensor`): Optional. Dequantization scale parameter used after the `MatmulQcQr` matrix multiplication operation. The quantization mode is `perchannel`. This parameter must be 2D with shape `(1, N * (D + Dr))`. The data type can be `float`. The data layout can be ND.
- **`dequant_scale_w_dkv_kr`** (`Tensor`): Optional. Dequantization scale parameter used after the `MatmulQcQr` matrix multiplication operation. The quantization mode is `perchannel`. This parameter must be 2D with shape `(1, Hckv + Dr)`. The data type can be `float`. The data layout can be ND.
- **`quant_scale_ckv`** (`Tensor`): Optional. Quantization scale parameter used when writing data to `kv_cache_out`. This parameter must be 2D with shape `(1, Hckv)`. The data type can be `float`. The data layout can be ND.
- **`quant_scale_ckr`** (`Tensor`): Optional. Quantization scale parameter used when writing data to `kr_cache_out`. This parameter must be 2D with shape `(1, Dr)`. The data type can be `float`. The data layout can be ND.
- **`smooth_scales_cq`** (`Tensor`): Optional. Scaling factors used for dynamic quantization of the `RmsNormCq` output. This parameter must be 2D with shape `(1, Hcq)`. The data type can be `float`. The data layout can be ND.
- **`rmsnorm_epsilon_cq`** (`float`): Optional. The $\epsilon$ parameter in the `RmsNorm` formula for computing c<sup>Q</sup>. The default value is `1e-05`.
- **`rmsnorm_epsilon_ckv`** (`float`): Optional. The $\epsilon$ parameter in the `RmsNorm` formula for computing c<sup>KV</sup>. The default value is `1e-05`.
- **`cache_mode`** (`str`): Optional. Layout mode of `kv_cache`. The supported values are `"PA_BSND"` and `"PA_NZ"`. The default value is `"PA_BSND"`.

## Return Values

- **`query`** (`Tensor`): Query output tensor, corresponding to q<sup>N</sup> in the formulas. This parameter must be 3D or 4D with shape `(T, N, Hckv)` or `(B, S, N, Hckv)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`query_rope`** (`Tensor`): Output tensor for query positional encoding, q<sup>R</sup> in the formulas. This parameter must be 3D or 4D with shape `(T, N, Dr)` or `(B, S, N, Dr)`. The data type can be `bfloat16`. The data layout can be ND.
- **`kv_cache_out`** (`Tensor`): Tensor written to `kv_cache` through an in-place update, k<sup>C</sup> in the formulas. This parameter must be 4D with shape `(BlockNum, BlockSize, Nkv, Hckv)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`kr_cache_out`** (`Tensor`): Tensor written to `kr_cache` through an in-place update, corresponding to k<sup>R</sup> in the formulas. This parameter must be 4D with shape `(BlockNum, BlockSize, Nkv, Dr)`. The data type can be `bfloat16` or `int8`. The data layout can be ND.
- **`dequant_scale_q_nope`** (`Tensor`): Dequantization scale parameter for the query output tensor. This parameter must be 1D or 3D. In full KV cache quantization scenarios, the shape is `(T, N, 1)` or `(B * S, N, 1)`. In other scenarios, the shape is `(1,)`. The data type can be `float`. The data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- Field definitions for shape formats:
    - `B`: batch size, representing the input sample batch size. The value ranges from 0 to 65536.
    - `S`: seq-Length, representing the input sample sequence length. The value ranges from 0 to 16.
    - `He`: Head-Size, representing the hidden layer size. The value must be `7168`.

    - `Hcq`: dimension of the low-rank query matrix. The value must be `1536`.
    - `N`: head-Num, representing the attention head count. Supported values are `1`, `2`, `4`, `8`, `16`, `32`, `64`, and `128`.

    - `Hckv`: dimension of the low-rank key and value matrix. The value must be `512`.
    - `D`: dimension of query and key without positional encoding. The value must be `128`.
    - `Dr`: dimension of query and key positional encoding. The value must be `64`.
    - `Nkv`: attention head count for key and value. The value must be `1`.
    - `BlockNum`: Number of blocks in the PagedAttention scenario. That is, the length of the 0th dimension of `kv_cache` and `kr_cache`. Given batch size `B`, key and value sequence length `Skv`, and block size `BlockSize`, the condition $\text{BlockNum} \ge \lceil B * Skv/\text{BlockSize} \rceil$ must be satisfied. Here, `Skv` can be 0.
    - `BlockSize`: block size in the PagedAttention scenario. Supported values are `16` and `128`.
    - `T`: size after the fusion of the `B` and `S` axes. The value ranges from 0 to 1048576.

- Shape constraints:
    - When `token_x` uses `B` and `S` axis fusion, the layout is `(T, He)`. In this configuration, the shape of `rope_sin` and `rope_cos` must be `(T, Dr)`. The shape of `cache_index` must be `(T,)`. The shape of `dequant_scale_x` must be `(T, 1)`. The shape of `query` must be `(T, N, Hckv)`. The shape of `query_rope` must be `(T, N, Dr)`. In full KV cache quantization scenarios, the shape of `dequant_scale_q_nope` must be `(T, N, 1)`. In other scenarios, the shape of `dequant_scale_q_nope` must be `(1,)`.
    - When `token_x` does not use `B` and `S` axis fusion, the layout is `(B, S, He)`. In this configuration, the shape of `rope_sin` and `rope_cos` must be `(B, S, Dr)`. The shape of `cache_index` must be `(B, S)`. The shape of `dequant_scale_x` must be `(B * S, 1)`. The shape of `query` must be `(B, S, N, Hckv)`. The shape of `query_rope` must be `(B, S, N, Dr)`. In full KV cache quantization scenarios, the shape of `dequant_scale_q_nope` must be `(B * S, N, 1)`. In other scenarios, the shape of `dequant_scale_q_nope` must be `(1,)`.
    - One or more of `B`, `S`, `T`, and `Skv` can be `0`. That is, inputs whose shapes depend on these values can be empty tensors. Other inputs must not be empty tensors.
        - If `B`, `S`, or `T` is set to `0`, `query`, `query_rope`, and `dequant_scale_q_nope` output empty tensors. `kv_cache`, `kr_cache`, `kv_cache_out`, and `kr_cache_out` are not updated.
        - If `Skv` is set to `0`, `query`, `query_rope`, and `dequant_scale_q_nope` are computed normally. `kv_cache`, `kr_cache`, `kv_cache_out`, and `kr_cache_out` are not updated and output empty tensors.

- This operator supports the following scenarios.
    <a name="en-us_topic_0000002313328922_table664817810310"></a>
    <table><thead align="left"><tr id="en-us_topic_0000002313328922_row9649788313"><th class="cellrowborder" colspan="2" valign="top" id="mcps1.1.4.1.1"><p id="en-us_topic_0000002313328922_p14649381739"><a name="en-us_topic_0000002313328922_p14649381739"></a><a name="en-us_topic_0000002313328922_p14649381739"></a>Scenario</p>
    </th>
    <th class="cellrowborder" valign="top" id="mcps1.1.4.1.2"><p id="en-us_topic_0000002313328922_p1649781312"><a name="en-us_topic_0000002313328922_p1649781312"></a><a name="en-us_topic_0000002313328922_p1649781312"></a>Description</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="en-us_topic_0000002313328922_row36491488316"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p16649987312"><a name="en-us_topic_0000002313328922_p16649987312"></a><a name="en-us_topic_0000002313328922_p16649987312"></a>Non-quantization</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.1.4.1.2 "><p id="en-us_topic_0000002313328922_p96491986311"><a name="en-us_topic_0000002313328922_p96491986311"></a><a name="en-us_topic_0000002313328922_p96491986311"></a>Inputs: All input parameters are non-quantized data.<br>Outputs: All output parameters are non-quantized data.</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row230913715311"><td class="cellrowborder" rowspan="2" valign="top" width="8.780000000000001%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p330973715314"><a name="en-us_topic_0000002313328922_p330973715314"></a><a name="en-us_topic_0000002313328922_p330973715314"></a>Partial quantization</p>
    </td>
    <td class="cellrowborder" valign="top" width="18.18%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p9145842144313"><a name="en-us_topic_0000002313328922_p9145842144313"></a><a name="en-us_topic_0000002313328922_p9145842144313"></a><code>kv_cache</code> non-quantized</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.04%" headers="mcps1.1.4.1.2 "><p id="en-us_topic_0000002313328922_p18277152811188"><a name="en-us_topic_0000002313328922_p18277152811188"></a><a name="en-us_topic_0000002313328922_p18277152811188"></a>Inputs: <code>weight_uq_qr</code> is <code>per-token</code> quantized data. Other inputs are non-quantized data.</p>
    <p id="en-us_topic_0000002313328922_p8284162318619"><a name="en-us_topic_0000002313328922_p8284162318619"></a><a name="en-us_topic_0000002313328922_p8284162318619"></a>Outputs: All output parameters are non-quantized data.</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row6013117434"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p714610427438"><a name="en-us_topic_0000002313328922_p714610427438"></a><a name="en-us_topic_0000002313328922_p714610427438"></a><code>kv_cache</code> quantized</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p4752029876"><a name="en-us_topic_0000002313328922_p4752029876"></a><a name="en-us_topic_0000002313328922_p4752029876"></a>Inputs: <code>weight_uq_qr</code> is <code>pertoken</code> quantized data. <code>kv_cache</code> and <code>kr_cache</code> are <code>perchannel</code> quantized data. Other inputs are non-quantized data.</p>
    <p id="en-us_topic_0000002313328922_p4262738101812"><a name="en-us_topic_0000002313328922_p4262738101812"></a><a name="en-us_topic_0000002313328922_p4262738101812"></a>Outputs: <code>kv_cache_out</code> and <code>kr_cache_out</code> are <code>perchannel</code> quantized data. Other outputs are non-quantized data.</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row3649881538"><td class="cellrowborder" rowspan="2" valign="top" width="8.780000000000001%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p1764928538"><a name="en-us_topic_0000002313328922_p1764928538"></a><a name="en-us_topic_0000002313328922_p1764928538"></a>Full quantization</p>
    </td>
    <td class="cellrowborder" valign="top" width="18.18%" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p71465427433"><a name="en-us_topic_0000002313328922_p71465427433"></a><a name="en-us_topic_0000002313328922_p71465427433"></a><code>kv_cache</code> non-quantized</p>
    </td>
    <td class="cellrowborder" valign="top" width="73.04%" headers="mcps1.1.4.1.2 "><p id="en-us_topic_0000002313328922_p20649887310"><a name="en-us_topic_0000002313328922_p20649887310"></a><a name="en-us_topic_0000002313328922_p20649887310"></a>Inputs: <code>token_x</code> is <code>pertoken</code> quantized data. <code>weight_dq</code>, <code>weight_uq_qr</code>, and <code>weight_dkv_kr</code> are <code>perchannel</code> quantized data. Other inputs are non-quantized data.</p>
    <p id="en-us_topic_0000002313328922_p940213169195"><a name="en-us_topic_0000002313328922_p940213169195"></a><a name="en-us_topic_0000002313328922_p940213169195"></a>Outputs: All output parameters are non-quantized data.</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row576033534319"><td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p1714664220437"><a name="en-us_topic_0000002313328922_p1714664220437"></a><a name="en-us_topic_0000002313328922_p1714664220437"></a><code>kv_cache</code> quantized</p>
    </td>
    <td class="cellrowborder" valign="top" headers="mcps1.1.4.1.1 "><p id="en-us_topic_0000002313328922_p14760153518431"><a name="en-us_topic_0000002313328922_p14760153518431"></a><a name="en-us_topic_0000002313328922_p14760153518431"></a>Inputs: <code>token_x</code> is <code>pertoken</code> quantized data. <code>weight_dq</code>, <code>weight_uq_qr</code>, and <code>weight_dkv_kr</code> are <code>perchannel</code> quantized data. Other inputs are non-quantized data.</p>
    <p id="en-us_topic_0000002313328922_p1280815811110"><a name="en-us_topic_0000002313328922_p1280815811110"></a><a name="en-us_topic_0000002313328922_p1280815811110"></a>Outputs: <code>query</code> is <code>pertoken_head</code> dynamic quantized data. <code>kv_cache_out</code> is <code>pertensor</code> quantized data. Other outputs are non-quantized data.</p>
    </td>
    </tr>
    </tbody>
    </table>

- In different quantization scenarios, the data types (dtypes) and shapes of parameters must satisfy the following conditions.

    <a name="en-us_topic_0000002313328922_table1311951423117"></a>
    <table><tbody><tr id="en-us_topic_0000002313328922_row510181463115"><td class="cellrowborder" rowspan="3" valign="top"><p id="en-us_topic_0000002313328922_p21013144313"><a name="en-us_topic_0000002313328922_p21013144313"></a><a name="en-us_topic_0000002313328922_p21013144313"></a><strong id="en-us_topic_0000002313328922_b1423515521358"><a name="en-us_topic_0000002313328922_b1423515521358"></a><a name="en-us_topic_0000002313328922_b1423515521358"></a>Parameter</strong></p>
    </td>
    <td class="cellrowborder" rowspan="2" colspan="2" valign="top"><p id="en-us_topic_0000002313328922_p201012147314"><a name="en-us_topic_0000002313328922_p201012147314"></a><a name="en-us_topic_0000002313328922_p201012147314"></a><strong id="en-us_topic_0000002313328922_b1824916521457"><a name="en-us_topic_0000002313328922_b1824916521457"></a><a name="en-us_topic_0000002313328922_b1824916521457"></a>Non-quantization</strong></p>
    </td>
    <td class="cellrowborder" colspan="4" valign="top"><p id="en-us_topic_0000002313328922_p1810121413112"><a name="en-us_topic_0000002313328922_p1810121413112"></a><a name="en-us_topic_0000002313328922_p1810121413112"></a><strong id="en-us_topic_0000002313328922_b122631152959"><a name="en-us_topic_0000002313328922_b122631152959"></a><a name="en-us_topic_0000002313328922_b122631152959"></a>Partial Quantization</strong></p>
    </td>
    <td class="cellrowborder" colspan="4" valign="top"><p id="en-us_topic_0000002313328922_p10101131423113"><a name="en-us_topic_0000002313328922_p10101131423113"></a><a name="en-us_topic_0000002313328922_p10101131423113"></a><strong id="en-us_topic_0000002313328922_b172646522059"><a name="en-us_topic_0000002313328922_b172646522059"></a><a name="en-us_topic_0000002313328922_b172646522059"></a>Full Quantization</strong></p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row6101181419318"><td class="cellrowborder" colspan="2" valign="top"><p id="en-us_topic_0000002313328922_p1101101414315"><a name="en-us_topic_0000002313328922_p1101101414315"></a><a name="en-us_topic_0000002313328922_p1101101414315"></a><strong id="en-us_topic_0000002313328922_b42651521553"><a name="en-us_topic_0000002313328922_b42651521553"></a><a name="en-us_topic_0000002313328922_b42651521553"></a><code>kv_cache</code> Non-Quantized</strong></p>
    </td>
    <td class="cellrowborder" colspan="2" valign="top"><p id="en-us_topic_0000002313328922_p01012143316"><a name="en-us_topic_0000002313328922_p01012143316"></a><a name="en-us_topic_0000002313328922_p01012143316"></a><strong id="en-us_topic_0000002313328922_b1226655212514"><a name="en-us_topic_0000002313328922_b1226655212514"></a><a name="en-us_topic_0000002313328922_b1226655212514"></a><code>kv_cache</code> Quantized</strong></p>
    </td>
    <td class="cellrowborder" colspan="2" valign="top"><p id="en-us_topic_0000002313328922_p3101101493112"><a name="en-us_topic_0000002313328922_p3101101493112"></a><a name="en-us_topic_0000002313328922_p3101101493112"></a><strong id="en-us_topic_0000002313328922_b1526718527511"><a name="en-us_topic_0000002313328922_b1526718527511"></a><a name="en-us_topic_0000002313328922_b1526718527511"></a><code>kv_cache</code> Non-Quantized</strong></p>
    </td>
    <td class="cellrowborder" colspan="2" valign="top"><p id="en-us_topic_0000002313328922_p41011614203115"><a name="en-us_topic_0000002313328922_p41011614203115"></a><a name="en-us_topic_0000002313328922_p41011614203115"></a><strong id="en-us_topic_0000002313328922_b826811521456"><a name="en-us_topic_0000002313328922_b826811521456"></a><a name="en-us_topic_0000002313328922_b826811521456"></a><code>kv_cache</code> Quantized</strong></p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row210212145314"><td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p71014149318"><a name="en-us_topic_0000002313328922_p71014149318"></a><a name="en-us_topic_0000002313328922_p71014149318"></a><strong id="en-us_topic_0000002313328922_b152701452558"><a name="en-us_topic_0000002313328922_b152701452558"></a><a name="en-us_topic_0000002313328922_b152701452558"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p191011214113115"><a name="en-us_topic_0000002313328922_p191011214113115"></a><a name="en-us_topic_0000002313328922_p191011214113115"></a><strong id="en-us_topic_0000002313328922_b1127115521059"><a name="en-us_topic_0000002313328922_b1127115521059"></a><a name="en-us_topic_0000002313328922_b1127115521059"></a>shape</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p210171493118"><a name="en-us_topic_0000002313328922_p210171493118"></a><a name="en-us_topic_0000002313328922_p210171493118"></a><strong id="en-us_topic_0000002313328922_b52724521557"><a name="en-us_topic_0000002313328922_b52724521557"></a><a name="en-us_topic_0000002313328922_b52724521557"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p15101114203119"><a name="en-us_topic_0000002313328922_p15101114203119"></a><a name="en-us_topic_0000002313328922_p15101114203119"></a><strong id="en-us_topic_0000002313328922_b32734521655"><a name="en-us_topic_0000002313328922_b32734521655"></a><a name="en-us_topic_0000002313328922_b32734521655"></a>shape</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p1010113147310"><a name="en-us_topic_0000002313328922_p1010113147310"></a><a name="en-us_topic_0000002313328922_p1010113147310"></a><strong id="en-us_topic_0000002313328922_b1527420527515"><a name="en-us_topic_0000002313328922_b1527420527515"></a><a name="en-us_topic_0000002313328922_b1527420527515"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p161011014123117"><a name="en-us_topic_0000002313328922_p161011014123117"></a><a name="en-us_topic_0000002313328922_p161011014123117"></a><strong id="en-us_topic_0000002313328922_b32751752355"><a name="en-us_topic_0000002313328922_b32751752355"></a><a name="en-us_topic_0000002313328922_b32751752355"></a>shape</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p610216144319"><a name="en-us_topic_0000002313328922_p610216144319"></a><a name="en-us_topic_0000002313328922_p610216144319"></a><strong id="en-us_topic_0000002313328922_b192768521753"><a name="en-us_topic_0000002313328922_b192768521753"></a><a name="en-us_topic_0000002313328922_b192768521753"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p510211149315"><a name="en-us_topic_0000002313328922_p510211149315"></a><a name="en-us_topic_0000002313328922_p510211149315"></a><strong id="en-us_topic_0000002313328922_b192771952258"><a name="en-us_topic_0000002313328922_b192771952258"></a><a name="en-us_topic_0000002313328922_b192771952258"></a>shape</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p1410221423116"><a name="en-us_topic_0000002313328922_p1410221423116"></a><a name="en-us_topic_0000002313328922_p1410221423116"></a><strong id="en-us_topic_0000002313328922_b12278105218513"><a name="en-us_topic_0000002313328922_b12278105218513"></a><a name="en-us_topic_0000002313328922_b12278105218513"></a>dtype</strong></p>
    </td>
    <td class="cellrowborder" valign="top"><p id="en-us_topic_0000002313328922_p5102814163116"><a name="en-us_topic_0000002313328922_p5102814163116"></a><a name="en-us_topic_0000002313328922_p5102814163116"></a><strong id="en-us_topic_0000002313328922_b132796521952"><a name="en-us_topic_0000002313328922_b132796521952"></a><a name="en-us_topic_0000002313328922_b132796521952"></a>shape</strong></p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row19103514123115"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1310271493114"><a name="en-us_topic_0000002313328922_p1310271493114"></a><a name="en-us_topic_0000002313328922_p1310271493114"></a>token_x</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1102141414313"><a name="en-us_topic_0000002313328922_p1102141414313"></a><a name="en-us_topic_0000002313328922_p1102141414313"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="en-us_topic_0000002313328922_ul17461162913811"></a><a name="en-us_topic_0000002313328922_ul17461162913811"></a><ul id="en-us_topic_0000002313328922_ul17461162913811"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p8102514163119"><a name="en-us_topic_0000002313328922_p8102514163119"></a><a name="en-us_topic_0000002313328922_p8102514163119"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul472585016425"></a><a name="en-us_topic_0000002313328922_ul472585016425"></a><ul id="en-us_topic_0000002313328922_ul472585016425"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p17102314193112"><a name="en-us_topic_0000002313328922_p17102314193112"></a><a name="en-us_topic_0000002313328922_p17102314193112"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="en-us_topic_0000002313328922_ul53761545381"></a><a name="en-us_topic_0000002313328922_ul53761545381"></a><ul id="en-us_topic_0000002313328922_ul53761545381"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1210291473118"><a name="en-us_topic_0000002313328922_p1210291473118"></a><a name="en-us_topic_0000002313328922_p1210291473118"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul1124455434211"></a><a name="en-us_topic_0000002313328922_ul1124455434211"></a><ul id="en-us_topic_0000002313328922_ul1124455434211"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p010219141312"><a name="en-us_topic_0000002313328922_p010219141312"></a><a name="en-us_topic_0000002313328922_p010219141312"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="en-us_topic_0000002313328922_ul7208162853214"></a><a name="en-us_topic_0000002313328922_ul7208162853214"></a><ul id="en-us_topic_0000002313328922_ul7208162853214"><li>(B,S,He)</li><li>(T,He)</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row310371413111"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p191032141314"><a name="en-us_topic_0000002313328922_p191032141314"></a><a name="en-us_topic_0000002313328922_p191032141314"></a>weight_dq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p31032147310"><a name="en-us_topic_0000002313328922_p31032147310"></a><a name="en-us_topic_0000002313328922_p31032147310"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p171031714193111"><a name="en-us_topic_0000002313328922_p171031714193111"></a><a name="en-us_topic_0000002313328922_p171031714193111"></a>(He,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p110361423111"><a name="en-us_topic_0000002313328922_p110361423111"></a><a name="en-us_topic_0000002313328922_p110361423111"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p101038148313"><a name="en-us_topic_0000002313328922_p101038148313"></a><a name="en-us_topic_0000002313328922_p101038148313"></a>(He,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p161031714153111"><a name="en-us_topic_0000002313328922_p161031714153111"></a><a name="en-us_topic_0000002313328922_p161031714153111"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p4103191483114"><a name="en-us_topic_0000002313328922_p4103191483114"></a><a name="en-us_topic_0000002313328922_p4103191483114"></a>He,Hcq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p201031114103115"><a name="en-us_topic_0000002313328922_p201031114103115"></a><a name="en-us_topic_0000002313328922_p201031114103115"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p15103121413119"><a name="en-us_topic_0000002313328922_p15103121413119"></a><a name="en-us_topic_0000002313328922_p15103121413119"></a>(He,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p51032014193110"><a name="en-us_topic_0000002313328922_p51032014193110"></a><a name="en-us_topic_0000002313328922_p51032014193110"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p51031414143117"><a name="en-us_topic_0000002313328922_p51031414143117"></a><a name="en-us_topic_0000002313328922_p51031414143117"></a>(He,Hcq)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row161042141311"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p910314141313"><a name="en-us_topic_0000002313328922_p910314141313"></a><a name="en-us_topic_0000002313328922_p910314141313"></a>weight_uq_qr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p19103151418313"><a name="en-us_topic_0000002313328922_p19103151418313"></a><a name="en-us_topic_0000002313328922_p19103151418313"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1610321416310"><a name="en-us_topic_0000002313328922_p1610321416310"></a><a name="en-us_topic_0000002313328922_p1610321416310"></a>(Hcq,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p6103131473113"><a name="en-us_topic_0000002313328922_p6103131473113"></a><a name="en-us_topic_0000002313328922_p6103131473113"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p10104191410311"><a name="en-us_topic_0000002313328922_p10104191410311"></a><a name="en-us_topic_0000002313328922_p10104191410311"></a>(Hcq,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p9104171413118"><a name="en-us_topic_0000002313328922_p9104171413118"></a><a name="en-us_topic_0000002313328922_p9104171413118"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p189211613184413"><a name="en-us_topic_0000002313328922_p189211613184413"></a><a name="en-us_topic_0000002313328922_p189211613184413"></a>(Hcq,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p0104101413314"><a name="en-us_topic_0000002313328922_p0104101413314"></a><a name="en-us_topic_0000002313328922_p0104101413314"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p17104214133117"><a name="en-us_topic_0000002313328922_p17104214133117"></a><a name="en-us_topic_0000002313328922_p17104214133117"></a>(Hcq,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p51041914123119"><a name="en-us_topic_0000002313328922_p51041914123119"></a><a name="en-us_topic_0000002313328922_p51041914123119"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p1910410144316"><a name="en-us_topic_0000002313328922_p1910410144316"></a><a name="en-us_topic_0000002313328922_p1910410144316"></a>(Hcq,N*(D+Dr))</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row3105131493117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1610421413118"><a name="en-us_topic_0000002313328922_p1610421413118"></a><a name="en-us_topic_0000002313328922_p1610421413118"></a>weight_uk</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p710491419315"><a name="en-us_topic_0000002313328922_p710491419315"></a><a name="en-us_topic_0000002313328922_p710491419315"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p14104121453113"><a name="en-us_topic_0000002313328922_p14104121453113"></a><a name="en-us_topic_0000002313328922_p14104121453113"></a>(N,D,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p41041414163116"><a name="en-us_topic_0000002313328922_p41041414163116"></a><a name="en-us_topic_0000002313328922_p41041414163116"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p17295153120442"><a name="en-us_topic_0000002313328922_p17295153120442"></a><a name="en-us_topic_0000002313328922_p17295153120442"></a>(N,D,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p91041114113114"><a name="en-us_topic_0000002313328922_p91041114113114"></a><a name="en-us_topic_0000002313328922_p91041114113114"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p299953374419"><a name="en-us_topic_0000002313328922_p299953374419"></a><a name="en-us_topic_0000002313328922_p299953374419"></a>(N,D,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p121041145312"><a name="en-us_topic_0000002313328922_p121041145312"></a><a name="en-us_topic_0000002313328922_p121041145312"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p25501735154412"><a name="en-us_topic_0000002313328922_p25501735154412"></a><a name="en-us_topic_0000002313328922_p25501735154412"></a>(N,D,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p410441493120"><a name="en-us_topic_0000002313328922_p410441493120"></a><a name="en-us_topic_0000002313328922_p410441493120"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p14409163815449"><a name="en-us_topic_0000002313328922_p14409163815449"></a><a name="en-us_topic_0000002313328922_p14409163815449"></a>(N,D,Hckv)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row510581423111"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p6105414123114"><a name="en-us_topic_0000002313328922_p6105414123114"></a><a name="en-us_topic_0000002313328922_p6105414123114"></a>weight_dkv_kr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p13105014123115"><a name="en-us_topic_0000002313328922_p13105014123115"></a><a name="en-us_topic_0000002313328922_p13105014123115"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p41059141311"><a name="en-us_topic_0000002313328922_p41059141311"></a><a name="en-us_topic_0000002313328922_p41059141311"></a>(He,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p161051514173116"><a name="en-us_topic_0000002313328922_p161051514173116"></a><a name="en-us_topic_0000002313328922_p161051514173116"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p4135250144412"><a name="en-us_topic_0000002313328922_p4135250144412"></a><a name="en-us_topic_0000002313328922_p4135250144412"></a>(He,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1910571416319"><a name="en-us_topic_0000002313328922_p1910571416319"></a><a name="en-us_topic_0000002313328922_p1910571416319"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p9441185294416"><a name="en-us_topic_0000002313328922_p9441185294416"></a><a name="en-us_topic_0000002313328922_p9441185294416"></a>(He,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p191051614143117"><a name="en-us_topic_0000002313328922_p191051614143117"></a><a name="en-us_topic_0000002313328922_p191051614143117"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p12951554154420"><a name="en-us_topic_0000002313328922_p12951554154420"></a><a name="en-us_topic_0000002313328922_p12951554154420"></a>(He,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p121058149314"><a name="en-us_topic_0000002313328922_p121058149314"></a><a name="en-us_topic_0000002313328922_p121058149314"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p52995561441"><a name="en-us_topic_0000002313328922_p52995561441"></a><a name="en-us_topic_0000002313328922_p52995561441"></a>(He,Hckv+Dr)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row10106161463117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1510561410319"><a name="en-us_topic_0000002313328922_p1510561410319"></a><a name="en-us_topic_0000002313328922_p1510561410319"></a>rmsnorm_gamma_cq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p201051914193111"><a name="en-us_topic_0000002313328922_p201051914193111"></a><a name="en-us_topic_0000002313328922_p201051914193111"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p9105514123113"><a name="en-us_topic_0000002313328922_p9105514123113"></a><a name="en-us_topic_0000002313328922_p9105514123113"></a>(Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p1910551410318"><a name="en-us_topic_0000002313328922_p1910551410318"></a><a name="en-us_topic_0000002313328922_p1910551410318"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p20412171411451"><a name="en-us_topic_0000002313328922_p20412171411451"></a><a name="en-us_topic_0000002313328922_p20412171411451"></a>(Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p11106914183110"><a name="en-us_topic_0000002313328922_p11106914183110"></a><a name="en-us_topic_0000002313328922_p11106914183110"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p996811614518"><a name="en-us_topic_0000002313328922_p996811614518"></a><a name="en-us_topic_0000002313328922_p996811614518"></a>(Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p51061814193115"><a name="en-us_topic_0000002313328922_p51061814193115"></a><a name="en-us_topic_0000002313328922_p51061814193115"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p3552171824517"><a name="en-us_topic_0000002313328922_p3552171824517"></a><a name="en-us_topic_0000002313328922_p3552171824517"></a>(Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p1110681414313"><a name="en-us_topic_0000002313328922_p1110681414313"></a><a name="en-us_topic_0000002313328922_p1110681414313"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p14312204455"><a name="en-us_topic_0000002313328922_p14312204455"></a><a name="en-us_topic_0000002313328922_p14312204455"></a>(Hcq)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row8107151423115"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1110616141319"><a name="en-us_topic_0000002313328922_p1110616141319"></a><a name="en-us_topic_0000002313328922_p1110616141319"></a>rmsnorm_gamma_ckv</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1110631463118"><a name="en-us_topic_0000002313328922_p1110631463118"></a><a name="en-us_topic_0000002313328922_p1110631463118"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p15106111423114"><a name="en-us_topic_0000002313328922_p15106111423114"></a><a name="en-us_topic_0000002313328922_p15106111423114"></a>(Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p71064146316"><a name="en-us_topic_0000002313328922_p71064146316"></a><a name="en-us_topic_0000002313328922_p71064146316"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p1110619142318"><a name="en-us_topic_0000002313328922_p1110619142318"></a><a name="en-us_topic_0000002313328922_p1110619142318"></a>(Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p151061114153117"><a name="en-us_topic_0000002313328922_p151061114153117"></a><a name="en-us_topic_0000002313328922_p151061114153117"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p6106181416316"><a name="en-us_topic_0000002313328922_p6106181416316"></a><a name="en-us_topic_0000002313328922_p6106181416316"></a>(Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p5106714193110"><a name="en-us_topic_0000002313328922_p5106714193110"></a><a name="en-us_topic_0000002313328922_p5106714193110"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p61061314183118"><a name="en-us_topic_0000002313328922_p61061314183118"></a><a name="en-us_topic_0000002313328922_p61061314183118"></a>(Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p141063148312"><a name="en-us_topic_0000002313328922_p141063148312"></a><a name="en-us_topic_0000002313328922_p141063148312"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p1310611418319"><a name="en-us_topic_0000002313328922_p1310611418319"></a><a name="en-us_topic_0000002313328922_p1310611418319"></a>(Hckv)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row1107191463116"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p810731415313"><a name="en-us_topic_0000002313328922_p810731415313"></a><a name="en-us_topic_0000002313328922_p810731415313"></a>rope_sin</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1810781411315"><a name="en-us_topic_0000002313328922_p1810781411315"></a><a name="en-us_topic_0000002313328922_p1810781411315"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="en-us_topic_0000002313328922_ul6327114983818"></a><a name="en-us_topic_0000002313328922_ul6327114983818"></a><ul id="en-us_topic_0000002313328922_ul6327114983818"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p181071814133111"><a name="en-us_topic_0000002313328922_p181071814133111"></a><a name="en-us_topic_0000002313328922_p181071814133111"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul254803553916"></a><a name="en-us_topic_0000002313328922_ul254803553916"></a><ul id="en-us_topic_0000002313328922_ul254803553916"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p610721403111"><a name="en-us_topic_0000002313328922_p610721403111"></a><a name="en-us_topic_0000002313328922_p610721403111"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="en-us_topic_0000002313328922_ul16821173773914"></a><a name="en-us_topic_0000002313328922_ul16821173773914"></a><ul id="en-us_topic_0000002313328922_ul16821173773914"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p7107201413315"><a name="en-us_topic_0000002313328922_p7107201413315"></a><a name="en-us_topic_0000002313328922_p7107201413315"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul199164010395"></a><a name="en-us_topic_0000002313328922_ul199164010395"></a><ul id="en-us_topic_0000002313328922_ul199164010395"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p1310701410316"><a name="en-us_topic_0000002313328922_p1310701410316"></a><a name="en-us_topic_0000002313328922_p1310701410316"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="en-us_topic_0000002313328922_ul241342173912"></a><a name="en-us_topic_0000002313328922_ul241342173912"></a><ul id="en-us_topic_0000002313328922_ul241342173912"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row16108131453117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p16107114103120"><a name="en-us_topic_0000002313328922_p16107114103120"></a><a name="en-us_topic_0000002313328922_p16107114103120"></a>rope_cos</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p510781414315"><a name="en-us_topic_0000002313328922_p510781414315"></a><a name="en-us_topic_0000002313328922_p510781414315"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="en-us_topic_0000002313328922_ul6804164519391"></a><a name="en-us_topic_0000002313328922_ul6804164519391"></a><ul id="en-us_topic_0000002313328922_ul6804164519391"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p15107131419311"><a name="en-us_topic_0000002313328922_p15107131419311"></a><a name="en-us_topic_0000002313328922_p15107131419311"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul1171554793920"></a><a name="en-us_topic_0000002313328922_ul1171554793920"></a><ul id="en-us_topic_0000002313328922_ul1171554793920"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p0108141463115"><a name="en-us_topic_0000002313328922_p0108141463115"></a><a name="en-us_topic_0000002313328922_p0108141463115"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="en-us_topic_0000002313328922_ul1998674913913"></a><a name="en-us_topic_0000002313328922_ul1998674913913"></a><ul id="en-us_topic_0000002313328922_ul1998674913913"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p4108414123111"><a name="en-us_topic_0000002313328922_p4108414123111"></a><a name="en-us_topic_0000002313328922_p4108414123111"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul66071251153916"></a><a name="en-us_topic_0000002313328922_ul66071251153916"></a><ul id="en-us_topic_0000002313328922_ul66071251153916"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p151081314153116"><a name="en-us_topic_0000002313328922_p151081314153116"></a><a name="en-us_topic_0000002313328922_p151081314153116"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="en-us_topic_0000002313328922_ul1621625303919"></a><a name="en-us_topic_0000002313328922_ul1621625303919"></a><ul id="en-us_topic_0000002313328922_ul1621625303919"><li>(B,S,Dr)</li><li>(T,Dr)</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row910901403114"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p71081514113115"><a name="en-us_topic_0000002313328922_p71081514113115"></a><a name="en-us_topic_0000002313328922_p71081514113115"></a>cache_index</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1510861463119"><a name="en-us_topic_0000002313328922_p1510861463119"></a><a name="en-us_topic_0000002313328922_p1510861463119"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="en-us_topic_0000002313328922_ul19841958123912"></a><a name="en-us_topic_0000002313328922_ul19841958123912"></a><ul id="en-us_topic_0000002313328922_ul19841958123912"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p181081514133118"><a name="en-us_topic_0000002313328922_p181081514133118"></a><a name="en-us_topic_0000002313328922_p181081514133118"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul1850621614011"></a><a name="en-us_topic_0000002313328922_ul1850621614011"></a><ul id="en-us_topic_0000002313328922_ul1850621614011"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p18108314143115"><a name="en-us_topic_0000002313328922_p18108314143115"></a><a name="en-us_topic_0000002313328922_p18108314143115"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="en-us_topic_0000002313328922_ul1814741813406"></a><a name="en-us_topic_0000002313328922_ul1814741813406"></a><ul id="en-us_topic_0000002313328922_ul1814741813406"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p15109131433112"><a name="en-us_topic_0000002313328922_p15109131433112"></a><a name="en-us_topic_0000002313328922_p15109131433112"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul14946151914019"></a><a name="en-us_topic_0000002313328922_ul14946151914019"></a><ul id="en-us_topic_0000002313328922_ul14946151914019"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p31091614153119"><a name="en-us_topic_0000002313328922_p31091614153119"></a><a name="en-us_topic_0000002313328922_p31091614153119"></a>int64</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="en-us_topic_0000002313328922_ul12670621204012"></a><a name="en-us_topic_0000002313328922_ul12670621204012"></a><ul id="en-us_topic_0000002313328922_ul12670621204012"><li>(B,S)</li><li>(T)</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row110918146313"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1010914146318"><a name="en-us_topic_0000002313328922_p1010914146318"></a><a name="en-us_topic_0000002313328922_p1010914146318"></a>kv_cache</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1610919146315"><a name="en-us_topic_0000002313328922_p1610919146315"></a><a name="en-us_topic_0000002313328922_p1610919146315"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p91099145315"><a name="en-us_topic_0000002313328922_p91099145315"></a><a name="en-us_topic_0000002313328922_p91099145315"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p510911413112"><a name="en-us_topic_0000002313328922_p510911413112"></a><a name="en-us_topic_0000002313328922_p510911413112"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p1248419487458"><a name="en-us_topic_0000002313328922_p1248419487458"></a><a name="en-us_topic_0000002313328922_p1248419487458"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1610921483112"><a name="en-us_topic_0000002313328922_p1610921483112"></a><a name="en-us_topic_0000002313328922_p1610921483112"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p102779548454"><a name="en-us_topic_0000002313328922_p102779548454"></a><a name="en-us_topic_0000002313328922_p102779548454"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p61097141319"><a name="en-us_topic_0000002313328922_p61097141319"></a><a name="en-us_topic_0000002313328922_p61097141319"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p269055604512"><a name="en-us_topic_0000002313328922_p269055604512"></a><a name="en-us_topic_0000002313328922_p269055604512"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p101091814193117"><a name="en-us_topic_0000002313328922_p101091814193117"></a><a name="en-us_topic_0000002313328922_p101091814193117"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p188285815456"><a name="en-us_topic_0000002313328922_p188285815456"></a><a name="en-us_topic_0000002313328922_p188285815456"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row1011013147312"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p110971483115"><a name="en-us_topic_0000002313328922_p110971483115"></a><a name="en-us_topic_0000002313328922_p110971483115"></a>kr_cache</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p16109101411314"><a name="en-us_topic_0000002313328922_p16109101411314"></a><a name="en-us_topic_0000002313328922_p16109101411314"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p20110161453119"><a name="en-us_topic_0000002313328922_p20110161453119"></a><a name="en-us_topic_0000002313328922_p20110161453119"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p1311013144317"><a name="en-us_topic_0000002313328922_p1311013144317"></a><a name="en-us_topic_0000002313328922_p1311013144317"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p811019146316"><a name="en-us_topic_0000002313328922_p811019146316"></a><a name="en-us_topic_0000002313328922_p811019146316"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p11110414103116"><a name="en-us_topic_0000002313328922_p11110414103116"></a><a name="en-us_topic_0000002313328922_p11110414103116"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p420731810463"><a name="en-us_topic_0000002313328922_p420731810463"></a><a name="en-us_topic_0000002313328922_p420731810463"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p4110121414313"><a name="en-us_topic_0000002313328922_p4110121414313"></a><a name="en-us_topic_0000002313328922_p4110121414313"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p111101814123114"><a name="en-us_topic_0000002313328922_p111101814123114"></a><a name="en-us_topic_0000002313328922_p111101814123114"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p9110121463118"><a name="en-us_topic_0000002313328922_p9110121463118"></a><a name="en-us_topic_0000002313328922_p9110121463118"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p345722317462"><a name="en-us_topic_0000002313328922_p345722317462"></a><a name="en-us_topic_0000002313328922_p345722317462"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row1211161411319"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p111020146314"><a name="en-us_topic_0000002313328922_p111020146314"></a><a name="en-us_topic_0000002313328922_p111020146314"></a>dequant_scale_x</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p511031410310"><a name="en-us_topic_0000002313328922_p511031410310"></a><a name="en-us_topic_0000002313328922_p511031410310"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p121101014193110"><a name="en-us_topic_0000002313328922_p121101014193110"></a><a name="en-us_topic_0000002313328922_p121101014193110"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p81101214163111"><a name="en-us_topic_0000002313328922_p81101214163111"></a><a name="en-us_topic_0000002313328922_p81101214163111"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p5111214133114"><a name="en-us_topic_0000002313328922_p5111214133114"></a><a name="en-us_topic_0000002313328922_p5111214133114"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p11111114153110"><a name="en-us_topic_0000002313328922_p11111114153110"></a><a name="en-us_topic_0000002313328922_p11111114153110"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p5111111412314"><a name="en-us_topic_0000002313328922_p5111111412314"></a><a name="en-us_topic_0000002313328922_p5111111412314"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p14111161416312"><a name="en-us_topic_0000002313328922_p14111161416312"></a><a name="en-us_topic_0000002313328922_p14111161416312"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul1875112920408"></a><a name="en-us_topic_0000002313328922_ul1875112920408"></a><ul id="en-us_topic_0000002313328922_ul1875112920408"><li>(B*S,1)</li><li>(T,1)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p511141420316"><a name="en-us_topic_0000002313328922_p511141420316"></a><a name="en-us_topic_0000002313328922_p511141420316"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="en-us_topic_0000002313328922_ul187764617408"></a><a name="en-us_topic_0000002313328922_ul187764617408"></a><ul id="en-us_topic_0000002313328922_ul187764617408"><li>(B*S,1)</li><li>(T,1)</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row16112614153117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p2011112146316"><a name="en-us_topic_0000002313328922_p2011112146316"></a><a name="en-us_topic_0000002313328922_p2011112146316"></a>dequant_scale_w_dq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p511141433113"><a name="en-us_topic_0000002313328922_p511141433113"></a><a name="en-us_topic_0000002313328922_p511141433113"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p111111214133114"><a name="en-us_topic_0000002313328922_p111111214133114"></a><a name="en-us_topic_0000002313328922_p111111214133114"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p2111131411312"><a name="en-us_topic_0000002313328922_p2111131411312"></a><a name="en-us_topic_0000002313328922_p2111131411312"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p151123140315"><a name="en-us_topic_0000002313328922_p151123140315"></a><a name="en-us_topic_0000002313328922_p151123140315"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1011291411315"><a name="en-us_topic_0000002313328922_p1011291411315"></a><a name="en-us_topic_0000002313328922_p1011291411315"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p131120149316"><a name="en-us_topic_0000002313328922_p131120149316"></a><a name="en-us_topic_0000002313328922_p131120149316"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p9112111473120"><a name="en-us_topic_0000002313328922_p9112111473120"></a><a name="en-us_topic_0000002313328922_p9112111473120"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p12112121423118"><a name="en-us_topic_0000002313328922_p12112121423118"></a><a name="en-us_topic_0000002313328922_p12112121423118"></a>1,Hcq</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p011241416315"><a name="en-us_topic_0000002313328922_p011241416315"></a><a name="en-us_topic_0000002313328922_p011241416315"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p1611251453115"><a name="en-us_topic_0000002313328922_p1611251453115"></a><a name="en-us_topic_0000002313328922_p1611251453115"></a>1,Hcq</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row411310144314"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p61124144317"><a name="en-us_topic_0000002313328922_p61124144317"></a><a name="en-us_topic_0000002313328922_p61124144317"></a>dequant_scale_w_uq_qr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1511391413316"><a name="en-us_topic_0000002313328922_p1511391413316"></a><a name="en-us_topic_0000002313328922_p1511391413316"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1611321414310"><a name="en-us_topic_0000002313328922_p1611321414310"></a><a name="en-us_topic_0000002313328922_p1611321414310"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p191131214143118"><a name="en-us_topic_0000002313328922_p191131214143118"></a><a name="en-us_topic_0000002313328922_p191131214143118"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p1011381416313"><a name="en-us_topic_0000002313328922_p1011381416313"></a><a name="en-us_topic_0000002313328922_p1011381416313"></a>(1,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p171131514113119"><a name="en-us_topic_0000002313328922_p171131514113119"></a><a name="en-us_topic_0000002313328922_p171131514113119"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p1774723914611"><a name="en-us_topic_0000002313328922_p1774723914611"></a><a name="en-us_topic_0000002313328922_p1774723914611"></a>(1,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p11139145317"><a name="en-us_topic_0000002313328922_p11139145317"></a><a name="en-us_topic_0000002313328922_p11139145317"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p13113171420315"><a name="en-us_topic_0000002313328922_p13113171420315"></a><a name="en-us_topic_0000002313328922_p13113171420315"></a>(1,N*(D+Dr))</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p16113314153120"><a name="en-us_topic_0000002313328922_p16113314153120"></a><a name="en-us_topic_0000002313328922_p16113314153120"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p1081344154613"><a name="en-us_topic_0000002313328922_p1081344154613"></a><a name="en-us_topic_0000002313328922_p1081344154613"></a>(1,N*(D+Dr))</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row19114101463113"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p13113181419315"><a name="en-us_topic_0000002313328922_p13113181419315"></a><a name="en-us_topic_0000002313328922_p13113181419315"></a>dequant_scale_w_dkv_kr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p2113114133115"><a name="en-us_topic_0000002313328922_p2113114133115"></a><a name="en-us_topic_0000002313328922_p2113114133115"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p91131145310"><a name="en-us_topic_0000002313328922_p91131145310"></a><a name="en-us_topic_0000002313328922_p91131145310"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p15113111415314"><a name="en-us_topic_0000002313328922_p15113111415314"></a><a name="en-us_topic_0000002313328922_p15113111415314"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p18113191433113"><a name="en-us_topic_0000002313328922_p18113191433113"></a><a name="en-us_topic_0000002313328922_p18113191433113"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p91131914103116"><a name="en-us_topic_0000002313328922_p91131914103116"></a><a name="en-us_topic_0000002313328922_p91131914103116"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p911341412318"><a name="en-us_topic_0000002313328922_p911341412318"></a><a name="en-us_topic_0000002313328922_p911341412318"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p18114131413113"><a name="en-us_topic_0000002313328922_p18114131413113"></a><a name="en-us_topic_0000002313328922_p18114131413113"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p711471414312"><a name="en-us_topic_0000002313328922_p711471414312"></a><a name="en-us_topic_0000002313328922_p711471414312"></a>(1,Hckv+Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p91141614143115"><a name="en-us_topic_0000002313328922_p91141614143115"></a><a name="en-us_topic_0000002313328922_p91141614143115"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p152341456204613"><a name="en-us_topic_0000002313328922_p152341456204613"></a><a name="en-us_topic_0000002313328922_p152341456204613"></a>(1,Hckv+Dr)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row1811491433118"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p151141514103110"><a name="en-us_topic_0000002313328922_p151141514103110"></a><a name="en-us_topic_0000002313328922_p151141514103110"></a>quant_scale_ckv</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p10114514143118"><a name="en-us_topic_0000002313328922_p10114514143118"></a><a name="en-us_topic_0000002313328922_p10114514143118"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1311418145312"><a name="en-us_topic_0000002313328922_p1311418145312"></a><a name="en-us_topic_0000002313328922_p1311418145312"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p1114141413111"><a name="en-us_topic_0000002313328922_p1114141413111"></a><a name="en-us_topic_0000002313328922_p1114141413111"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p9114171417313"><a name="en-us_topic_0000002313328922_p9114171417313"></a><a name="en-us_topic_0000002313328922_p9114171417313"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p2011441433113"><a name="en-us_topic_0000002313328922_p2011441433113"></a><a name="en-us_topic_0000002313328922_p2011441433113"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p61141614173119"><a name="en-us_topic_0000002313328922_p61141614173119"></a><a name="en-us_topic_0000002313328922_p61141614173119"></a>(1,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p51141714123118"><a name="en-us_topic_0000002313328922_p51141714123118"></a><a name="en-us_topic_0000002313328922_p51141714123118"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p911415141314"><a name="en-us_topic_0000002313328922_p911415141314"></a><a name="en-us_topic_0000002313328922_p911415141314"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p161143149317"><a name="en-us_topic_0000002313328922_p161143149317"></a><a name="en-us_topic_0000002313328922_p161143149317"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p1513369122213"><a name="en-us_topic_0000002313328922_p1513369122213"></a><a name="en-us_topic_0000002313328922_p1513369122213"></a>(1,Hckv)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row711512142317"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p10114181412318"><a name="en-us_topic_0000002313328922_p10114181412318"></a><a name="en-us_topic_0000002313328922_p10114181412318"></a>quant_scale_ckr</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p5115131412312"><a name="en-us_topic_0000002313328922_p5115131412312"></a><a name="en-us_topic_0000002313328922_p5115131412312"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p3115814193118"><a name="en-us_topic_0000002313328922_p3115814193118"></a><a name="en-us_topic_0000002313328922_p3115814193118"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p121151414103110"><a name="en-us_topic_0000002313328922_p121151414103110"></a><a name="en-us_topic_0000002313328922_p121151414103110"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p18115314183111"><a name="en-us_topic_0000002313328922_p18115314183111"></a><a name="en-us_topic_0000002313328922_p18115314183111"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p4115114113119"><a name="en-us_topic_0000002313328922_p4115114113119"></a><a name="en-us_topic_0000002313328922_p4115114113119"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p511518143318"><a name="en-us_topic_0000002313328922_p511518143318"></a><a name="en-us_topic_0000002313328922_p511518143318"></a>(1,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1411591443113"><a name="en-us_topic_0000002313328922_p1411591443113"></a><a name="en-us_topic_0000002313328922_p1411591443113"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p16115181419312"><a name="en-us_topic_0000002313328922_p16115181419312"></a><a name="en-us_topic_0000002313328922_p16115181419312"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p1011511473112"><a name="en-us_topic_0000002313328922_p1011511473112"></a><a name="en-us_topic_0000002313328922_p1011511473112"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p171151914133118"><a name="en-us_topic_0000002313328922_p171151914133118"></a><a name="en-us_topic_0000002313328922_p171151914133118"></a>/</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row10116114103112"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p11115161418311"><a name="en-us_topic_0000002313328922_p11115161418311"></a><a name="en-us_topic_0000002313328922_p11115161418311"></a>smooth_scales_cq</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p12115191483119"><a name="en-us_topic_0000002313328922_p12115191483119"></a><a name="en-us_topic_0000002313328922_p12115191483119"></a>Not required</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p111151147315"><a name="en-us_topic_0000002313328922_p111151147315"></a><a name="en-us_topic_0000002313328922_p111151147315"></a>/</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p81152014153118"><a name="en-us_topic_0000002313328922_p81152014153118"></a><a name="en-us_topic_0000002313328922_p81152014153118"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p1411591483118"><a name="en-us_topic_0000002313328922_p1411591483118"></a><a name="en-us_topic_0000002313328922_p1411591483118"></a>(1,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p311514144313"><a name="en-us_topic_0000002313328922_p311514144313"></a><a name="en-us_topic_0000002313328922_p311514144313"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p171151147319"><a name="en-us_topic_0000002313328922_p171151147319"></a><a name="en-us_topic_0000002313328922_p171151147319"></a>(1,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p0116191443116"><a name="en-us_topic_0000002313328922_p0116191443116"></a><a name="en-us_topic_0000002313328922_p0116191443116"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p104931614134716"><a name="en-us_topic_0000002313328922_p104931614134716"></a><a name="en-us_topic_0000002313328922_p104931614134716"></a>(1,Hcq)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p19116614173110"><a name="en-us_topic_0000002313328922_p19116614173110"></a><a name="en-us_topic_0000002313328922_p19116614173110"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p1880211619471"><a name="en-us_topic_0000002313328922_p1880211619471"></a><a name="en-us_topic_0000002313328922_p1880211619471"></a>(1,Hcq)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row1611711147313"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p18116191473120"><a name="en-us_topic_0000002313328922_p18116191473120"></a><a name="en-us_topic_0000002313328922_p18116191473120"></a>query</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p0116171433116"><a name="en-us_topic_0000002313328922_p0116171433116"></a><a name="en-us_topic_0000002313328922_p0116171433116"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="en-us_topic_0000002313328922_ul15120190144120"></a><a name="en-us_topic_0000002313328922_ul15120190144120"></a><ul id="en-us_topic_0000002313328922_ul15120190144120"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p11116191483116"><a name="en-us_topic_0000002313328922_p11116191483116"></a><a name="en-us_topic_0000002313328922_p11116191483116"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul1945521894110"></a><a name="en-us_topic_0000002313328922_ul1945521894110"></a><ul id="en-us_topic_0000002313328922_ul1945521894110"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p31167143319"><a name="en-us_topic_0000002313328922_p31167143319"></a><a name="en-us_topic_0000002313328922_p31167143319"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="en-us_topic_0000002313328922_ul5222182074120"></a><a name="en-us_topic_0000002313328922_ul5222182074120"></a><ul id="en-us_topic_0000002313328922_ul5222182074120"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p181162148319"><a name="en-us_topic_0000002313328922_p181162148319"></a><a name="en-us_topic_0000002313328922_p181162148319"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul4809112113415"></a><a name="en-us_topic_0000002313328922_ul4809112113415"></a><ul id="en-us_topic_0000002313328922_ul4809112113415"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p611671419319"><a name="en-us_topic_0000002313328922_p611671419319"></a><a name="en-us_topic_0000002313328922_p611671419319"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="en-us_topic_0000002313328922_ul1831425194112"></a><a name="en-us_topic_0000002313328922_ul1831425194112"></a><ul id="en-us_topic_0000002313328922_ul1831425194112"><li>(B,S,N,Hckv)</li><li>(T,N,Hckv)</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row1411711410316"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p12117111413112"><a name="en-us_topic_0000002313328922_p12117111413112"></a><a name="en-us_topic_0000002313328922_p12117111413112"></a>query_rope</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p4117514153118"><a name="en-us_topic_0000002313328922_p4117514153118"></a><a name="en-us_topic_0000002313328922_p4117514153118"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><a name="en-us_topic_0000002313328922_ul10316123710419"></a><a name="en-us_topic_0000002313328922_ul10316123710419"></a><ul id="en-us_topic_0000002313328922_ul10316123710419"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p1511719141313"><a name="en-us_topic_0000002313328922_p1511719141313"></a><a name="en-us_topic_0000002313328922_p1511719141313"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul225985618412"></a><a name="en-us_topic_0000002313328922_ul225985618412"></a><ul id="en-us_topic_0000002313328922_ul225985618412"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p711741414318"><a name="en-us_topic_0000002313328922_p711741414318"></a><a name="en-us_topic_0000002313328922_p711741414318"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><a name="en-us_topic_0000002313328922_ul183131705429"></a><a name="en-us_topic_0000002313328922_ul183131705429"></a><ul id="en-us_topic_0000002313328922_ul183131705429"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p3117201420317"><a name="en-us_topic_0000002313328922_p3117201420317"></a><a name="en-us_topic_0000002313328922_p3117201420317"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><a name="en-us_topic_0000002313328922_ul383917194210"></a><a name="en-us_topic_0000002313328922_ul383917194210"></a><ul id="en-us_topic_0000002313328922_ul383917194210"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p111731415319"><a name="en-us_topic_0000002313328922_p111731415319"></a><a name="en-us_topic_0000002313328922_p111731415319"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="en-us_topic_0000002313328922_ul537183164214"></a><a name="en-us_topic_0000002313328922_ul537183164214"></a><ul id="en-us_topic_0000002313328922_ul537183164214"><li>(B,S,N,Dr)</li><li>(T,N,Dr)</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row111871453119"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p711719148317"><a name="en-us_topic_0000002313328922_p711719148317"></a><a name="en-us_topic_0000002313328922_p711719148317"></a>kv_cache</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1117814153111"><a name="en-us_topic_0000002313328922_p1117814153111"></a><a name="en-us_topic_0000002313328922_p1117814153111"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p115809401478"><a name="en-us_topic_0000002313328922_p115809401478"></a><a name="en-us_topic_0000002313328922_p115809401478"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p11178144316"><a name="en-us_topic_0000002313328922_p11178144316"></a><a name="en-us_topic_0000002313328922_p11178144316"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p3418442164713"><a name="en-us_topic_0000002313328922_p3418442164713"></a><a name="en-us_topic_0000002313328922_p3418442164713"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p12118141419319"><a name="en-us_topic_0000002313328922_p12118141419319"></a><a name="en-us_topic_0000002313328922_p12118141419319"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p86601644114719"><a name="en-us_topic_0000002313328922_p86601644114719"></a><a name="en-us_topic_0000002313328922_p86601644114719"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p5118201443110"><a name="en-us_topic_0000002313328922_p5118201443110"></a><a name="en-us_topic_0000002313328922_p5118201443110"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p47124724714"><a name="en-us_topic_0000002313328922_p47124724714"></a><a name="en-us_topic_0000002313328922_p47124724714"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p5118014193110"><a name="en-us_topic_0000002313328922_p5118014193110"></a><a name="en-us_topic_0000002313328922_p5118014193110"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p980949134717"><a name="en-us_topic_0000002313328922_p980949134717"></a><a name="en-us_topic_0000002313328922_p980949134717"></a>(BlockNum,BlockSize,Nkv,Hckv)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row0119171483111"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p101182014173116"><a name="en-us_topic_0000002313328922_p101182014173116"></a><a name="en-us_topic_0000002313328922_p101182014173116"></a>kr_cache</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p41180146312"><a name="en-us_topic_0000002313328922_p41180146312"></a><a name="en-us_topic_0000002313328922_p41180146312"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p18382381877"><a name="en-us_topic_0000002313328922_p18382381877"></a><a name="en-us_topic_0000002313328922_p18382381877"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p8118141411318"><a name="en-us_topic_0000002313328922_p8118141411318"></a><a name="en-us_topic_0000002313328922_p8118141411318"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p17382411973"><a name="en-us_topic_0000002313328922_p17382411973"></a><a name="en-us_topic_0000002313328922_p17382411973"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p10118101413114"><a name="en-us_topic_0000002313328922_p10118101413114"></a><a name="en-us_topic_0000002313328922_p10118101413114"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p127212315481"><a name="en-us_topic_0000002313328922_p127212315481"></a><a name="en-us_topic_0000002313328922_p127212315481"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p51181614113119"><a name="en-us_topic_0000002313328922_p51181614113119"></a><a name="en-us_topic_0000002313328922_p51181614113119"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p1088320441076"><a name="en-us_topic_0000002313328922_p1088320441076"></a><a name="en-us_topic_0000002313328922_p1088320441076"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p18118181493113"><a name="en-us_topic_0000002313328922_p18118181493113"></a><a name="en-us_topic_0000002313328922_p18118181493113"></a>bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><p id="en-us_topic_0000002313328922_p18451144612718"><a name="en-us_topic_0000002313328922_p18451144612718"></a><a name="en-us_topic_0000002313328922_p18451144612718"></a>(BlockNum,BlockSize,Nkv,Dr)</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002313328922_row17119414153117"><td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p17119121411315"><a name="en-us_topic_0000002313328922_p17119121411315"></a><a name="en-us_topic_0000002313328922_p17119121411315"></a>dequant_scale_q_nope</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1111920146316"><a name="en-us_topic_0000002313328922_p1111920146316"></a><a name="en-us_topic_0000002313328922_p1111920146316"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1511911418312"><a name="en-us_topic_0000002313328922_p1511911418312"></a><a name="en-us_topic_0000002313328922_p1511911418312"></a>(1<span id="en-us_topic_0000002313328922_ph18586141419271"><a name="en-us_topic_0000002313328922_ph18586141419271"></a><a name="en-us_topic_0000002313328922_ph18586141419271"></a>,</span>)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p1611918142315"><a name="en-us_topic_0000002313328922_p1611918142315"></a><a name="en-us_topic_0000002313328922_p1611918142315"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p19351141464819"><a name="en-us_topic_0000002313328922_p19351141464819"></a><a name="en-us_topic_0000002313328922_p19351141464819"></a>(1<span id="en-us_topic_0000002313328922_ph35291817152714"><a name="en-us_topic_0000002313328922_ph35291817152714"></a><a name="en-us_topic_0000002313328922_ph35291817152714"></a>,</span>)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p6119101414312"><a name="en-us_topic_0000002313328922_p6119101414312"></a><a name="en-us_topic_0000002313328922_p6119101414312"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.080908090809082%"><p id="en-us_topic_0000002313328922_p14893131519488"><a name="en-us_topic_0000002313328922_p14893131519488"></a><a name="en-us_topic_0000002313328922_p14893131519488"></a>(1<span id="en-us_topic_0000002313328922_ph166319122718"><a name="en-us_topic_0000002313328922_ph166319122718"></a><a name="en-us_topic_0000002313328922_ph166319122718"></a>,</span>)</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.09090909090909%"><p id="en-us_topic_0000002313328922_p1611911414313"><a name="en-us_topic_0000002313328922_p1611911414313"></a><a name="en-us_topic_0000002313328922_p1611911414313"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="9.100910091009101%"><p id="en-us_topic_0000002313328922_p13273111784815"><a name="en-us_topic_0000002313328922_p13273111784815"></a><a name="en-us_topic_0000002313328922_p13273111784815"></a>(1<span id="en-us_topic_0000002313328922_ph10953620192719"><a name="en-us_topic_0000002313328922_ph10953620192719"></a><a name="en-us_topic_0000002313328922_ph10953620192719"></a>,</span>)</p>
    </td>
    <td class="cellrowborder" valign="top" width="8.04080408040804%"><p id="en-us_topic_0000002313328922_p1911971463117"><a name="en-us_topic_0000002313328922_p1911971463117"></a><a name="en-us_topic_0000002313328922_p1911971463117"></a>float</p>
    </td>
    <td class="cellrowborder" valign="top" width="10.141014101410141%"><a name="en-us_topic_0000002313328922_ul5238121054219"></a><a name="en-us_topic_0000002313328922_ul5238121054219"></a><ul id="en-us_topic_0000002313328922_ul5238121054219"><li>(B*S,N,1)</li><li>(T,N,1)</li></ul>
    </td>
    </tr>
    </tbody>
    </table>

## Examples<a name="en-us_topic_0000002313328922_section983519211229"></a>

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
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    cache_num_slots = kv_cache.size(0) * kv_cache.size(1)
    cache_index = torch.randint(0, cache_num_slots, (B, S), dtype=torch.int64).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    
    # Call the MlaProlog operator.
    query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla, dequant_scale_q_nope_mla = torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq, rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
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
    kv_cache = torch.rand(BlockNum, BlockSize, Nkv, Hckv, dtype=torch.bfloat16).npu()
    kr_cache = torch.rand(BlockNum, BlockSize, Nkv, Dr, dtype=torch.bfloat16).npu()
    cache_num_slots = kv_cache.size(0) * kv_cache.size(1)
    cache_index = torch.randint(0, cache_num_slots, (B, S), dtype=torch.int64).npu()
    rmsnorm_epsilon_cq = 1.0e-5
    rmsnorm_epsilon_ckv = 1.0e-5
    cache_mode = "PA_BSND"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        query_mla, query_rope_mla, kv_cache_out_mla, kr_cache_out_mla, dequant_scale_q_nope_mla = torch_npu.npu_mla_prolog_v2(token_x, w_dq_cast, w_uq_qr_cast, w_uk, w_dkv_kr_cast, rmsnorm_gamma_cq,rmsnorm_gamma_ckv, rope_sin, rope_cos, cache_index, kv_cache, kr_cache, rmsnorm_epsilon_cq=rmsnorm_epsilon_cq, rmsnorm_epsilon_ckv=rmsnorm_epsilon_ckv, cache_mode=cache_mode)
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
