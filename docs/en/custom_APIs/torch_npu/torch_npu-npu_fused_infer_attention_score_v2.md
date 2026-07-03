# torch\_npu.npu\_fused\_infer\_attention\_score\_v2<a name="en-us_topic_0000001979260729"></a>

## Supported Products<a name="en-us_topic_0000001832267082_section14441124184110"></a>

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A3 inference products</term>  | √  |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas A2 inference products</term>|    √     |

## Function<a name="en-us_topic_0000001832267082_section14441124184110"></a>

- Description: Adapts to the `FlashAttention` operator in the incremental and full inference scenarios, supporting both full computation (`PromptFlashAttention`) and incremental computation (`IncreFlashAttention`). This API is recommended when system prefix, left padding, unified KV quantization parameters, and `pertensor` full quantization are not involved. Otherwise, use the legacy API `npu_fused_infer_attention_score`.
- Formula:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    The product of $Q$ and $K^T$ represents the attention scores of the input $x$, where $d$ denotes the minimum unit size of the hidden layer. To prevent this value from becoming excessively large, it is typically scaled by dividing by the square root of $d$, followed by row-wise softmax normalization. Multiplying this result by $V$ yields an $n$-by-$d$ matrix, where $n$ is the number of rows in the output matrix.

## Prototype<a name="en-us_topic_0000001832267082_section45077510411"></a>

```python
torch_npu.npu_fused_infer_attention_score_v2(query, key, value, *, query_rope=None, key_rope=None, pse_shift=None, atten_mask=None, actual_seq_qlen=None, actual_seq_kvlen=None, block_table=None, dequant_scale_query=None, dequant_scale_key=None, dequant_offset_key=None, dequant_scale_value=None, dequant_offset_value=None, dequant_scale_key_rope=None, quant_scale_out=None, quant_offset_out=None, quant_scale_p=None, learnable_sink=None, num_query_heads=1, num_key_value_heads=0, softmax_scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", sparse_mode=0, block_size=0, query_quant_mode=0, key_quant_mode=0, value_quant_mode=0, inner_precise=0, return_softmax_lse=False, query_dtype=None, key_dtype=None, value_dtype=None, query_rope_dtype=None, key_rope_dtype=None, key_shared_prefix_dtype=None, value_shared_prefix_dtype=None, dequant_scale_query_dtype=None, dequant_scale_key_dtype=None, dequant_scale_value_dtype=None, dequant_scale_key_rope_dtype=None, out_dtype=None) -> (Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000001832267082_section112637109429"></a>

> [!NOTE]   
>
> - Dimension definitions for the `query`, `key`, and `value` parameters:<br>`B` (`Batch Size`) indicates the input sample batch size.<br>`S` (`Sequence Length`) indicates the input sample sequence length.<br>`H` (`Head Size`) indicates the hidden layer size.<br>`N` (`Head Num`) indicates the number of heads.<br> `D` (`Head Dim`) indicates the minimum unit size of the hidden layer, satisfying `D = H/N`.<br>`T` indicates the cumulative sum of the sequence lengths of all batch input samples.
> - `Q_S` and `S1` indicate the `S` dimension in the shape of `query`.<br>`KV_S` and `S2` indicate the `S` dimension in the shape of `key` and `value`.<br>`Q_N` indicates `num_query_heads`.<br>`KV_N` indicates `num_key_value_heads`.
>

**Parameter Reference**

| Parameter| Required (Yes/No)| Type| Default Value| Description|
|------|-----------|------|--------|------|
| query | Yes| Tensor | - | Query input.|
| key | Yes| Tensor | - | Key input|
| value | Yes| Tensor | - | Value input|
| query_rope | No| Tensor | None | RoPE information of `query` in the MLA structure.|
| key_rope | No| Tensor | None | Rope information of `key` in the MLA structure.|
| pse_shift | No| Tensor | None | Positional encoding parameter.|
| atten_mask | No| Tensor | None | Attention mask.|
| actual_seq_qlen | No| List[Int] | None | Valid sequence length of `query`.|
| actual_seq_kvlen | No| List[Int] | None | Valid sequence length of `key` and `value`.|
| block_table | No| Tensor | None | Block mapping table for page attention.|
| dequant_scale_query | No| Tensor | None | Dequantization parameter of `query`.|
| dequant_scale_key | No| Tensor | None | Dequantization factor of `key`.|
| dequant_offset_key | No| Tensor | None | Dequantization offset of `key`.|
| dequant_scale_value | No| Tensor | None | Dequantization factor of `value`.|
| dequant_offset_value | No| Tensor | None | Dequantization offset of `value`.|
| dequant_scale_key_rope | No| Tensor | None | This parameter is reserved and not used currently.|
| quant_scale_out | No| Tensor | None | Output quantization factor.|
| quant_offset_out | No| Tensor | None | Output quantization offset.|
| quant_scale_p | No| Tensor | None | This parameter is reserved and not used currently.|
| learnable_sink | No| Tensor | None | Learnable sink token.|
| num_query_heads | No| int | 1 | Number of query heads.|
| num_key_value_heads | No| int | 0 | Number of heads for `key` and `value`. `0` indicates that the value is identical to the number of query heads.|
| softmax_scale | No| float | 1.0 | Scale factor. Provide a value that equals `1/√D`.|
| pre_tokens | No| int | 2147483647 | Number of preceding tokens for sparse computation.|
| next_tokens | No| int | 2147483647 | Number of subsequent tokens for sparse computation.|
| input_layout | No| str | "BSH" | Input data layout.|
| sparse_mode | No| int | 0 | Sparse mode.|
| block_size | No| int | 0 | Maximum number of tokens in each block of PageAttention.|
| query_quant_mode | No| int | 0 | Fake-quantization mode for `query`.|
| key_quant_mode | No| int | 0 | Fake-quantization mode for `key`.|
| value_quant_mode | No| int | 0 | Fake-quantization mode for `value`.|
| inner_precise | No| int | 0 | Precision mode.|
| return_softmax_lse | No| bool | False | Specifies whether to output `softmax_lse`.|
| query_dtype | No| int | None | This parameter is reserved and not used currently.|
| key_dtype | No| int | None | This parameter is reserved and not used currently.|
| value_dtype | No| int | None | This parameter is reserved and not used currently.|
| query_rope_dtype | No| int | None | This parameter is reserved and not used currently.|
| key_rope_dtype | No| int | None | This parameter is reserved and not used currently.|
| key_shared_prefix_dtype | No| int | None | This parameter is reserved and not used currently.|
| value_shared_prefix_dtype | No| int | None | This parameter is reserved and not used currently.|
| dequant_scale_query_dtype | No| int | None | This parameter is reserved and not used currently.|
| dequant_scale_key_dtype | No| int | None | This parameter is reserved and not used currently.|
| dequant_scale_value_dtype | No| int | None | This parameter is reserved and not used currently.|
| dequant_scale_key_rope_dtype | No| int | None | This parameter is reserved and not used currently.|
| out_dtype | No| int | None | Output data type.|

- **`query`** (`Tensor`): Required. Query input of the attention structure, `Q` in the formula. Non-contiguous tensors are not supported. The data type can be `float16` or `bfloat16`. The data layout can be ND.   
    
- **`key`** (`Tensor`): Required. Key input of the attention structure, `K` in the formula. Non-contiguous tensors are not supported. The data type can be `float16`, `bfloat16`, `int8`, or `int4` (`int32`). The data layout can be ND.  
     
- **`value`** (`Tensor`): Required. Value input of the attention structure, `V` in the formula. Non-contiguous tensors are not supported. The data type can be `float16`, `bfloat16`, `int8`, or `int4` (`int32`). The data layout can be ND.   
    
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`query_rope`** (`Tensor`): Optional. RoPE information of the query in the Multi-head Latent Attention (MLA) structure. The data type can be `float16` or `bfloat16`. Non-contiguous tensors are not supported. The data layout can be ND.
- **`key_rope`** (`Tensor`): Optional. RoPE information of the key in the Multi-head Latent Attention (MLA) structure. The data type can be `float16` or `bfloat16`. Non-contiguous tensors are not supported. The data layout can be ND.
- **`pse_shift`** (`Tensor`): Optional. Positional encoding parameter within the attention structure. The data type can be `float16` or `bfloat16`. The data type of this parameter and that of `query` must comply with type promotion rules. Non-contiguous tensors are not supported. The data layout can be ND. This parameter can be set to `None` if this feature is not used.

    - When `Q_S` is greater than 1: if `pse_shift` is `float16`, `query` must be `float16` or `int8`; if `pse_shift` is `bfloat16`, `query` must be `bfloat16`. The input shape must be `(B, Q_N, Q_S, KV_S)` or `(1, Q_N, Q_S, KV_S)`. If `KV_S` of `pse_shift` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. There are no specific requirements for the padding values of the padded portion.
    - When `Q_S` is equal to 1: if `pse_shift` is `float16`, `query` must be `float16`; if `pse_shift` is `bfloat16`, `query` must be `bfloat16`. The input shape must be (B, Q_N, 1, KV_S) or (1, Q_N, 1, KV_S). If `KV_S` of `pse_shift` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. There are no specific requirements for the padding values of the padded portion.

- **`atten_mask`** (`Tensor`): Optional. Masks the QK product to indicate whether correlation between tokens is computed. The data type can be `bool`, `int8`, or `uint8`. Non-contiguous tensors are not supported. The data layout can be ND. This parameter can be set to `None` if this feature is not used.
    - When `sparse_mode` is `0` or `1`:
        - The input shape can be (1, Q_S, KV_S), (B, 1, Q_S, KV_S), or (1, 1, Q_S, KV_S).
        - When `input_layout` is `BSH`, `BSND`, `BNSD`, or `BNSD_BSND`, the `D` dimensions of `query`, `key`, and `value` are identical, and `query_rope` and `key_rope` are omitted, a shape of (B, KV_S) can be passed if `Q_S` is 1, and (Q_S, KV_S) can be passed if `Q_S` is greater than 1.
        - If `Q_S` or `KV_S` is not 16-byte or 32-byte aligned, it can be rounded up to the aligned value. For details about the constraints, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint).
    - When `sparse_mode` is `2`, `3`, or `4`, the input shape can be `(2048, 2048)`, `(1, 2048, 2048)`, or `(1, 1, 2048, 2048)`.
    - When `sparse_mode` is `9`:
        - If `input_layout` is `BSH`, `BSND`, or `BNSD`, the input shape can be `(B, Q_S, Q_S)`.
        - If `input_layout` is `TND`, the input shape can be a 1D tensor `(∑Q_Si²,)`, where the total length equals the sum of the squared query sequence lengths of all batches. That is, the individual matrix mask of each batch is concatenated into a single 1D tensor.
- **`actual_seq_qlen`** (`List[int]`): Optional. Valid sequence length (`seqlen`) of `query` in different batches. The data type can be `int64`. The default value is `None`, indicating that the value is identical to the sequence length `S` in the `query` shape.
    The valid `seqlen` of each batch in this parameter must not exceed that of the corresponding batch in `query`. If the input length of `seqlen` is `1`, all batches use the same `seqlen`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlen` are used. Other lengths are not supported. When `input_layout` of `query` is `TND`, this parameter is required, and the number of elements in this parameter is used as the batch size. The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element, and cannot be negative.

- **`actual_seq_kvlen`** (`List[int]`): Optional. Valid sequence length of `key` and `value` across different batches. The data type can be `int64`. The default value is `None`, indicating that the value is identical to the sequence length `S` in the `key` and `value` shapes. Different `Q_S` values are subject to different constraints. For details, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint) and [`Q_S = 1` (Incremental Inference) Constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint).
- **`block_table`** (`Tensor`): Optional. Block mapping table used for KV storage in page attention. The data type can be `int32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used.
- **`dequant_scale_query`** (`Tensor`): Optional. Dequantization parameter of `query`, supporting only `pertoken + perhead` mode. The data type can be `float32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
- **`dequant_scale_key`** (`Tensor`): Optional. Dequantization factor of `key` when the KV fake-quantization parameters are separated. The data type can be `float16`, `bfloat16`, or `float32`. The data layout can be ND. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with scale managed by page attention, and `pertoken` + `perhead` with scale managed by page attention. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [`Q_S = 1` (Incremental Inference) Constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint), [GQA Fake-Quantization + KV NZ Format Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
    
- **`dequant_offset_key`** (`Tensor`): Optional. Dequantization offset of `key` when the KV fake-quantization parameters are separated. The data type can be `float16`, `bfloat16`, or `float32`. The data layout can be ND. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with offset managed by page attention, and `pertoken` + `perhead` with offset managed by page attention. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [`Q_S = 1` (Incremental Inference) Constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint), [GQA Fake-Quantization + KV NZ Format Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
- **`dequant_scale_value`** (`Tensor`): Optional. Dequantization factor of `value` when the KV fake-quantization parameters are separated. The data type can be `float16`, `bfloat16`, or `float32`. The data layout can be ND. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with scale managed by page attention, and `pertoken` + `perhead` with scale managed by page attention. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [`Q_S = 1` (Incremental Inference) Constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint), [GQA Fake-Quantization + KV NZ Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
    
- **`dequant_offset_value`** (`Tensor`): Optional. Dequantization offset of `value` when the KV fake-quantization parameters are separated. The data type can be `float16`, `bfloat16`, or `float32`. The data layout can be ND. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with offset managed by page attention, and `pertoken` + `perhead` with offset managed by page attention. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [`Q_S = 1` (Incremental Inference) Constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint), [GQA Fake-Quantization + KV NZ Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
- **`dequant_scale_key_rope`** (`Tensor`): Optional. **Reserved parameter, currently unused. Retain the default value.**
- **`quant_scale_out`** (`Tensor`): Optional. Quantization factor for the output. The data type can be `float32` or `bfloat16`. The data layout can be ND. `pertensor` and `perchannel` modes are supported. When the input data type is `bfloat16`, both `float32` and `bfloat16` are supported. Otherwise, only `float32` is supported. In `perchannel` mode, if the output layout is `BSH`, the product of all dimensions of `quant_scale_out` must be identical to `H`. For other layouts, the product must equal `Q_N * D`. (You are advised to pass a shape of `(1, 1, H)` or `(H,)` when the output layout is `BSH`; `(1, Q_N, 1, D)` or `(Q_N, D)` when the output layout is `BNSD`; and `(1, 1, Q_N, D)` or `(Q_N, D)` when the output layout is `BSND`). This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [General Constraints](#en-us_topic_0000001832267082_section_general_constraint).
- **`quant_offset_out`** (`Tensor`): Optional. Quantization offset for the output. The data type can be `float32` or `bfloat16`. The data layout can be ND. `pertensor` and `perchannel` modes are supported. If `quant_offset_out` is provided, its data type and shape information must be identical to those of `quant_scale_out`. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [General Constraints](#en-us_topic_0000001832267082_section_general_constraint).
- **`quant_scale_p`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`learnable_sink`** (`Tensor`): Optional. Learnable sink token used to absorb attention scores. The data type can be `bfloat16`. The data layout can be ND, and the shape must be `(Q_N,)`. The default value is `None`. For details about the constraints, see [`learnable_sink` Constraints](#en-us_topic_0000001832267082_section_learnable_sink_constraint).

- **`num_query_heads`** (`int`): Optional. Number of heads for `query`. The data type can be `int64`. In `BNSD` scenarios, this value must be identical to the shape of the N axis of `query`. Otherwise, an execution exception occurs. For details about the comprehensive constraints, see [GQA Fake-Quantization + KV NZ Format Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint).
- **`num_key_value_heads`** (`int`): Optional. Number of heads for `key` and `value`, used to support Grouped-Query Attention (GQA) scenarios. The data type can be `int64`. The default value is `0`, indicating that the number of heads for `key`, `value`, and `query` are identical. The value of `num_query_heads` must be divisible by `num_key_value_heads`, and the ratio of `num_query_heads` to `num_key_value_heads` cannot exceed 64. In `BSND`, `BNSD`, and `BNSD_BSND` (supported only when `Q_S` is greater than 1) scenarios, this value must also be identical to the shape of the N axis of `key` and `value`. Otherwise, an execution exception occurs. For details about the comprehensive constraints, see [GQA Fake-Quantization + KV NZ Format Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint).
- **`softmax_scale`** (`float`): Optional. Scaling factor, the reciprocal of the square root of `d` in the formula, used as a scalar value for Muls in the computation flow. The data type can be `float32`. Its data type and the data type of `query` must meet the type deduction rules. The default value is `1.0`, indicating that no scaling is performed. **Provide a value that equals `1/√D` (where `D` represents the head dimension)**. For example, if `D` is `128`, provide `1/math.sqrt(128.0)` to yield the correct attention computation result.
- **`pre_tokens`** (`int`): Optional. Used for sparse computation, indicating the number of preceding tokens with which the attention is associated. The data type can be `int64`. The default value is `2147483647`. This parameter is invalid when `Q_S` is `1`.
- **`next_tokens`** (`int`): Optional. Used for sparse computation, indicating the number of subsequent tokens with which the attention is associated. The data type can be `int64`. The default value is `2147483647`. This parameter is invalid when `Q_S` is `1`.
- **`input_layout`** (`str`): Optional. Data layout of the input `query`, `key`, and `value`. The default value is "BSH".

    > [!NOTE]   
    > When a layout format contains an underscore (_), the portion to the left of the underscore represents the layout of the input `query`, and the portion to the right represents the layout of the output. The operator performs layout conversion internally.

    Supported layouts: `BSH`, `BSND`, `BNSD`, `BNSD_BSND` (when the input is `BNSD`, the output format is `BSND`; supported only when `Q_S` is greater than 1), `BSH_NBSD`, `BSND_NBSD`, `BNSD_NBSD` (when the output format is `NBSD`, supported only when `Q_S` is greater than 1 and less than or equal to 16), `TND`, `TND_NTD`, and `NTD_TND`. For details about the comprehensive constraints on `TND`-related scenarios, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint). `BNSD_BSND` indicates that the input is `BNSD` and the output format is `BSND`. It is supported only when `Q_S` is greater than 1.

    | `input_layout` | `query` shape | `key` shape | `value` shape | Output (`attention_out`) Shape| Description|
    |---------------|-------------|-----------|-------------|------------|------|
    | BSH | (B, Q\_S, H) | (B, KV\_S, H) | (B, KV\_S, H) | (B, Q\_S, H) | H=N\*D |
    | BSND | (B, Q\_S, Q\_N, D) | (B, KV\_S, KV\_N, D) | (B, KV\_S, KV\_N, D) | (B, Q\_S, Q\_N, D) | $N$ and $D$ dimensions are separated.|
    | BNSD | (B, Q\_N, Q\_S, D) | (B, KV\_N, KV\_S, D) | (B, KV\_N, KV\_S, D) | (B, Q\_N, Q\_S, D) | $N$ and $D$ dimensions are separated, with $N$ in front.|
    | BNSD\_BSND | (B, Q\_N, Q\_S, D) | (B, KV\_N, KV\_S, D) | (B, KV\_N, KV\_S, D) | (B, Q\_S, Q\_N, D) | Input layout is `BNSD` and output layout is `BSND`; supported only when $Q\_S > 1$.|
    | BSH\_NBSD | (B, Q\_S, H) | (B, KV\_S, H) | (B, KV\_S, H) | (Q\_N, B, Q\_S, D) | Input layout is `BSH` and output layout is `NBSD`.|
    | BSND\_NBSD | (B, Q\_S, Q\_N, D) | (B, KV\_S, KV\_N, D) | (B, KV\_S, KV\_N, D) | (Q\_N, B, Q\_S, D) | Input layout is `BSND` and output layout is `NBSD`.|
    | BNSD\_NBSD | (B, Q\_N, Q\_S, D) | (B, KV\_N, KV\_S, D) | (B, KV\_N, KV\_S, D) | (Q\_N, B, Q\_S, D) | Input layout is `BNSD` and output layout is `NBSD`; supported only when $Q\_S$ ranges from 1 to 16.|
    | TND | (T, Q\_N, D) | (T, KV\_N, D) | (T, KV\_N, D) | (T, Q\_N, D) | $T$ represents the cumulative sequence length sum across all batches.|
    | TND\_NTD | (T, Q\_N, D) | (T, KV\_N, D) | (T, KV\_N, D) | (Q\_N, T, D) | Input layout is `TND` and output layout is `NTD`.|
    | NTD\_TND | (Q\_N, T, D) | (KV\_N, T, D) | (KV\_N, T, D) | (T, Q\_N, D) | Input layout is `NTD` and output layout is `TND`.|

- **`sparse_mode`** (`int`): Optional. Sparse attention mode. The default value is `0`. The data type can be `int64`. This parameter is invalid when `Q_S` is `1` and no RoPE input is provided. For details about the comprehensive constraints when `input_layout` is `TND`, `TND_NTD`, or `NTD_TND`, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint). In GQA fake-quantization scenarios, see [GQA Fake-Quantization + KV NZ Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint). Currently, only values of `0`, `1`, `2`, `3`, `4`, and `9` are supported; values of `5`, `6`, `7`, and `8` (representing `prefix`, `global`, `dilated`, and `block_local`, respectively) are currently not implemented and must not be used.

    | Value| Mode| Description| Requirement for `atten_mask`|
    |------|----------|------|-----------------|
    | 0 | defaultMask | If `atten_mask` is omitted, no masking is performed, and `pre_tokens` and `next_tokens` are ignored (internally set to `INT_MAX`). If `atten_mask` is provided, a complete `atten_mask` matrix with shape (S1, S2) must be provided, indicating that the region between `pre_tokens` and `next_tokens` is computed.| Optional.|
    | 1 | allMask | A complete `atten_mask` matrix with shape `(S1, S2)` must be provided.| Required. Shape must be `(S1, S2)`.|
    | 2 | leftUpCausal | Causal masking aligned to the upper-left corner.| Optimized `atten_mask` matrix with shape ``(2048, 2048)``.|
    | 3 | rightDownCausal | Causal masking aligned to the lower-right corner, corresponding to the lower-triangular matrix divided by the rightmost vertex.| Optimized `atten_mask` matrix with shape ``(2048, 2048)``.|
    | 4 | band | Band attention masking configuration.| Optimized `atten_mask` matrix with shape ``(2048, 2048)``.|
    | 9 | treeMask | Tree attention mask for speculative decoding scenarios. Supported only in MLA scenarios where `query_rope` and `key_rope` are both provided. Left padding, `pse_shift`, and `sharedPrefix` are not supported. The output data type cannot be `int8`. Each batch must satisfy the condition $Q\_S \le KV\_S$.| A custom tree mask must be provided.|
    
- **`block_size`** (`int`): Optional. Maximum number of tokens in each block of KV storage in page attention. The data type can be `int64`. The default value is 0.
- **`query_quant_mode`** (`int`): Optional. Fake-quantization mode for `query`. Only 3 can be passed, indicating mode 3: `pertoken` + `perhead` mode.
- **`key_quant_mode`** (`int`): Optional. Fake-quantization mode for `key`. The default value is 0. Except for scenarios where `key_quant_mode` is `0` and `value_quant_mode` is `1`, this value must be identical to `value_quant_mode`. For details about the comprehensive constraints, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint) and [`Q_S = 1` (Incremental Inference) Constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint).

    When `Q_S` is greater than or equal to `2`, only values of `0` and `1` are supported. When `Q_S` is `1`, the supported values are `0`, `1`, `2`, `3`, `4`, and `5`.

    - `0`: enables `perchannel` mode (which includes `pertensor`).
    - `1`: enables `pertoken` mode.
    - `2`: enables `pertensor` + `perhead` mode.
    - `3`: enables `pertoken` + `perhead` mode.
    - `4`: enables `pertoken` mode with scale or offset managed by page attention.
    - `5`: enables `pertoken` + `perhead` mode with scale or offset managed by page attention.

- **`value_quant_mode`** (`int`): Optional. Fake-quantization mode for `value`. Mode numbering is identical to that of `key_quant_mode`. The default value is `0`. Except for scenarios where `key_quant_mode` is `0` and `value_quant_mode` is `1`, this value must be identical to `key_quant_mode`. For details about the comprehensive constraints, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [`Q_S = 1` (Incremental Inference) Constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint), and [GQA Fake-Quantization + KV NZ Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint).

    When `Q_S` is greater than or equal to `2`, only values of `0` and `1` are supported. When `Q_S` is `1`, the supported values are `0`, `1`, `2`, `3`, `4`, and `5`.

- **`inner_precise`** (`int`): Optional. The data type can be `int64`. Four modes are supported: `0`, `1`, `2`, and `3`. Each option is represented by a 2-bit value. Bit 0 specifies high-precision or high-performance mode, and bit 1 specifies whether to perform invalid row correction. When `Q_S` is greater than 1, if `sparse_mode` is `0` or `1` and a user-defined mask is provided, enabling invalid row correction is recommended. When `Q_S` is `1`, you can only set `inner_precise` to `0` or `1`. For details about the comprehensive constraints, see [`Q_S > 1` (Full Inference) Constraints](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [`Q_S = 1` (Incremental Inference) Constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).

    - `0`: enables high-precision mode and disables invalid row correction.
    - `1`: enables high-performance mode and disables invalid row correction.
    - `2`: enables high-precision mode and enables invalid row correction.
    - `3`: enables high-performance mode and enables invalid row correction.

    > [!NOTE]   
    > For `bfloat16` and `int8`, no distinction is made between high-precision and high-performance modes. Invalid row correction applies to `float16`, `bfloat16`, and `int8`. Currently, `0` and `1` are reserved values. If any row in the mask involved in computation consists entirely of `1`s, precision may degrade. In this case, you can set this parameter to `2` or `3` to enable invalid row correction and improve precision. However, this configuration may degrade performance.

- **`return_softmax_lse`** (`bool`): Optional. Specifies whether to output `softmax_lse`, supporting S-axis tiling (splitting along the sequence length dimension). `True` outputs `softmax_lse`, and `False` disables it. The default value is `False`.
- **`query_dtype`** (`int`): Optional. Data type of `query`. **Reserved parameter, currently not used. Retain the default value.**
- **`key_dtype`** (`int`): Optional. Data type of `key`. **Reserved parameter, currently not used. Retain the default value.**
- **`value_dtype`** (`int`): Optional. Data type of `value`. **Reserved parameter, currently not used. Retain the default value.**
- **`query_rope_dtype`** (`int`): Optional. Data type of `query_rope`. **Reserved parameter, currently not used. Retain the default value.**
- **`key_rope_dtype`** (`int`): Optional. Data type of `key_rope`. **Reserved parameter, currently not used. Retain the default value.**
- **`key_shared_prefix_dtype`** (`int`): Optional. Data type of `key_shared_prefix`. **Reserved parameter, currently not used. Retain the default value.**
- **`value_shared_prefix_dtype`** (`int`): Optional. Data type of `value_shared_prefix`. **Reserved parameter, currently not used. Retain the default value.**
- **`dequant_scale_query_dtype`** (`int`): Optional. Data type of `dequant_scale_query`. **Reserved parameter, currently not used. Retain the default value.**
- **`dequant_scale_key_dtype`** (`int`): Optional. Data type of `dequant_scale_key`. **Reserved parameter, currently not used. Retain the default value.**
- **`dequant_scale_value_dtype`** (`int`): Optional. Data type of `dequant_scale_value`. **Reserved parameter, currently not used. Retain the default value.**
- **`dequant_scale_key_rope_dtype`** (`int`): Optional. Data type of `dequant_scale_key_rope`. **Reserved parameter, currently not used. Retain the default value.**
- **`out_dtype`** (`int`): Optional. Output data type. When the input is `int8` or `float8_e4m3fn`, this parameter can be configured to specify the output data type (such as `float8_e5m2`). This parameter can be set to `None` if this feature is not used.

## Return Values<a name="en-us_topic_0000001832267082_section22231435517"></a>

- **`attention_out`** (`Tensor`): Output tensor in the formula. The data type can be `float16`, `bfloat16`, or `int8`. The data layout can be ND. Restriction: The `D` dimension of this parameter must be identical to that of `value`, and all other dimensions must match those of the input parameter `query`.
- **`softmax_lse`** (`Tensor`): log-sum-exp result computed over the Query-Key product by using the Ring Attention algorithm. It is obtained by first computing `query` * `key`, subtracting the row-wise maximum (`softmax_max`), applying `exp`, summing the results (`softmax_sum`), taking the natural logarithm of `softmax_sum`, and finally adding `softmax_max`. The data type is `float32`. When `return_softmax_lse` is `True`, the output shape is `(B, Q_N, Q_S, 1)` under standard layouts, and `(T, Q_N, 1)` when `input_layout` is `TND` or `NTD_TND`. When `return_softmax_lse` is `False`, the output is a tensor of shape `[1]` with value `0`.

## Constraints<a name="en-us_topic_0000001832267082_section12345537164214"></a>

> [!NOTICE]
> Constraints are organized by scenario and can be referenced as needed:
>
> - [**General Constraints**](#en-us_topic_0000001832267082_section_general_constraint): empty input handling, `key` and `value` shape consistency, and `int8` quantization.
> - [**MLA Constraints**](#en-us_topic_0000001832267082_section_mla_constraint) (when `query_rope` and `key_rope` are provided): `D = 512`, `D = 128`, and `TND`.
> - [**GQA Fake-Quantization + KV NZ Constraints**](#en-us_topic_0000001832267082_section_gqa_nz_constraint): `KV` NZ input layouts, `dequant_scale`, `sparse_mode`, and `num_query_heads`/`num_key_value_heads` combinations.
> - [**`learnable_sink` Constraints**](#en-us_topic_0000001832267082_section_learnable_sink_constraint): usage of `learnable_sink`.
> - [**`Q_S > 1` (Full Inference) Constraints**](#en-us_topic_0000001832267082_section_qs_gt1_constraint): input shapes, `sparse_mode`, page attention, quantization, `pse_shift`, and KV fake-quantization parameter separation.
> - [**`Q_S = 1` (Incremental Inference) Constraints**](#en-us_topic_0000001832267082_section_qs_eq1_constraint): input shapes, page attention, and KV fake-quantization parameter separation.

### General Constraints<a name="en-us_topic_0000001832267082_section_general_constraint"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Empty input handling: The operator checks whether `query` is empty internally. If `query` is empty, an empty value is returned. If `query` is a non-empty tensor but `key` and `value` are empty tensors (`S2` is `0`), `attention_out` returns a tensor filled with all `0`s matching its corresponding shape. If `attention_out` evaluates to an empty tensor, an empty value is returned.
- The corresponding tensors of `key` and `value` must have identical shapes. In non-contiguous scenarios, each tensor in the tensorlist of `key` and `value` must have a batch size of 1. The number of tensors must be equal to `B` in `query`, and the `N` and `D` dimensions must be identical.
- Constraints on the number of input parameters and output data formats related to INT8 quantization:
    - For `int8` output: The input parameter `quant_scale_out` must be provided. `quant_offset_out` is optional and defaults to `0` if not specified.
        - <term>Atlas A2 training products/Atlas A2 inference products</term>: The input data type is `int8`.
        - <term>Atlas A3 training products/Atlas A3 inference products</term>: The input data type is `int8`.

    - For `float16` output: If the input parameter `quant_offset_out` or `quant_scale_out` is provided (not `None`), an error is raised and execution returns.
    - `quant_offset_out` and `quant_scale_out` support `pertensor` or `perchannel` modes, and their data types can be `float32` or `bfloat16`.

### MLA Constraints<a name="en-us_topic_0000001832267082_section_mla_constraint"></a>

- When `query_rope` and `key_rope` are provided (MLA scenario), the following constraints apply:
    - The data type and data format of `query_rope` must be identical to those of `query`.
    - The data type and data format of `key_rope` must be identical to those of `key`.
    - `query_rope` and `key_rope` must either both be omitted or both be provided. Partial configuration is not supported.
    - When `query_rope` and `key_rope` are non-empty, the `D` dimension of `query` must be `512` or `128`.
        - When `D` of `query` is `512`:
            - `sparse_mode`: Valid values: `0`, `3`, `4`, or `9`.
            - When `query_rope` is configured, the valid values of the `N` dimension of `query` are `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`. The `D` dimension of `query_rope` must be `64`, and all other dimensions must match those of `query`.
            - When `key_rope` is configured: the `N` dimension must be `1` and the `D` dimension must be `512` for `key`. The `D` dimension of `key_rope` must be `64`, and all other dimensions must match those of `key`.
            - Supported data layouts for `key`, `value`, and `key_rope`: ND or NZ. When the data layout is NZ, the layout of the input parameters `key` and `value` is `[blockNum, KV_N, D/16, blockSize, 16]` for the `float16` or `bfloat16` data type, and `[blockNum, KV_N, D/32, blockSize, 32]` for the `int8` input data type.
            - Supported `input_layout` values: `BSH`, `BSND`, `BNSD`, `BNSD_NBSD`, `BSND_NBSD`, `BSH_NBSD`, `TND`, and `TND_NTD`.
            - Page attention is supported: `block_size` must be divisible by 16 and cannot exceed 1024.
            - Left padding, tensorlist, PSE, prefix, fake-quantization, post-quantization, and empty tensors are not supported.
            - Full quantization is supported (scenarios where the data types of the input parameters `query`, `key`, and `value` are entirely `int8`, `query_rope`, and `key_rope` are `bfloat16`, and the output data type is `bfloat16`):
                - The parameters `dequant_scale_query`, `dequant_scale_key`, and `dequant_scale_value` must all be provided, and their data types must be `float32`.
                - Providing `quant_scale_out`, `quant_offset_out`, `dequant_offset_key`, or `dequant_offset_value` is not supported. Otherwise, an error is raised and execution returns.
                - `query_quant_mode` supports only `pertoken + perhead` mode, and `key_quant_mode` and `value_quant_mode` support only `pertensor` mode.
                - Supported data layout for `key`, `value`, and `key_rope`: NZ.
        - When `D` of `query` is `128`:
            - Supported `input_layout` values: `BSH`, `BSND`, `TND`, `BNSD`, `NTD`, `BSH_BNSD`, `BSND_BNSD`, `BNSD_BSND`, and `NTD_TND`.   
            - When `query_rope` is configured: The `D` dimension of `query_rope` must be `64`, and all other dimensions must match those of `query`. 
            - When `key_rope` is configured: The `D` dimension of `key_rope` must be `64`, and all other dimensions must match those of `key`. 
            - Left padding, tensorlist, PSE, prefix, fake-quantization, full quantization, post-quantization, and empty tensors are not supported.
            - Other constraints are the same as those when the layout is `TND` or `NTD_TND`.

    - Constraints on the input parameters `query`, `key`, and `value` in `TND`, `TND_NTD`, and `NTD_TND` layouts:
        - `actual_seq_qlen` and `actual_seq_kvlen` must both be provided, and the number of elements in these parameters is used as the batch size (the number of elements must be less than or equal to 4096). The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element.
        - When `D` of `query` is `512`:
            - `sparse_mode`: Valid values: `0`, `3`, `4`, or `9`.
            - `TND` and `TND_NTD` are supported.
            - Page attention can be enabled. In this case, the length of `actual_seq_kvlen` must be identical to the batch size of `key` or `value`, representing the actual sequence length of each batch, and each value must not exceed `KV_S`.
            - The `N` dimension of `query` must be `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`, while the `N` dimension of `key` and `value` must be `1`.
            - `query_rope` and `key_rope` must both be provided, and their `D` dimension must be `64`.
            - Left padding, tensorlist, PSE, prefix, fake-quantization, full quantization, post-quantization, and empty tensors are not supported.

        - When `D` of `query` is not `512`:
            - When `query_rope` and `key_rope` are omitted: in `TND` layouts, `Q_D` (`D` dimension of `query`), `K_D` (`D` dimension of `key`), and `V_D` (`D` dimension of `value`) must all be `128`, or `Q_D` and `K_D` must be `192` with `V_D` equal to `128` or `192`; in `NTD` layouts, `V_D` must not be `192`; and in `NTD_TND` layouts, `Q_D` and `K_D` must be `128` or `192`, and `V_D` must be `128`. When `query_rope` and `key_rope` are provided, `Q_D`, `K_D`, and `V_D` must all be `128`. In GQA and page attention scenarios, `V_D` must not be `192`. In MHA scenarios, `Q_D`, `K_D`, and `V_D` must all be `64` or all be `128`, or when `Q_D` and `K_D` are `192`, `V_D` must be `128`.
            - `TND`, `NTD`, and `NTD_TND` layouts are supported.
            - In page attention scenarios, `block_size` must be a multiple of 16 and less than or equal to 1024.
            - In MHA scenarios, only `float16` and `bfloat16` data types are supported. When the data type is `float16`, `inner_precise` supports only values of `0` and `1`. When the data type is `bfloat16`, `inner_precise` supports only a value of `0`. When `sparse_mode` is `0`, `atten_mask` must not be provided. When `sparse_mode` is `3` or `4`, an optimized `atten_mask` matrix must be provided. Page attention supports only the `BnBsH` format, where the KV cache layout is `(blockNum, blocksize, H)`, with `blockNum` indicating the number of blocks, `blocksize` the number of tokens per block, and `H` the hidden layer size.
            - Left padding, tensorlist, PSE, prefix, fake quantization, and full quantization are not supported.

### GQA Fake-Quantization + KV NZ Constraints<a name="en-us_topic_0000001832267082_section_gqa_nz_constraint"></a>

- Constraints on parameters in GQA fake-quantization scenarios when `key` and `value` are in `NZ` format:
    - `perchannel` and `pertoken` modes are supported. The data type of `query` is fixed to `bfloat16`, and the data types of `key` and `value` are fixed to `int8`. The `D` dimension of `query`, `key`, and `value` must be `128`. The sequence length (`S`) of `query` must be 1 to 16.
    - `input_layout` must be `BSH`, `BSND`, and `BNSD`.
    - Only `page_attention` scenarios are supported, where `blockSize` supports only `128` or `512`.
    - `key` and `value` support only `NZ` layout inputs with shape `[blockNum, KV_N, D/32, blockSize, 32]`.
    - Data type requirements for `dequant_scale_key` and `dequant_scale_value`: in `perchannel` mode, only `bfloat16` is supported; in `pertoken` mode, only `float32` is supported.
    - Shape requirements for `dequant_scale_key` and `dequant_scale_value`: in `perchannel` mode, when `input_layout` is `BSH`, the shape must be `[H]`, when `input_layout` is `BNSD`, the shape must be `[KV_N, 1, D]`, and when the output layout is `BSND`, the shape must be `[KV_N, D]`; in `pertoken` mode, the shape must be `[B, KV_S]`, where `S` must be greater than or equal to the product of the second dimension of `block_table` and `block_size`.
    - Only KV separation is supported.
    - Only high-performance mode is supported.
    - When MTP is `0`, `sparse_mode` can be `0`, and `atten_mask` must not be provided. When MTP is greater than 0 and less than 16, `sparse_mode` can be `3`, and an optimized `atten_mask` matrix must be provided with shape `(2048, 2048)`, or `sparse_mode` can be `9`, requiring a tree mask where the recommended shape is `(B, Q_S, Q_S)` for the `BSH`, `BSND`, or `BNSD` layouts, and a 1D tensor shape `(∑Q_Si²,)` for the `TND` layout where the total length equals the sum of the squared query sequence lengths of all batches.
    - Configuring `dequant_offset_key` and `dequant_offset_value` is not supported.
    - Configuring `query_rope` and `key_rope` is not supported.
    - Left padding, tensorlist, PSE, prefix, and post-quantization are not supported.
    - Supported combinations of (`num_query_heads`, `num_key_value_heads`) are `(10, 1)`, `(64, 8)`, `(80, 8)`, and `(128, 16)`.

### `learnable_sink` Constraints<a name="en-us_topic_0000001832267082_section_learnable_sink_constraint"></a>

- Constraints on `learnable_sink`:
    - Only `TND` and `NTD_TND` layouts are supported.
    - The `D` dimension of `value` must be less than or equal to `128`.
    - Only non-quantization scenarios are supported.
    - PSE, left padding, shared prefix, and post-quantization are not supported.

### `Q_S > 1` (Full Inference) Constraints<a name="en-us_topic_0000001832267082_section_qs_gt1_constraint"></a>

- **When `Q_S` is greater than 1**:
    - Usage constraints for `query`, `key`, and `value` inputs:
        - The `B` dimension must be less than or equal to 65536. If the `D` dimension is not 32-byte aligned, only values up to 128 are supported.
        - The `N` dimension must be less than or equal to 256, and the `D` dimension must be less than or equal to 512. When `input_layout` is `BSH` or `BSND`, the product of `N` and `D` should be less than 65535.
        - The `S` dimension must be less than or equal to 20971520 (20M). In certain long-sequence scenarios, excessive computation may cause the PFA operator to time out (raising an aicore error with an `errorStr` of "timeout or trap error"). In such cases, slicing along the `S` dimension is recommended. (Note: The computation volume depends on `B`, `S`, `N`, and `D`. Larger values result in higher computation cost). Typical timeout-prone long-sequence scenarios (where the product of `B`, `S`, `N`, and `D` is large) include but are not limited to:
            - `B = 1`, `Q_N = 20`, `Q_S = 2097152`, `D = 256`, `KV_N = 1`, `KV_S = 2097152`
            - `B = 1`, `Q_N = 2`, `Q_S = 20971520`, `D = 256`, `KV_N = 2`, `KV_S = 20971520`
            - `B = 20`, `Q_N = 1`, `Q_S = 2097152`, `D = 256`, `KV_N = 1`, `KV_S = 2097152`
            - `B = 1`, `Q_N = 10`, `Q_S = 2097152`, `D = 512`, `KV_N = 1`, `KV_S = 2097152`

        - When the data types of `query`, `key`, or `value` include `int8`, the `D` dimension must be a multiple of 32. When the data types are all `float16` or `bfloat16`, the `D` dimension must be a multiple of 16.
        - Constraints on the `D` dimension:
            - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: When the data types of `query`, `key`, or `value` include `int8`, the `D` dimension must be a multiple of 32. When the data types of `query`, `key`, `value`, or `attentionOut` include `int4`, the `D` dimension must be a multiple of 64. When the data types are all `float16` or `bfloat16`, the `D` dimension must be a multiple of 16.

    - Constraints on `actual_seq_qlen`:
    
        <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The valid sequence length of each batch in this parameter must be less than or equal to the sequence length of the corresponding batch in `query`. If the input length of `seqlen` is `1`, all batches use the same `seqlen`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlen` are used. Other lengths are not supported. For details about the comprehensive constraints when `input_layout` of `query` is `TND` or `NTD_TND`, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
        
    - Constraints on `actual_seq_kvlen`:
    
        <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The valid sequence length of each batch in this parameter must be less than or equal to the sequence length of the corresponding batch in `key` and `value`. If the length of this parameter is 1, all batches use the same sequence length. If the length is greater than or equal to the batch size, only the first `batch_size` elements are used. Other lengths are not supported. For details about the comprehensive constraints when the `input_layout` of `key` and `value` is `TND` or `NTD_TND`, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
        
    - The `sparse_mode` parameter supports only values of `0`, `1`, `2`, `3`, `4`, and `9`. Other values result in an error.

        - When `sparse_mode` is `0`, if `atten_mask` is `None`, the input parameters `pre_tokens` and `next_tokens` are ignored and internally set to `INT_MAX`.
        - When `sparse_mode` is `2`, `3`, or `4`, the shape of `atten_mask` must be `(S, S)`, `(1, S, S)`, or `(1, 1, S, S)`, where `S` must be `2048`. The provided `atten_mask` must be a lower triangular matrix. If `atten_mask` is omitted or has an invalid shape, an error is raised.
        - When `sparse_mode` is `1`, `2`, or `3`, the input parameters `pre_tokens` and `next_tokens` are ignored and assigned based on the relevant rules.
        - When `sparse_mode` is `9`, this mode is supported only in MLA scenarios where `query_rope` and `key_rope` are both provided. `atten_mask` must not be `None`. When `input_layout` is `BSH`, `BSND`, or `BNSD`, the shape must be `(B, Q_S, Q_S)`; when `input_layout` is `TND`, the shape must be a 1D tensor `(∑Q_Si²,)` where the total length equals the sum of the squared query sequence lengths of all batches. Left padding, `pse_shift`, and `sharedPrefix` are not supported.

    - Page attention scenarios:
        - Page attention can be enabled only when `block_table` exists and is valid, and `key` and `value` are arranged in contiguous memory based on the indices in `block_table`. The supported data types for `key` and `value` are `float16` and `bfloat16`. In this case, `input_layout` of `key` and `value` is ignored. The `block_table` contains block IDs. Their validity is not verified and must be ensured by the user.
        - `block_size` is a user-defined parameter that affects page attention performance. When page attention is enabled, `block_size` must range from 128 to 512 and must be a multiple of 128. Generally, page attention improves throughput but reduces performance.
    
        - In page attention scenarios, when the input KV cache layout is `(blocknum, blocksize, H)` and the product of `KV_N` and `D` is greater than 65535, execution is blocked and an error is raised due to hardware constraints. This can be resolved by enabling GQA (to reduce `KV_N`) or by changing the KV cache layout to `(blocknum, KV_N, blocksize, D)`. When the `input_layout` of `query` is `BNSD` or `TND`, the KV cache supports both `(blocknum, blocksize, H)` and `(blocknum, KV_N, blocksize, D)` layouts. When it is `BSH` or `BSND`, only `(blocknum, blocksize, H)` is supported. The value of `blocknum` must not be less than the total number of blocks required across all batches, calculated from `actual_seq_kvlen` and `block_size`. The shapes of `key` and `value` must be identical.
        - Page attention does not support fake quantization or tensorlist.
        - In page attention scenarios, `actual_seq_kvlen` must be provided.
        - In page attention scenarios, `block_table` must be a 2D tensor. Its first dimension must equal `B`, and its second dimension must be no less than `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actual_seq_kvlen` across batches).
        - In page attention scenarios, two layout formats are supported, and the supported data types are `float32` and `bfloat16`. Cases where `query` is `int8` are not supported.
        - When page attention is enabled, the input sequence length `KV_S` must be greater than or equal to the product of `maxBlockNumPerSeq` and `block_size` in the following scenarios:
            - When `atten_mask` is provided and its shape is identical to `(B, 1, Q_S, KV_S)`.
            - When `pse_shift` is provided and its shape is identical to `(B, Q_N, Q_S, KV_S)`.

    - The input parameters `quant_scale_out` and `quant_offset_out` support `pertensor` or `perchannel` quantization, and their data types can be `float32` or `bfloat16`. If `quant_offset_out` is provided, its data type and shape must be identical to those of `quant_scale_out`. When the input data type is `bfloat16`, both `float32` and `bfloat16` are supported. Otherwise, only `float32` is supported. In `perchannel` scenarios, when the output layout is `BSH`, the product of all dimensions of `quant_scale_out` must equal `H`. For other layouts, the product must equal `Q_N * D`. The recommended shape for `quant_scale_out` is `(1, 1, H)` or `(H,)` for the `BSH` output layout, `(1, Q_N, 1, D)` or `(Q_N, D)` for the `BNSD` output layout, and `(1, 1, Q_N, D)` or `(Q_N, D)` for the `BSND` output layout.
    - When the output data type is `int8` and `quant_scale_out` and `quant_offset_out` are in `perchannel` mode, left padding, Ring Attention, and a non-32-byte-aligned `D` dimension are not supported.
    - When the output data type is `int8`, scenarios where `sparse_mode` is `band` and `pre_tokens` or `next_tokens` are negative are not supported.
    - Usage constraints for the `pse_shift` feature:
        
        - Supported when the data type of `query` is `float16`, `bfloat16`, or `int8`.
        - When `query`, `key`, and `value` are all `float16` and `pse_shift` is provided, high-precision mode is enforced. The constraints are the same as those of high-precision mode.
        - `Q_S` must be greater than or equal to the sequence length (`S`) of `query`, and `KV_S` must be greater than or equal to the sequence length (`S`) of `key`.

    - When the output data type is `int8`, if `quant_offset_out` is a non-`None` and non-empty tensor, and `sparse_mode`, `pre_tokens`, and `next_tokens` meet the execution blocking conditions, certain matrix rows will be excluded from computation. This causes computation errors, and execution will be blocked:
        - When `sparse_mode` is `0`, if `atten_mask` is not `None` and, for each batch, `actual_seq_qlen-actual_seq_kvlen - pre_tokens > 0`, or `next_tokens < 0`, the execution will be blocked.
        - When `sparse_mode` is `1` or `2`, the execution will not be blocked.
        - When `sparse_mode` is `3`, if, for each batch, `actual_seq_kvlen - actual_seq_qlen < 0`, the execution will be blocked.
        - When `sparse_mode` is `4`, if `pre_tokens < 0` or, for each batch, `next_tokens + actual_seq_kvlen - actual_seq_qlen < 0`, the execution will be blocked.

    - Constraints on KV fake-quantization parameter separation:
        - When both fake-quantization parameters and KV-separated quantization parameters are provided, the KV-separated quantization parameters take precedence.    
        - `key_quant_mode` and `value_quant_mode` must have identical values.
        - `dequant_scale_key` and `dequant_scale_value` must either both be omitted or both be provided. Likewise, `dequant_offset_key` and `dequant_offset_value` must either both be omitted or both be provided.
        - When `dequant_scale_key` and `dequant_scale_value` are both provided, their shapes must be identical. When `dequant_offset_key` and `dequant_offset_value` are both provided, their shapes must be identical. 
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:  
            - Only `pertoken` and `perchannel` modes are supported. In `pertoken` mode, the shapes of both parameters must be `(B, KV_S)`, and their data types must be `float32`. In `perchannel` mode, the shapes of both parameters must be `(KV_N, D)` or `(H)`, and their data types must be `bfloat16`, where `H` equals the product of `KV_N` and `D`.
            - When `dequant_scale_key` and `dequant_scale_value` are both provided, the sequence length `S` of `query` must be less than or equal to 16. The data type of `query` must be `bfloat16`, the data types of `key` and `value` must be `int8`, and the output data type must be `bfloat16`. This configuration does not support tensorlist or page attention.
        
        - The following table describes quantization modes for scale and offset management.
        
            > [!NOTE]   
            > The scale and offset parameters are `dequant_scale_key`, `dequant_scale_value`, `dequant_offset_key`, and `dequant_offset_value`.

            <a name="en-us_topic_0000001832267082_table3276159203213"></a>
            <table><thead align="left"><tr id="en-us_topic_0000001832267082_row192767598320"><th class="cellrowborder" valign="top" width="16.950000000000003%" id="mcps1.1.5.1.1"><p id="en-us_topic_0000001832267082_p19276135910323"><a name="en-us_topic_0000001832267082_p19276135910323"></a><a name="en-us_topic_0000001832267082_p19276135910323"></a>Quantization Mode</p>
            </th>
            <th class="cellrowborder" valign="top" width="23.09%" id="mcps1.1.5.1.2"><p id="en-us_topic_0000001832267082_p1627615594327"><a name="en-us_topic_0000001832267082_p1627615594327"></a><a name="en-us_topic_0000001832267082_p1627615594327"></a>Scale and Offset Criteria</p>
            </th>
            <th class="cellrowborder" valign="top" width="46.660000000000004%" id="mcps1.1.5.1.3"><p id="en-us_topic_0000001832267082_p17276195963213"><a name="en-us_topic_0000001832267082_p17276195963213"></a><a name="en-us_topic_0000001832267082_p17276195963213"></a>Key and Value Criteria</p>
            </th>
            <th class="cellrowborder" valign="top" width="13.3%" id="mcps1.1.5.1.4"><p id="en-us_topic_0000001832267082_p227695933219"><a name="en-us_topic_0000001832267082_p227695933219"></a><a name="en-us_topic_0000001832267082_p227695933219"></a>Supported Products</p>
            </th>
            </tr>
            </thead>
            <tbody><tr id="en-us_topic_0000001832267082_row172761159123213"><td class="cellrowborder" valign="top" width="16.950000000000003%" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p14277165915327"><a name="en-us_topic_0000001832267082_p14277165915327"></a><a name="en-us_topic_0000001832267082_p14277165915327"></a>perchannel</p>
            </td>
            <td class="cellrowborder" valign="top" width="23.09%" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p113847501392"><a name="en-us_topic_0000001832267082_p113847501392"></a><a name="en-us_topic_0000001832267082_p113847501392"></a>The shapes of these parameters can be <code>(KV_N, 1, D)</code>, <code>(KV_N, D)</code>, <code>(H,)</code>, <code>(1, KV_N, 1, D)</code>, <code>(1, KV_N, D)</code> and <code>(1, H)</code>. Their data types must match that of <code>query</code>.</p>
            </td>
            <td class="cellrowborder" valign="top" width="46.660000000000004%" headers="mcps1.1.5.1.3 "><a name="en-us_topic_0000001832267082_ul2277759183216"></a><a name="en-us_topic_0000001832267082_ul2277759183216"></a><ul id="en-us_topic_0000001832267082_ul2277759183216"><li><span id="en-us_topic_0000001832267082_ph112776597327"><a name="en-us_topic_0000001832267082_ph112776597327"></a><a name="en-us_topic_0000001832267082_ph112776597327"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_19"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_19"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_19"></a>Atlas A2 training products/Atlas A2 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int8</code>. </li><li><span id="en-us_topic_0000001832267082_ph327717592325"><a name="en-us_topic_0000001832267082_ph327717592325"></a><a name="en-us_topic_0000001832267082_ph327717592325"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_19"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_19"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_19"></a>Atlas A3 training products/Atlas A3 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int8</code></li></ul>
            </td>
            <td class="cellrowborder" rowspan="2" valign="top" width="13.3%" headers="mcps1.1.5.1.4 "><a name="en-us_topic_0000001832267082_ul7277159143219"></a><a name="en-us_topic_0000001832267082_ul7277159143219"></a><ul id="en-us_topic_0000001832267082_ul7277159143219"><li><span id="en-us_topic_0000001832267082_ph527775920323"><a name="en-us_topic_0000001832267082_ph527775920323"></a><a name="en-us_topic_0000001832267082_ph527775920323"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_20"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_20"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_20"></a>Atlas A2 training products/Atlas A2 inference products</term></span></li><li><span id="en-us_topic_0000001832267082_ph22772059103210"><a name="en-us_topic_0000001832267082_ph22772059103210"></a><a name="en-us_topic_0000001832267082_ph22772059103210"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_20"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_20"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_20"></a>Atlas A3 training products/Atlas A3 inference products</term></span></li></ul>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row027816595321"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p19936325114616"><a name="en-us_topic_0000001832267082_p19936325114616"></a><a name="en-us_topic_0000001832267082_p19936325114616"></a>pertoken</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p39361225144616"><a name="en-us_topic_0000001832267082_p39361225144616"></a><a name="en-us_topic_0000001832267082_p39361225144616"></a>The shapes of these parameters must be <code>(B, KV_S)</code>, and their data types must be <code>float32</code>.</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="en-us_topic_0000001832267082_ul7475108135214"></a><a name="en-us_topic_0000001832267082_ul7475108135214"></a><ul id="en-us_topic_0000001832267082_ul7475108135214"><li><span id="en-us_topic_0000001832267082_ph1947698135214"><a name="en-us_topic_0000001832267082_ph1947698135214"></a><a name="en-us_topic_0000001832267082_ph1947698135214"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_21"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_21"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_21"></a>Atlas A2 training products/Atlas A2 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int8</code>. </li><li><span id="en-us_topic_0000001832267082_ph12476287523"><a name="en-us_topic_0000001832267082_ph12476287523"></a><a name="en-us_topic_0000001832267082_ph12476287523"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_21"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_21"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_21"></a>Atlas A3 training products/Atlas A3 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int8</code>.</li></ul>
            </td>
            </tr>
            </tbody>
            </table>
    
### `Q_S = 1` (Incremental Inference) Constraints<a name="en-us_topic_0000001832267082_section_qs_eq1_constraint"></a>

- **When `Q_S` is equal to 1**:
    - Usage constraints for `query`, `key`, and `value` inputs:
        - The `B` dimension must be less than or equal to 65536, the `N` dimension must be less than or equal to 256, the `S` dimension must be less than or equal to 262144, and the `D` dimension must be less than or equal to 512.
        - Scenarios where the input data types of `query`, `key`, and `value` are all `int8` are not supported.
        - In `int4` (`int32`) fake-quantization scenarios, PyTorch graph-mode execution only supports inputs where KV `int4` values are packed into `int32` tensors. (Using dynamic quantization to generate data in `int4` format is recommended, as each `int32` value encapsulates eight `int4` values).
        - In `int4` (`int32`) fake-quantization scenarios, when KV `int4` values are packed into `int32` inputs, the `N`, `D`, or `H` dimensions of `key` and `value` must be one-eighth of their actual values. Additionally, `int4` fake quantization requires the `D` dimension to be a multiple of 64, whereas the underlying `int32` tensor requires the `D` dimension to be a multiple of 8.

    - Constraints on `actual_seq_qlen`:
    
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: When `input_layout` of `query` is not `TND` and `Q_S` is 1, this parameter is ignored. For details about the comprehensive constraints when the `input_layout` of `query` is `TND` or `TND_NTD`, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
        
    - Constraints on `actual_seq_kvlen`:
    
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The valid sequence length of each batch in this parameter must be less than or equal to the sequence length of the corresponding batch in `key` and `value`. If the length of this parameter is 1, all batches use the same sequence length. If the length is greater than or equal to the batch size, only the first `batch_size` elements are used. Other lengths are not supported. For details about the comprehensive constraints when the `input_layout` of `key` and `value` is `TND` or `TND_NTD`, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
        
    - Page attention scenarios:
        - Page attention can be enabled only when `block_table` exists and is valid, and `key` and `value` are arranged in a contiguous memory space based on the indices in `block_table`. In this case, the `input_layout` parameter of `key` and `value` is ignored.
        - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
            - The data types of `key` and `value` can be `float16`, `bfloat16`, or `int8`.
            - Scenarios where `query` is `bfloat16` or `float16` while `key` and `value` are `int4` (`int32`) are not supported.
    
        - In page attention scenarios, `block_size` is a user-defined parameter whose value affects page attention performance. When the input data types of `key` and `value` are `float16` or `bfloat16`, the `block_size` must be a multiple of 16. When the input data type is `int8`, it must be a multiple of 32. A value of 128 is recommended. Generally, page attention improves throughput but reduces performance.
        - The product of all dimensions of the tensors corresponding to `key` and `value` must not exceed the representable range of `int32`.
        - In page attention scenarios, `block_table` must be a 2D tensor. Its first dimension must equal `B`, and its second dimension must be no less than `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actual_seq_kvlen` across batches).
        - In page attention scenarios, when `input_layout` of `query` is `BNSD` or `TND`, the KV cache supports both `(blocknum, blocksize, H)` and `(blocknum, KV_N, blocksize, D)` layouts. When `input_layout` of `query` is `BSH` or `BSND`, only the `(blocknum, blocksize, H)` format is supported. The value of `blocknum` must not be less than the total number of blocks required across all batches, calculated from `actual_seq_kvlen` and `block_size`. The shapes of `key` and `value` must be identical.
        - In page attention scenarios, the `(blocknum, KV_N, blocksize, D)` layout format of the KV cache typically provides better performance than `(blocknum, blocksize, H)`, and is therefore recommended.
        - In page attention scenarios, when the KV cache layout format is (blocknum, blocksize, H) and the product of `numKvHeads` and `headDim` exceeds 64K, hardware instruction constraints block execution and raise an error. This can be resolved by enabling GQA (to reduce `numKvHeads`) or by adjusting the KV cache layout format to `(blocknum, numKvHeads, blocksize, D)`.
        - In page attention scenarios, the product of all dimensions of the tensors corresponding to `key` and `value` must not exceed the representable range of `int32`.

    - Constraints on KV fake-quantization parameter separation:
        - Except for scenarios where `key_quant_mode` is 0 and `value_quant_mode` is 1, the values of `key_quant_mode` and `value_quant_mode` must be identical.  
        - `dequant_scale_key` and `dequant_scale_value` must either both be omitted or both be provided. Likewise, `dequant_offset_key` and `dequant_offset_value` must either both be omitted or both be provided.
        - Except for scenarios where `key_quant_mode` is 0 and `value_quant_mode` is 1, when `dequant_scale_key` and `dequant_scale_value` are both provided, their shapes must be identical; when `dequant_offset_key` and `dequant_offset_value` are both provided, their shapes must also be identical.
        - Post-quantization is not supported in `int4` (`int32`) fake-quantization scenarios.
        - The following table describes quantization modes for scale and offset management.
    
            > [!NOTE]   
            > The scale and offset parameters are `dequant_scale_key`, `dequant_scale_value`, `dequant_offset_key`, and `dequant_offset_value`.
    
            <a name="en-us_topic_0000001832267082_table4401182238"></a>
            <table><thead align="left"><tr id="en-us_topic_0000001832267082_row124112817233"><th class="cellrowborder" valign="top" width="16.950000000000003%" id="mcps1.1.5.1.1"><p id="en-us_topic_0000001832267082_p341780235"><a name="en-us_topic_0000001832267082_p341780235"></a><a name="en-us_topic_0000001832267082_p341780235"></a>Quantization Mode</p>
            </th>
            <th class="cellrowborder" valign="top" width="23.09%" id="mcps1.1.5.1.2"><p id="en-us_topic_0000001832267082_p144118852314"><a name="en-us_topic_0000001832267082_p144118852314"></a><a name="en-us_topic_0000001832267082_p144118852314"></a>Scale and Offset Criteria</p>
            </th>
            <th class="cellrowborder" valign="top" width="46.660000000000004%" id="mcps1.1.5.1.3"><p id="en-us_topic_0000001832267082_p123481541027"><a name="en-us_topic_0000001832267082_p123481541027"></a><a name="en-us_topic_0000001832267082_p123481541027"></a>Key and Value Criteria</p>
            </th>
            <th class="cellrowborder" valign="top" width="13.3%" id="mcps1.1.5.1.4"><p id="en-us_topic_0000001832267082_p147001940151615"><a name="en-us_topic_0000001832267082_p147001940151615"></a><a name="en-us_topic_0000001832267082_p147001940151615"></a>Supported Products</p>
            </th>
            </tr>
            </thead>
            <tbody><tr id="en-us_topic_0000001832267082_row10411185232"><td class="cellrowborder" valign="top" width="16.950000000000003%" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p154120882315"><a name="en-us_topic_0000001832267082_p154120882315"></a><a name="en-us_topic_0000001832267082_p154120882315"></a>perchannel</p>
            </td>
            <td class="cellrowborder" valign="top" width="23.09%" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p21391147716"><a name="en-us_topic_0000001832267082_p21391147716"></a><a name="en-us_topic_0000001832267082_p21391147716"></a>The shapes of these parameters can be <code>(1, KV_N, 1, D)</code>, <code>(1, KV_N, D)</code>, and <code>(1, H)</code>. Their data types must match that of <code>query</code>.</p>
            </td>
            <td class="cellrowborder" valign="top" width="46.660000000000004%" headers="mcps1.1.5.1.3 "><a name="en-us_topic_0000001832267082_ul0858154962714"></a><a name="en-us_topic_0000001832267082_ul0858154962714"></a><ul id="en-us_topic_0000001832267082_ul0858154962714"><li><span id="en-us_topic_0000001832267082_ph1163117183317"><a name="en-us_topic_0000001832267082_ph1163117183317"></a><a name="en-us_topic_0000001832267082_ph1163117183317"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_26"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_26"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_26"></a>Atlas A2 training products/Atlas A2 inference products</term></span>: The data types of <code>key</code> and <code>value</code> can be <code>int4</code> (<code>int32</code>) or <code>int8</code>. </li><li><span id="en-us_topic_0000001832267082_ph10252223193315"><a name="en-us_topic_0000001832267082_ph10252223193315"></a><a name="en-us_topic_0000001832267082_ph10252223193315"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_26"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_26"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_26"></a>Atlas A3 training products/Atlas A3 inference products</term></span>: The data types of <code>key</code> and <code>value</code> can be <code>int4</code> (<code>int32</code>) or <code>int8</code>.</li></ul>
            </td>
            <td class="cellrowborder" rowspan="9" valign="top" width="13.3%" headers="mcps1.1.5.1.4 "><a name="en-us_topic_0000001832267082_ul15575120101720"></a><a name="en-us_topic_0000001832267082_ul15575120101720"></a><ul id="en-us_topic_0000001832267082_ul15575120101720"><li><span id="en-us_topic_0000001832267082_ph112662491714"><a name="en-us_topic_0000001832267082_ph112662491714"></a><a name="en-us_topic_0000001832267082_ph112662491714"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_27"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_27"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_27"></a>Atlas A2 training products/Atlas A2 inference products</term></span></li><li><span id="en-us_topic_0000001832267082_ph1290711610176"><a name="en-us_topic_0000001832267082_ph1290711610176"></a><a name="en-us_topic_0000001832267082_ph1290711610176"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_27"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_27"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_27"></a>Atlas A3 training products/Atlas A3 inference products</term></span></li></ul>
            <p id="en-us_topic_0000001832267082_p13700840111618"><a name="en-us_topic_0000001832267082_p13700840111618"></a><a name="en-us_topic_0000001832267082_p13700840111618"></a></p>
            <p id="en-us_topic_0000001832267082_p19700134031612"><a name="en-us_topic_0000001832267082_p19700134031612"></a><a name="en-us_topic_0000001832267082_p19700134031612"></a></p>
            <p id="en-us_topic_0000001832267082_p7700740201616"><a name="en-us_topic_0000001832267082_p7700740201616"></a><a name="en-us_topic_0000001832267082_p7700740201616"></a></p>
            <p id="en-us_topic_0000001832267082_p4700040131614"><a name="en-us_topic_0000001832267082_p4700040131614"></a><a name="en-us_topic_0000001832267082_p4700040131614"></a></p>
            <p id="en-us_topic_0000001832267082_p107002403160"><a name="en-us_topic_0000001832267082_p107002403160"></a><a name="en-us_topic_0000001832267082_p107002403160"></a></p>
            <p id="en-us_topic_0000001832267082_p1670174011616"><a name="en-us_topic_0000001832267082_p1670174011616"></a><a name="en-us_topic_0000001832267082_p1670174011616"></a></p>
            <p id="en-us_topic_0000001832267082_p67011840191613"><a name="en-us_topic_0000001832267082_p67011840191613"></a><a name="en-us_topic_0000001832267082_p67011840191613"></a></p>
            <p id="en-us_topic_0000001832267082_p1870113406160"><a name="en-us_topic_0000001832267082_p1870113406160"></a><a name="en-us_topic_0000001832267082_p1870113406160"></a></p>
            <p id="en-us_topic_0000001832267082_p107011540141611"><a name="en-us_topic_0000001832267082_p107011540141611"></a><a name="en-us_topic_0000001832267082_p107011540141611"></a></p>
            <p id="en-us_topic_0000001832267082_p070174061612"><a name="en-us_topic_0000001832267082_p070174061612"></a><a name="en-us_topic_0000001832267082_p070174061612"></a></p>
            <p id="en-us_topic_0000001832267082_p970174013162"><a name="en-us_topic_0000001832267082_p970174013162"></a><a name="en-us_topic_0000001832267082_p970174013162"></a></p>
            <p id="en-us_topic_0000001832267082_p18701134016166"><a name="en-us_topic_0000001832267082_p18701134016166"></a><a name="en-us_topic_0000001832267082_p18701134016166"></a></p>
            <p id="en-us_topic_0000001832267082_p107011040191616"><a name="en-us_topic_0000001832267082_p107011040191616"></a><a name="en-us_topic_0000001832267082_p107011040191616"></a></p>
            <p id="en-us_topic_0000001832267082_p107072401161"><a name="en-us_topic_0000001832267082_p107072401161"></a><a name="en-us_topic_0000001832267082_p107072401161"></a></p>
            <p id="en-us_topic_0000001832267082_p87072401163"><a name="en-us_topic_0000001832267082_p87072401163"></a><a name="en-us_topic_0000001832267082_p87072401163"></a></p>
            <p id="en-us_topic_0000001832267082_p8707640151615"><a name="en-us_topic_0000001832267082_p8707640151615"></a><a name="en-us_topic_0000001832267082_p8707640151615"></a></p>
            <p id="en-us_topic_0000001832267082_p1870774011617"><a name="en-us_topic_0000001832267082_p1870774011617"></a><a name="en-us_topic_0000001832267082_p1870774011617"></a></p>
            <p id="en-us_topic_0000001832267082_p12707174013166"><a name="en-us_topic_0000001832267082_p12707174013166"></a><a name="en-us_topic_0000001832267082_p12707174013166"></a></p>
            <p id="en-us_topic_0000001832267082_p14707184011619"><a name="en-us_topic_0000001832267082_p14707184011619"></a><a name="en-us_topic_0000001832267082_p14707184011619"></a></p>
            <p id="en-us_topic_0000001832267082_p15707540141620"><a name="en-us_topic_0000001832267082_p15707540141620"></a><a name="en-us_topic_0000001832267082_p15707540141620"></a></p>
            <p id="en-us_topic_0000001832267082_p2707164021613"><a name="en-us_topic_0000001832267082_p2707164021613"></a><a name="en-us_topic_0000001832267082_p2707164021613"></a></p>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row84115813237"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p10419882318"><a name="en-us_topic_0000001832267082_p10419882318"></a><a name="en-us_topic_0000001832267082_p10419882318"></a>pertensor</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p04120872317"><a name="en-us_topic_0000001832267082_p04120872317"></a><a name="en-us_topic_0000001832267082_p04120872317"></a>The shapes of these parameters must be <code>(1,)</code>, and their data types must be identical to that of <code>query</code>.</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="en-us_topic_0000001832267082_ul19978121525716"></a><a name="en-us_topic_0000001832267082_ul19978121525716"></a><ul id="en-us_topic_0000001832267082_ul19978121525716"><li><span id="en-us_topic_0000001832267082_ph19978111585717"><a name="en-us_topic_0000001832267082_ph19978111585717"></a><a name="en-us_topic_0000001832267082_ph19978111585717"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_28"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_28"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_28"></a>Atlas A2 training products/Atlas A2 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int8</code>. </li><li><span id="en-us_topic_0000001832267082_ph13978111545713"><a name="en-us_topic_0000001832267082_ph13978111545713"></a><a name="en-us_topic_0000001832267082_ph13978111545713"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_28"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_28"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_28"></a>Atlas A3 training products/Atlas A3 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int8</code>.</li></ul>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row1341138172312"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p134114862314"><a name="en-us_topic_0000001832267082_p134114862314"></a><a name="en-us_topic_0000001832267082_p134114862314"></a>pertoken</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p1037135185914"><a name="en-us_topic_0000001832267082_p1037135185914"></a><a name="en-us_topic_0000001832267082_p1037135185914"></a>The shapes of these parameters must be <code>(1, B, KV_S)</code>, and their data types must be <code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="en-us_topic_0000001832267082_p174417474211"><a name="en-us_topic_0000001832267082_p174417474211"></a><a name="en-us_topic_0000001832267082_p174417474211"></a>The data types of <code>key</code> and <code>value</code> can be <code>int4</code> (<code>int32</code>) or <code>int8</code>.</p>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row12620173672311"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p166201636132311"><a name="en-us_topic_0000001832267082_p166201636132311"></a><a name="en-us_topic_0000001832267082_p166201636132311"></a>pertensor + perhead</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p662110362235"><a name="en-us_topic_0000001832267082_p662110362235"></a><a name="en-us_topic_0000001832267082_p662110362235"></a>The shapes of these parameters must be <code>(KV_N,)</code>, and their data types must be identical to that of <code>query</code>.</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><a name="en-us_topic_0000001832267082_ul519618222020"></a><a name="en-us_topic_0000001832267082_ul519618222020"></a><ul id="en-us_topic_0000001832267082_ul519618222020"><li><span id="en-us_topic_0000001832267082_ph121961922601"><a name="en-us_topic_0000001832267082_ph121961922601"></a><a name="en-us_topic_0000001832267082_ph121961922601"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_29"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_29"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_29"></a>Atlas A2 training products/Atlas A2 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int8</code>. </li><li><span id="en-us_topic_0000001832267082_ph11961022801"><a name="en-us_topic_0000001832267082_ph11961022801"></a><a name="en-us_topic_0000001832267082_ph11961022801"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_29"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_29"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_29"></a>Atlas A3 training products/Atlas A3 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int8</code>.</li></ul>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row136211336192318"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p22659468119"><a name="en-us_topic_0000001832267082_p22659468119"></a><a name="en-us_topic_0000001832267082_p22659468119"></a>pertoken + perhead</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p116212367239"><a name="en-us_topic_0000001832267082_p116212367239"></a><a name="en-us_topic_0000001832267082_p116212367239"></a>The shapes of these parameters must be <code>(B, KV_N, KV_S)</code>, and their data types must be <code>float32</code>.</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="en-us_topic_0000001832267082_p162285131891"><a name="en-us_topic_0000001832267082_p162285131891"></a><a name="en-us_topic_0000001832267082_p162285131891"></a>The data types of <code>key</code> and <code>value</code> must be <code>int4</code> (<code>int32</code>) or <code>int8</code>.</p>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row1037716581001"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p1757135414112"><a name="en-us_topic_0000001832267082_p1757135414112"></a><a name="en-us_topic_0000001832267082_p1757135414112"></a>pertoken + page attention</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p1025316567221"><a name="en-us_topic_0000001832267082_p1025316567221"></a><a name="en-us_topic_0000001832267082_p1025316567221"></a>The shapes of these parameters must be <code>(blocknum, blocksize)</code>, and their data types must be <code>float32</code>.</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="en-us_topic_0000001832267082_p04417476217"><a name="en-us_topic_0000001832267082_p04417476217"></a><a name="en-us_topic_0000001832267082_p04417476217"></a>The data types of <code>key</code> and <code>value</code> must be <code>int8</code>.</p>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row15621736192312"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p1986911315215"><a name="en-us_topic_0000001832267082_p1986911315215"></a><a name="en-us_topic_0000001832267082_p1986911315215"></a>pertoken + per head + page attention</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p2621173692313"><a name="en-us_topic_0000001832267082_p2621173692313"></a><a name="en-us_topic_0000001832267082_p2621173692313"></a>The shapes of these parameters must be <code>(blocknum, KV_N, blocksize)</code>, and their data types must be <code>float32</code>.</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.3 "><p id="en-us_topic_0000001832267082_p551893313915"><a name="en-us_topic_0000001832267082_p551893313915"></a><a name="en-us_topic_0000001832267082_p551893313915"></a>The data types of <code>key</code> and <code>value</code> must be <code>int8</code>.</p>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row915113171020"><td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p01521217025"><a name="en-us_topic_0000001832267082_p01521217025"></a><a name="en-us_topic_0000001832267082_p01521217025"></a><code>perchannel</code> for <code>key</code> + <code>pertoken</code> for <code>value</code> </p>
            <p id="en-us_topic_0000001832267082_p74743213101"><a name="en-us_topic_0000001832267082_p74743213101"></a><a name="en-us_topic_0000001832267082_p74743213101"></a></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.1.5.1.2 "><p id="en-us_topic_0000001832267082_p1867213119113"><a name="en-us_topic_0000001832267082_p1867213119113"></a><a name="en-us_topic_0000001832267082_p1867213119113"></a>In <code>perchannel</code> mode for <code>key</code>, the shapes of these parameters must be <code>(1, KV_N, 1, D)</code>, <code>(1, KV_N, D)</code>, or <code>(1, H)</code>, and their data types must be identical to that of <code>query</code>.</p>
            </td>
            <td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.1.5.1.3 "><a name="en-us_topic_0000001832267082_ul1037202951112"></a><a name="en-us_topic_0000001832267082_ul1037202951112"></a><ul id="en-us_topic_0000001832267082_ul1037202951112"><li><span id="en-us_topic_0000001832267082_ph10271547141214"><a name="en-us_topic_0000001832267082_ph10271547141214"></a><a name="en-us_topic_0000001832267082_ph10271547141214"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_30"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_30"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term11962195213215_30"></a>Atlas A2 training products/Atlas A2 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int4</code> (<code>int32</code>) or <code>int8</code>. When their data types are <code>int8</code>, the data types of <code>query</code> and the output must be <code>float16</code>. </li><li><span id="en-us_topic_0000001832267082_ph427116472125"><a name="en-us_topic_0000001832267082_ph427116472125"></a><a name="en-us_topic_0000001832267082_ph427116472125"></a><term id="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_30"><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_30"></a><a name="en-us_topic_0000001832267082_en-us_topic_0000001312391781_term1253731311225_30"></a>Atlas A3 training products/Atlas A3 inference products</term></span>: The data types of <code>key</code> and <code>value</code> must be <code>int4</code> (<code>int32</code>) or <code>int8</code>. When their data types are <code>int8</code>, the data types of <code>query</code> and the output must be <code>float16</code>.</li></ul>
            </td>
            </tr>
            <tr id="en-us_topic_0000001832267082_row194748261012"><td class="cellrowborder" valign="top" headers="mcps1.1.5.1.1 "><p id="en-us_topic_0000001832267082_p1154111491113"><a name="en-us_topic_0000001832267082_p1154111491113"></a><a name="en-us_topic_0000001832267082_p1154111491113"></a>In <code>pertoken</code> mode for <code>value</code>, the shapes of these parameters must be <code>(1, B, KV_S)</code>, and their data types must be <code>float32</code>.</p>
            </td>
            </tr>
            </tbody>
            </table>
    
    - Usage constraints for the `pse_shift` feature:
        - The data types of `pse_shift` and `query` must be identical.
        - Only D-axis alignment is supported. That is, the D dimension must be divisible by 16.

## Examples<a name="en-us_topic_0000001832267082_section14459801435"></a>

- Single-operator call

    ```python
    import torch
    import torch_npu
    import math
    # Generate random data and send it to the NPU.
    q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    softmax_scale = 1/math.sqrt(128.0)
    actseqlen = [164]
    actseqlenkv = [1024]

    # Call the FIA operator.
    out, _ = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, 
    actual_seq_qlen = actseqlen, actual_seq_kvlen = actseqlenkv,
    num_query_heads = 8, input_layout = "BNSD", softmax_scale = softmax_scale, pre_tokens=65535, next_tokens=65535)

    print(out)
    # Expected output of the preceding code sample:
    tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ..
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.float16)
    ```

- Graph mode call

    ```python
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

    # Generate data
    q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    softmax_scale = 1/math.sqrt(128.0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)
        print("single op output with mask:", single_op[0], single_op[0].shape)
        print("graph output with mask:", graph_output[0], graph_output[0].shape)
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
            device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])

    graph output with mask: tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
            [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
            [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
            ...,
            [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
            [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
            [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
            device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
    ```
