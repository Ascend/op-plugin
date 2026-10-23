# torch\_npu.npu\_fused\_infer\_attention\_score\_v2<a name="ZH-CN_TOPIC_0000001979260729"></a>

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products<a name="en-us_topic_0000001832267082_section14441124184110"></a>

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id4 -->

## Function<a name="en-us_topic_0000001832267082_section14441124184110"></a>

- Adapts to the `FlashAttention` operator in the incremental and full inference scenarios, supporting both full computation (`PromptFlashAttention`) and incremental computation (`IncreFlashAttention`). This API is recommended when system prefix, left padding, unified KV quantization parameters, and `pertensor` full quantization are not involved. Otherwise, use the legacy API `npu_fused_infer_attention_score`.
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

**Quick Parameter Reference**

| Parameter | Required (Yes/No) | Type | Default Value | Description |
| ------ | ----------- | ------ | -------- | ------ |
| query | Yes | Tensor | - | Query input |
| key | Yes | Tensor | - | Key input |
| value | Yes | Tensor | - | Value input |
| query_rope | No | Tensor | None | RoPE information for `query` in the MLA structure |
| key_rope | No | Tensor | None | RoPE information for `key` in the MLA structure |
| pse_shift | No | Tensor | None | Position embedding parameter |
| atten_mask | No | Tensor | None | Attention mask |
| actual_seq_qlen | No | List[int] | None | Valid sequence lengths of `query` |
| actual_seq_kvlen | No | List[int] | None | Valid sequence lengths of `key`/`value` |
| block_table | No | Tensor | None | Block mapping table for PagedAttention |
| dequant_scale_query | No | Tensor | None | Dequantization factor for `query` |
| dequant_scale_key | No | Tensor | None | Dequantization factor for `key` |
| dequant_offset_key | No | Tensor | None | Dequantization offset for `key` |
| dequant_scale_value | No | Tensor | None | Dequantization factor for `value` |
| dequant_offset_value | No | Tensor | None | Dequantization offset for `value` |
| dequant_scale_key_rope | No | Tensor | None | Reserved parameter, currently not used |
| quant_scale_out | No | Tensor | None | Quantization factor for the output |
| quant_offset_out | No | Tensor | None | Quantization offset for the output |
| quant_scale_p | No | Tensor | None | Reserved parameter, currently not used |
| learnable_sink | No | Tensor | None | Learnable sink token |
| num_query_heads | No | int | 1 | Number of `query` heads |
| num_key_value_heads | No | int | 0 | Number of `key`/`value` heads; `0` indicates the same number as `query` |
| softmax_scale | No | float | 1.0 | Scaling factor; passing `$1/\sqrt{D}$` is recommended |
| pre_tokens | No | int | 2147483647 | Number of forward tokens for sparse computation |
| next_tokens | No | int | 2147483647 | Number of backward tokens for sparse computation |
| input_layout | No | str | "BSH" | Input data layout |
| sparse_mode | No | int | 0 | Sparse mode |
| block_size | No | int | 0 | Maximum number of tokens per block for PagedAttention |
| query_quant_mode | No | int | 0 | Pseudo-quantization mode for `query` |
| key_quant_mode | No | int | 0 | Pseudo-quantization mode for `key` |
| value_quant_mode | No | int | 0 | Pseudo-quantization mode for `value` |
| inner_precise | No | int | 0 | Precision mode |
| return_softmax_lse | No | bool | False | Whether to output `softmax_lse` |
| query_dtype | No | int | None | Reserved parameter, currently not used |
| key_dtype | No | int | None | Data type of `key` |
| value_dtype | No | int | None | Data type of `value` |
| query_rope_dtype | No | int | None | Reserved parameter, currently not used |
| key_rope_dtype | No | int | None | Reserved parameter, currently not used |
| key_shared_prefix_dtype | No | int | None | Reserved parameter, currently not used |
| value_shared_prefix_dtype | No | int | None | Reserved parameter, currently not used |
| dequant_scale_query_dtype | No | int | None | Reserved parameter, currently not used |
| dequant_scale_key_dtype | No | int | None | Data type of `dequant_scale_key` |
| dequant_scale_value_dtype | No | int | None | Data type of `dequant_scale_value` |
| dequant_scale_key_rope_dtype | No | int | None | Reserved parameter, currently not used |
| out_dtype | No | int | None | Output data type |

- **`query`** (`Tensor`): Required. Query input of the attention structure, `Q` in the formula. Non-contiguous tensors are not supported. The data type can be `torch.float16` or `torch.bfloat16`. The data layout can be ND.

  <!-- npu="A3,910b" id6 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.int8`, `torch_npu.hifloat8`, or `torch.float8_e4m3fn`.
  <!-- end id7 -->

- **`key`** (`Tensor`): Required. Key input of the attention structure, corresponding to `K` in the formula. Except in MXFP8 single-operator direct-call scenarios, non-contiguous Tensors are not supported. The data layout supports ND.

  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.int8`, or `torch_npu.int4` (`torch.int32`).
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.int8`, `torch_npu.int4` (`torch.int32`), `torch_npu.hifloat8`, `torch.float8_e4m3fn`, or `torch.float4_e2m1fn_x2`.
  <!-- end id9 -->

- **`value`** (`Tensor`): Required. Value input of the attention structure, corresponding to `V` in the formula. Except in MXFP8 single-operator direct-call scenarios, non-contiguous Tensors are not supported. The data layout supports ND.

  <!-- npu="A3,910b" id10 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.int8`, or `torch_npu.int4` (`torch.int32`).
  <!-- end id10 -->
  <!-- npu="950" id11 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.int8`, `torch_npu.int4` (`torch.int32`), `torch_npu.hifloat8`, `torch.float8_e4m3fn`, or `torch.float4_e2m1fn_x2`.
  <!-- end id11 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`query_rope`** (`Tensor`): Optional. RoPE information of the query in the Multi-Head Latent Attention (MLA) structure. The data type can be `torch.float16` or `torch.bfloat16`. Non-contiguous tensors are not supported. The data layout can be ND.
- **`key_rope`** (`Tensor`): Optional. RoPE information of the key in the Multi-Head Latent Attention (MLA) structure. The data type can be `torch.float16` or `torch.bfloat16`. Non-contiguous tensors are not supported. The data layout can be ND.
- **`pse_shift`** (`Tensor`): Optional. Position embedding parameter within the attention structure. The data type can be `torch.float16` or `torch.bfloat16`. The data type of this parameter and that of `query` must comply with type promotion rules. Non-contiguous tensors are not supported. The data layout can be ND. This parameter can be set to `None` if this feature is not used.

    - When `Q_S` is greater than 1: if `pse_shift` is `torch.float16`, `query` must be `torch.float16` or `torch.int8`; if `pse_shift` is `torch.bfloat16`, `query` must be `torch.bfloat16`. The input shape must be `(B, Q_N, Q_S, KV_S)` or `(1, Q_N, Q_S, KV_S)`. If `KV_S` of `pse_shift` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. There are no specific requirements for the padding values of the padded portion.
    - When `Q_S` is equal to 1: if `pse_shift` is `torch.float16`, `query` must be `torch.float16`; if `pse_shift` is `torch.bfloat16`, `query` must be `torch.bfloat16`. The input shape must be (B, Q_N, 1, KV_S) or (1, Q_N, 1, KV_S). If `KV_S` of `pse_shift` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. There are no specific requirements for the padding values of the padded portion.

- **`atten_mask`** (`Tensor`): Optional. Masks the QK product to indicate whether correlation between tokens is computed. The data type can be `torch.bool`, `torch.int8`, or `torch.uint8`. Non-contiguous tensors are not supported. The data layout can be ND. This parameter can be set to `None` if this feature is not used.

  <!-- npu="A3,910b" id12 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - When `sparse_mode` is `0` or `1`:
        - The input shape can be (1, Q_S, KV_S), (B, 1, Q_S, KV_S), or (1, 1, Q_S, KV_S).
        - When `input_layout` is `BSH`, `BSND`, `BNSD`, or `BNSD_BSND`, the `D` dimensions of `query`, `key`, and `value` are identical, and `query_rope` and `key_rope` are omitted, a shape of (B, KV_S) can be passed if `Q_S` is 1, and (Q_S, KV_S) can be passed if `Q_S` is greater than 1.
        - If `Q_S` or `KV_S` is not 16-byte or 32-byte aligned, it can be rounded up to the aligned value. For details about the overall constraints, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint).
    - When `sparse_mode` is `2`, `3`, or `4`, the input shape can be `(2048, 2048)`, `(1, 2048, 2048)`, or `(1, 1, 2048, 2048)`.
    - When `sparse_mode` is `9`:
        - If `input_layout` is `BSH`, `BSND`, or `BNSD`, the input shape can be `(B, Q_S, Q_S)`.
        - If `input_layout` is `TND`, the input shape can be a 1D tensor `(∑Q_Si²,)`, where the total length equals the sum of the squared query sequence lengths of all batches. That is, the individual matrix mask of each batch is concatenated into a single 1D tensor.
  <!-- end id12 -->
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT products</term>:
    - When Q_S is not 1, the recommended input shapes are (B, Q_S, KV_S), (1, Q_S, KV_S), (B, 1, Q_S, KV_S), and (1, 1, Q_S, KV_S).
    - When Q_S is 1, the recommended input shapes are (B, 1, KV_S) and (B, 1, 1, KV_S).
    - If Q_S and KV_S are not aligned to 16 or 32, they can be rounded up to the aligned S.
  <!-- end id13 -->

- **`actual_seq_qlen`** (`List[int]`): Optional. Valid sequence length (`seqlen`) of `query` in different batches. The data type can be `torch.int64`. The default value is `None`, indicating that the value is identical to the sequence length `S` in the `query` shape.

  <!-- npu="A3,910b" id14 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:  
    The valid `seqlen` of each batch in this parameter must not exceed that of the corresponding batch in `query`. If the input length of `seqlen` is `1`, all batches use the same `seqlen`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlen` are used. Other lengths are not supported. When `input_layout` of `query` is `TND`, this parameter is required, and the number of elements in this parameter is used as the batch size. The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element, and cannot be negative.
  <!-- end id14 -->

- **`actual_seq_kvlen`** (`List[int]`): Optional. Valid sequence length of `key` and `value` across different batches. The data type can be `torch.int64`. The default value is `None`, indicating that the value is identical to the sequence length `S` in the `key` and `value` shapes. Constraints vary across different `Q_S` values. For details, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint) and [Constraints for Q_S=1](#en-us_topic_0000001832267082_section_qs_eq1_constraint).
- **`block_table`** (`Tensor`): Optional. Block mapping table used for KV storage in PagedAttention. The data type can be `torch.int32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used.
- **`dequant_scale_query`** (`Tensor`): Optional. Dequantization parameter of `query`, supporting only `pertoken + perhead` mode. The data type can be `torch.float32`. The data format can be $ND$. Pass `None` if this feature is not used. For details about the overall constraints, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).

  <!-- npu="A3,910b" id15 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only pertoken combined with perhead is supported. The data type can be `torch.float32`.
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT products</term>: Per-token combined with per-head and per-block are supported, in which case the data type is `torch.float32`; per-channel-group is supported, in which case the data type is `torch.float8_e8m0fnu`.
  <!-- end id16 -->

- **`dequant_scale_key`** (`Tensor`): Optional. Dequantization factor of `key` when the KV fake-quantization parameters are separated. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with scale managed by PagedAttention, and `pertoken` + `perhead` with scale managed by PagedAttention. Pass `None` if this feature is not used. For details about the overall constraints, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [Constraints for Q_S=1](#en-us_topic_0000001832267082_section_qs_eq1_constraint), [GQA Fake Quantization + KV NZ Layout Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).

  <!-- npu="A3,910b" id17 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id17 -->
  <!-- npu="950" id18 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.float32`, or `torch.float8_e8m0fnu`.
  <!-- end id18 -->

- **`dequant_offset_key`** (`Tensor`): Optional. Dequantization offset of `key` when the KV fake-quantization parameters are separated. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with offset managed by PagedAttention, and `pertoken` + `perhead` with offset managed by PagedAttention. Pass `None` if this feature is not used. For details about the overall constraints, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [Constraints for Q_S=1](#en-us_topic_0000001832267082_section_qs_eq1_constraint), [GQA Fake Quantization + KV NZ Layout Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
- **`dequant_scale_value`** (`Tensor`): Optional. Dequantization factor of `value` when the KV fake-quantization parameters are separated. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with scale managed by PagedAttention, and `pertoken` + `perhead` with scale managed by PagedAttention. Pass `None` if this feature is not used. For details about the overall constraints, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [Constraints for Q_S=1](#en-us_topic_0000001832267082_section_qs_eq1_constraint), [GQA Fake Quantization + KV NZ Layout Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).

  <!-- npu="A3,910b" id19 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`.
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16`, `torch.bfloat16`, `torch.float32`, or `torch.float8_e8m0fnu`.
  <!-- end id20 -->

- **`dequant_offset_value`** (`Tensor`): Optional. Dequantization offset of `value` when the KV fake-quantization parameters are separated. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with offset managed by PagedAttention, and `pertoken` + `perhead` with offset managed by PagedAttention. Pass `None` if this feature is not used. For details about the overall constraints, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [Constraints for Q_S=1](#en-us_topic_0000001832267082_section_qs_eq1_constraint), [GQA Fake Quantization + KV NZ Layout Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
- **`dequant_scale_key_rope`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**
- **`quant_scale_out`** (`Tensor`): Optional. Quantization factor for the output. The data type can be `torch.float32` or `torch.bfloat16`. The data layout can be ND. `pertensor` and `perchannel` modes are supported. When the input data type is `torch.bfloat16`, both `torch.float32` and `torch.bfloat16` are supported. Otherwise, only `torch.float32` is supported. In `perchannel` mode, if the output layout is `BSH`, the product of all dimensions of `quant_scale_out` must be identical to `H`. For other layouts, the product must equal `Q_N * D`. (You are advised to pass a shape of `(1, 1, H)` or `(H,)` when the output layout is `BSH`; `(1, Q_N, 1, D)` or `(Q_N, D)` when the output layout is `BNSD`; and `(1, 1, Q_N, D)` or `(Q_N, D)` when the output layout is `BSND`). Pass `None` if this feature is not used. For details about the overall constraints, see [General Constraints](#en-us_topic_0000001832267082_section_general_constraint).
- **`quant_offset_out`** (`Tensor`): Optional. Quantization offset for the output. The data type can be `torch.float32` or `torch.bfloat16`. The data layout can be ND. `pertensor` and `perchannel` modes are supported. If `quant_offset_out` is provided, its data type and shape must be identical to those of `quant_scale_out`. Pass `None` if this feature is not used. For details about the overall constraints, see [General Constraints](#en-us_topic_0000001832267082_section_general_constraint).
- **`quant_scale_p`** (`Tensor`): Optional. **Reserved parameter, currently not used. Retain the default value.**

  <!-- npu="A3,910b" id21 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Reserved parameter, not used currently. Use the default value.
  <!-- end id21 -->
  <!-- npu="950" id22 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32`. The data layout supports ND. Per-tensor is supported, and the shape must be (1). If this function is not used, None can be passed.
  <!-- end id22 -->

- **`learnable_sink`** (`Tensor`): Optional. Learnable sink token used to absorb attention scores. The data type can be `torch.bfloat16`. The data layout can be ND, and the shape must be `(Q_N,)`. The default value is `None`. For details about the overall constraints, see [learnable_sink Constraints](#en-us_topic_0000001832267082_section_learnable_sink_constraint).

- **`num_query_heads`** (`int`): Optional. Number of heads for `query`. The data type can be `torch.int64`. In `BNSD` scenarios, this value must be identical to the shape of the N axis of `query`. Otherwise, an execution exception occurs. For details about the overall constraints, see [GQA Fake Quantization + KV NZ Layout Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint).
- **`num_key_value_heads`** (`int`): Optional. Number of heads for `key` and `value`, used to support Grouped-Query Attention (GQA) scenarios. The data type can be `torch.int64`. The default value is `0`, indicating that the number of heads for `key`, `value`, and `query` are identical. The value of `num_query_heads` must be divisible by `num_key_value_heads`, and the ratio of `num_query_heads` to `num_key_value_heads` cannot exceed 64. In `BSND`, `BNSD`, and `BNSD_BSND` (supported only when `Q_S` is greater than 1) scenarios, this value must also be identical to the shape of the N axis of `key` and `value`. Otherwise, an execution exception occurs. For details about the overall constraints, see [GQA Fake Quantization + KV NZ Layout Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint).
- **`softmax_scale`** (`float`): Optional. Scaling factor, the reciprocal of the square root of `d` in the formula, used as a scalar value for Muls in the computation flow. The data type can be `torch.float32`. Its data type and the data type of `query` must meet the type deduction rules. The default value is `1.0`, which means no scaling is applied. **Passing `1/√D` (where D is the Head Dim) is recommended**. For example, when `D` is `128`, pass `1/math.sqrt(128.0)` to obtain correct attention calculation results.
- **`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `torch.int64`. The default value is `2147483647`. This parameter is invalid when `Q_S` is `1`.
- **`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `torch.int64`. The default value is `2147483647`. This parameter is invalid when `Q_S` is `1`.
- **`input_layout`** (`str`): Optional. Data layout of the input `query`, `key`, and `value`. The default value is "BSH".

    > [!NOTE]
    > When a layout format contains an underscore (_), the portion to the left of the underscore represents the layout of the input `query`, and the portion to the right represents the layout of the output. The operator performs layout conversion internally.

    <!-- npu="A3,910b" id23 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: BSH, BSND, BNSD, BNSD_BSND (when the input is BNSD, the output format is BSND, and only Q_S greater than 1 is supported), BSH_NBSD, BSND_NBSD, BNSD_NBSD (when the output format is NBSD, only Q_S greater than 1 and less than or equal to 16 is supported), TND, TND_NTD, and NTD_TND (for the comprehensive constraints of TND-related scenarios, see [MLA scenario constraints](#mla-scenario-constraints))
    <!-- end id23 -->
    <!-- npu="950" id24 -->
    - <term>Ascend 950PR&950DT products</term>: BSH, BSND, BNSD, BNSD_BSND (when the input is BNSD, the output format is BSND, and only Q_S greater than 1 is supported), BSH_NBSD, BSND_NBSD, BNSD_NBSD (when the output format is NBSD, only Q_S greater than 1 and less than or equal to 16 is supported), TND, NTD, TND_NTD, and NTD_TND (for the comprehensive constraints of TND-related scenarios, see [MLA scenario constraints](#mla-scenario-constraints)).
    <!-- end id24 -->

    | `input_layout` | `query` Shape | `key` Shape | `value` Shape | Output (`attention_out`) Shape | Description |
    | ------------- | ----------- | --------- | ----------- | ---------- | ---- |
    | BSH | (B, Q\_S, H) | (B, KV\_S, H) | (B, KV\_S, H) | (B, Q\_S, H) | H=N\*D |
    | BSND | (B, Q_S, Q_N, D) | (B, KV_S, KV_N, D) | (B, KV_S, KV_N, D) | (B, Q_S, Q_N, D) | N and D are separate |
    | BNSD | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, KV_N, KV_S, D) | (B, Q_N, Q_S, D) | N and D are separate; N comes before D |
    | BNSD_BSND | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, KV_N, KV_S, D) | (B, Q_S, Q_N, D) | Input: BNSD; output: BSND; supported only when Q_S > 1 |
    | BSH_NBSD | (B, Q_S, H) | (B, KV_S, H) | (B, KV_S, H) | (Q_N, B, Q_S, D) | Input: BSH; output: NBSD |
    | BSND_NBSD | (B, Q_S, Q_N, D) | (B, KV_S, KV_N, D) | (B, KV_S, KV_N, D) | (Q_N, B, Q_S, D) | Input: BSND; output: NBSD |
    | BNSD_NBSD | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, KV_N, KV_S, D) | (Q_N, B, Q_S, D) | The input is BNSD and the output is NBSD. Only Q_S greater than 1 and less than or equal to 16 is supported. |
    | BNSD_NBSD | (B, Q_N, Q_S, D) | (B, KV_N, KV_S, D) | (B, KV_N, KV_S, D) | (Q_N, B, Q_S, D) | Input: BNSD; output: NBSD; supported only when Q_S is 1–16 |
    | TND | (T, Q_N, D) | (T, KV_N, D) | (T, KV_N, D) | (T, Q_N, D) | T is the sum of S across all batches |
    | TND_NTD | (T, Q_N, D) | (T, KV_N, D) | (T, KV_N, D) | (Q_N, T, D) | Input: TND; output: NTD |
    | NTD_TND | (Q_N, T, D) | (KV_N, T, D) | (KV_N, T, D) | (T, Q_N, D) | Input: NTD; output: TND |

- **`sparse_mode`** (`int`): Optional. Sparse mode. The default value is `0`. The data type can be `torch.int64`. This parameter is invalid when `Q_S` is `1` and no RoPE input is provided. For details about the overall constraints when `input_layout` is `TND`, `TND_NTD`, or `NTD_TND`, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint). For details about the overall constraints in GQA fake-quantization scenarios, see [GQA Fake Quantization + KV NZ Layout Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint). Currently, only the values 0, 1, 2, 3, 4, and 9 are supported. The values 5, 6, 7, and 8 (representing `prefix`, `global`, `dilated`, and `block_local`, respectively) are not yet implemented and should not be used.

    <!-- npu="A3,910b" id25 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
        - Currently only the values 0, 1, 2, 3, 4, and 9 are supported.
    <!-- end id25 -->
    <!-- npu="950" id26 -->
    - <term>Ascend 950PR&950DT products</term>:
        - Currently only the values 0, 1, 2, 3, and 4 are supported.
    <!-- end id26 -->

    | Value | Mode Name | Description | `atten_mask` Requirement |
    | ------ | ---------- | ------ | ----------------- |
    | 0 | defaultMask | If `atten_mask` is not provided, no masking is performed, and `pre_tokens` and `next_tokens` are ignored (internally set to `INT_MAX`). If `atten_mask` is provided, a complete `atten_mask` matrix `(S1 * S2)` must be provided, indicating that the portion between `pre_tokens` and `next_tokens` needs to be computed. | Optional |
    | 1 | allMask | A complete `atten_mask` matrix `(S1 * S2)` must be provided. | Required `(S1 * S2)` |
    | 2 | leftUpCausal | Mask for the upper-left causal mode | Optimized `atten_mask` matrix `(2048 * 2048)` |
    | 3 | rightDownCausal | Mask for the lower-right causal mode, corresponding to the lower-triangular scenario with the top-right vertex as the partition point | Optimized `atten_mask` matrix `(2048 * 2048)` |
    | 4 | band | Mask for the band mode | Optimized `atten_mask` matrix `(2048 * 2048)` |
    | 9 | treeMask | Tree attention mask for speculative decoding scenarios. Supported only in MLA scenarios (where `query_rope` and `key_rope` are not `None`). Left padding, `pse_shift`, and `sharedPrefix` are not supported. The output `dtype` does not support `torch.int8`. Each batch must satisfy $Q\_S \le KV\_S$. | Required (custom tree mask) |

- **`block_size`** (`int`): Optional. Maximum number of tokens in each block of KV storage in PagedAttention. The data type can be `torch.int64`. The default value is 0.
- **`query_quant_mode`** (`int`): Optional. Fake-quantization mode for `query`. Only 3 can be passed, indicating mode 3: `pertoken` + `perhead` mode.

  <!-- npu="A3,910b" id27 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only the value 3 is supported, representing mode 3: pertoken combined with perhead mode.
  <!-- end id27 -->
  <!-- npu="950" id28 -->
  - <term>Ascend 950PR&950DT products</term>: Represents mode 3: per-token combined with per-head mode. Represents mode 6: per-token-group full quantization mode. Represents mode 7: FP8 per-block full quantization mode.
  <!-- end id28 -->

- **`key_quant_mode`** (`int`): Optional. Fake-quantization mode for `key`. The default value is 0. Except for scenarios where `key_quant_mode` is `0` and `value_quant_mode` is `1`, this value must be identical to `value_quant_mode`. For details about the overall constraints, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint) and [Constraints for Q_S=1](#en-us_topic_0000001832267082_section_qs_eq1_constraint).

    <!-- npu="A3,910b" id29 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When Q_S>=2, only the values 0 and 1 are supported; when Q_S=1, the values 0, 1, 2, 3, 4, and 5 are supported.
    <!-- end id29 -->
    <!-- npu="950" id30 -->
    - <term>Ascend 950PR&950DT products</term>: The values 0, 1, 2, 3, 4, 5, 6, and 7 are supported.
    <!-- end id30 -->

    | Value | Description |
    | ------ | ------ |
    | 0 | perchannel mode (perchannel includes pertensor) |
    | 1 | pertoken mode |
    | 2 | pertensor combined with perhead mode |
    | 3 | pertoken combined with perhead mode |
    | 4 | pertoken combined with the page attention mode for managing scale/offset |
    | 5 | pertoken combined with perhead and using the page attention mode for managing scale/offset |
    | 6 | per-token-group mode |
    | 7 | FP8 per-block full quantization mode |
    | 8 | per-channel-group full quantization mode |

- **`value_quant_mode`** (`int`): Optional. Fake-quantization mode for `value`. Mode numbering is identical to that of `key_quant_mode`. The default value is `0`. Except for scenarios where `key_quant_mode` is `0` and `value_quant_mode` is `1`, this value must be identical to `key_quant_mode`. For details about the overall constraints, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [Constraints for Q_S=1](#en-us_topic_0000001832267082_section_qs_eq1_constraint), and [GQA Fake Quantization + KV NZ Layout Constraints](#en-us_topic_0000001832267082_section_gqa_nz_constraint).

    <!-- npu="A3,910b" id31 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When Q_S>=2, only the values 0 and 1 are supported; when Q_S=1, the values 0, 1, 2, 3, 4, and 5 are supported.
    <!-- end id31 -->
    <!-- npu="950" id32 -->
    - <term>Ascend 950PR&950DT products</term>: The values 0, 1, 2, 3, 4, 5, 6, 7, and 8 are supported.
    <!-- end id32 -->

- **`inner_precise`** (`int`): Optional. The data type can be `torch.int64`. Four modes are supported: `0`, `1`, `2`, and `3`. Each option is represented by a 2-bit value. Bit 0 specifies high-precision or high-performance mode, and bit 1 specifies whether to perform invalid row correction. When `Q_S` is greater than 1, if `sparse_mode` is `0` or `1` and a user-defined mask is provided, enabling invalid row correction is recommended. When `Q_S` is `1`, you can only set `inner_precise` to `0` or `1`. For details about the overall constraints, see [Constraints for Q_S>1](#en-us_topic_0000001832267082_section_qs_gt1_constraint), [Constraints for Q_S=1](#en-us_topic_0000001832267082_section_qs_eq1_constraint), and [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).

    - `0`: enables high-precision mode and disables invalid row correction.
    - `1`: enables high-performance mode and disables invalid row correction.
    - `2`: enables high-precision mode and enables invalid row correction.
    - `3`: enables high-performance mode and enables invalid row correction.

    > [!NOTE]
    > For `torch.bfloat16` and `torch.int8`, no distinction is made between high-precision and high-performance modes. Invalid row correction applies to `torch.float16`, `torch.bfloat16`, and `torch.int8`. Currently, `0` and `1` are reserved values. If any row in the mask involved in computation consists entirely of `1`s, precision may degrade. In this case, you can set this parameter to `2` or `3` to enable invalid row correction and improve precision. However, this configuration may degrade performance.

- **`return_softmax_lse`** (`bool`): Optional. Specifies whether to output `softmax_lse`, supporting S-axis tiling (splitting along the sequence length dimension). `True` outputs `softmax_lse`, and `False` disables it. The default value is `False`.
- **`query_dtype`** (`int`): Optional. Data type of `query`. **Reserved parameter, currently not used. Retain the default value.**
- **`key_dtype`** (`int`): Optional. Data type of `key`. **Reserved parameter, currently not used. Retain the default value.**

  <!-- npu="A3,910b" id33 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Reserved parameter, not used currently. Use the default value.
  <!-- end id33 -->
  <!-- npu="950" id34 -->
  - <term>Ascend 950PR&950DT products</term>: A value must be passed when the data type of `key` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.hifloat8`.
  <!-- end id34 -->

- **`value_dtype`** (`int`): Optional. Data type of `value`. **Reserved parameter, currently not used. Retain the default value.**

  <!-- npu="A3,910b" id35 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Reserved parameter, not used currently. Use the default value.
  <!-- end id35 -->
  <!-- npu="950" id36 -->
  - <term>Ascend 950PR&950DT products</term>: A value must be passed when the data type of `value` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.hifloat8`.
  <!-- end id36 -->

- **`query_rope_dtype`** (`int`): Optional. Data type of `query_rope`. **Reserved parameter, currently not used. Retain the default value.**
- **`key_rope_dtype`** (`int`): Optional. Data type of `key_rope`. **Reserved parameter, currently not used. Retain the default value.**
- **`key_shared_prefix_dtype`** (`int`): Optional. Data type of `key_shared_prefix`. **Reserved parameter, currently not used. Retain the default value.**
- **`value_shared_prefix_dtype`** (`int`): Optional. Data type of `value_shared_prefix`. **Reserved parameter, currently not used. Retain the default value.**
- **`dequant_scale_query_dtype`** (`int`): Optional. Data type of `dequant_scale_query`. **Reserved parameter, currently not used. Retain the default value.**
- **`dequant_scale_key_dtype`** (`int`): Optional. Data type of `dequant_scale_key`. **Reserved parameter, currently not used. Retain the default value.**

  <!-- npu="A3,910b" id37 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Reserved parameter, not used currently. Use the default value.
  <!-- end id37 -->
  <!-- npu="950" id38 -->
  - <term>Ascend 950PR&950DT products</term>: A value must be passed when the data type of `dequant_scale_key` is `torch.float8_e8m0fnu`.
  <!-- end id38 -->

- **`dequant_scale_value_dtype`** (`int`): Optional. Data type of `dequant_scale_value`. **Reserved parameter, currently not used. Retain the default value.**

  <!-- npu="A3,910b" id39 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Reserved parameter, not used currently. Use the default value.
  <!-- end id39 -->
  <!-- npu="950" id40 -->
  - <term>Ascend 950PR&950DT products</term>: A value must be passed when the data type of `dequant_scale_value` is `torch.float8_e8m0fnu`.
  <!-- end id40 -->

- **`dequant_scale_key_rope_dtype`** (`int`): Optional. Data type of `dequant_scale_key_rope`. **Reserved parameter, currently not used. Retain the default value.**
- **`out_dtype`** (`int`): Optional. Output data type. When the input is `torch.int8` or `torch.float8_e4m3fn`, this parameter can be used to specify the output data type (such as `torch.float8_e5m2`). This parameter can be set to `None` if this feature is not used.

## Return Values<a name="en-us_topic_0000001832267082_section22231435517"></a>

- **`attention_out`** (`Tensor`): Output tensor in the formula. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`. The data layout can be ND. Restriction: The `D` dimension of the output must be identical to that of `value`, and all other dimensions must match those of the input parameter `query`.
- **`softmax_lse`** (`Tensor`): log-sum-exp result computed over the Query-Key product by using the Ring Attention algorithm. It is obtained by first computing `query` * `key`, subtracting the row-wise maximum (`softmax_max`), applying `exp`, summing the results (`softmax_sum`), taking the natural logarithm of `softmax_sum`, and finally adding `softmax_max`. The data type is `torch.float32`. When `return_softmax_lse` is `True`, the output shape is `(B, Q_N, Q_S, 1)` under standard layouts, and `(T, Q_N, 1)` when `input_layout` is `TND` or `NTD_TND`. When `return_softmax_lse` is `False`, the output is a tensor of shape `[1]` with value `0`.

## Constraints<a name="en-us_topic_0000001832267082_section12345537164214"></a>

> [!NOTICE]
> The constraints are organized by scenario and can be consulted as needed.
>
> - [**General Constraints**](#en-us_topic_0000001832267082_section_general_constraint): Handling of empty inputs, consistency between `key`/`value` shapes, and `torch.int8` quantization limitations.
> - [**MLA Scenario Constraints**](#en-us_topic_0000001832267082_section_mla_constraint) (when `query_rope` and `key_rope` are provided): Constraints on D=512, constraints on D=128, and TND scenario constraints.
> - [**GQA Fake Quantization + KV NZ Layout Constraints**](#en-us_topic_0000001832267082_section_gqa_nz_constraint): KV NZ input layout, `dequant_scale` constraints, `sparse_mode` constraints, and constraints on `num_query_heads`/`num_key_value_heads` combinations.
> - [**`learnable_sink` Constraints**](#en-us_topic_0000001832267082_section_learnable_sink_constraint): Scenario-specific constraints when the `learnable_sink` parameter is used.
> - [** Constraints for Q_S>1 (Full Inference)**](#en-us_topic_0000001832267082_section_qs_gt1_constraint): Input shape constraints, `sparse_mode` constraints, PagedAttention constraints, quantization constraints, `pse_shift` constraints, and separation of KV fake-quantization parameters.
> - [**Constraints for Q_S=1 (Incremental Inference)**](#en-us_topic_0000001832267082_section_qs_eq1_constraint): Input shape constraints, PagedAttention constraints, and separation of KV fake-quantization parameters.

### General constraints<a name="en-us_topic_0000001832267082_section_general_constraint"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Empty input handling: The operator checks whether `query` is empty internally. If `query` is empty, an empty value is returned. If `query` is a non-empty tensor but `key` and `value` are empty tensors (`S2` is `0`), `attention_out` returns a tensor filled with all `0`s matching its corresponding shape. If `attention_out` evaluates to an empty tensor, an empty value is returned.
- The corresponding tensors of `key` and `value` must have identical shapes. In non-contiguous scenarios, each tensor in the tensorlist of `key` and `value` must have a batch size of 1. The number of tensors must be equal to `B` in `query`, and the `N` and `D` dimensions must be identical.
- Constraints on the number of input parameters and output data formats related to `torch.int8` quantization:
    - For `torch.int8` output: The input parameter `quant_scale_out` must be provided. `quant_offset_out` is optional and defaults to `0` if not specified.

        <!-- npu="A3,910b" id41 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The input is `torch.int8`.
        <!-- end id41 -->

    - For `torch.float16` output: If the input parameter `quant_offset_out` or `quant_scale_out` is provided (not `None`), an error is raised and execution returns.
    - `quant_offset_out` and `quant_scale_out` support `pertensor` or `perchannel` modes, and their data types can be `torch.float32` or `torch.bfloat16`.

### MLA scenario constraints<a name="en-us_topic_0000001832267082_section_mla_constraint"></a>

- When `query_rope` and `key_rope` are provided (MLA scenario), the following constraints apply:
    - The data type and data format of `query_rope` must be identical to those of `query`.
    - The data type and data format of `key_rope` must be identical to those of `key`.
    - `query_rope` and `key_rope` must either both be omitted or both be provided. Partial configuration is not supported.
    - When `query_rope` and `key_rope` are non-empty, the `D` dimension of `query` must be `512` or `128`.
        - When `D` of `query` is `512`:

            <!-- npu="A3,910b" id42 -->
            - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
                - `sparse_mode`: Valid values: `0`, `3`, `4`, or `9`.
                - When `query_rope` is configured, the valid values of the `N` dimension of `query` are `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`. The `D` dimension of `query_rope` must be `64`, and all other dimensions must match those of `query`.
                - When `key_rope` is configured: the `N` dimension must be `1` and the `D` dimension must be `512` for `key`. The `D` dimension of `key_rope` must be `64`, and all other dimensions must match those of `key`.
                - Supported data layouts for `key`, `value`, and `key_rope`: ND or NZ. When the data layout is NZ, the layout of the input parameters `key` and `value` is `[blockNum, KV_N, D/16, blockSize, 16]` for the `torch.float16` or `torch.bfloat16` data type, and `[blockNum, KV_N, D/32, blockSize, 32]` for the `torch.int8` input data type.
                - Supported `input_layout` values: `BSH`, `BSND`, `BNSD`, `BNSD_NBSD`, `BSND_NBSD`, `BSH_NBSD`, `TND`, and `TND_NTD`.
                - PagedAttention is supported: `block_size` must be divisible by 16 and cannot exceed 1024.
                - Left padding, tensorlist, PSE, prefix, fake-quantization, post-quantization, and empty tensors are not supported.
                - Full quantization is supported (scenarios where the data types of the input parameters `query`, `key`, and `value` are entirely `torch.int8`, `query_rope`, and `key_rope` are `torch.bfloat16`, and the output data type is `torch.bfloat16`):
                    - The parameters `dequant_scale_query`, `dequant_scale_key`, and `dequant_scale_value` must all be provided, and their data types must be `torch.float32`.
                    - Providing `quant_scale_out`, `quant_offset_out`, `dequant_offset_key`, or `dequant_offset_value` is not supported. Otherwise, an error is raised and execution returns.
                    - `query_quant_mode` supports only `pertoken + perhead` mode, and `key_quant_mode` and `value_quant_mode` support only `pertensor` mode.
                    - Supported data layout for `key`, `value`, and `key_rope`: NZ.
            <!-- end id42 -->
            <!-- npu="950" id43 -->
            - <term>Ascend 950PR&950DT products</term>:
                - sparse: When Q_S equals 1, only sparse=0 without mask is supported; when Q_S is greater than 1, only sparse=3 with mask passed is supported.
                - When `query_rope` is configured, the s of `query` must be 1-16. In fp8 full quantization scenarios, n supports 32, 64, and 128; in `torch.int8` scenarios, n supports 1, 2, 4, 8, 16, 32, 64, and 128, and d is 512. In the shape of `query_rope`, b, n, and s are consistent with those of `query`, and d is 64.
                - When `key_rope` is configured, the n of `key` must be 1 and d must be 512. In the shape of `key_rope`, b, n, and s are consistent with those of `key`, and d is 64.
                - sparse: sparse=0 without mask and sparse=3 with mask passed are supported.
                - `key` and `value` support $ND$ input. In `torch.int8` full quantization scenarios, only PANZ input is supported.
                - `input_layout`: BSH, BSND, BNSD, BNSD\_BSND, TND, TND\_NTD.
                - fp8 full quantization supports the following `input_layout` values: BSH, BSND, BNSD, and TND.
                - `torch.int8` full quantization supports the following `input_layout` values: BSH, BSND, TND, BSH_NBSD, BSND_NBSD, and TND_NTD.
                - The `actual_seq_qlen` and `actual_seq_kvlen` parameters are supported.
            <!-- end id43 -->

        - When `D` of `query` is `128`:

            <!-- npu="A3,910b" id44 -->
            - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
                - `input_layout`: BSH, BSND, TND, BNSD, NTD, BSH\_BNSD, BSND\_BNSD, BNSD\_BSND, NTD\_TND.
                - When `query_rope` is configured, D in the shape of `query_rope` must be 64, and the other dimensions must be consistent with those of `query`.
                - When `key_rope` is configured, D in the shape of `key_rope` must be 64, and the other dimensions must be consistent with those of `key`.
                - Left padding, tensorlist, PSE, prefix, fake quantization, full quantization, post-quantization, and empty tensors are not supported.
                - The other constraints are consistent with the comprehensive restrictions in TND and NTD_TND scenarios.
            <!-- end id44 -->
            <!-- npu="950" id45 -->
            - <term>Ascend 950PR&950DT products</term>:
                - When `query_rope` is configured: The `D` dimension of `query_rope` must be `64`, and all other dimensions must match those of `query`. 
                - When `key_rope` is configured: The `D` dimension of `key_rope` must be `64`, and all other dimensions must match those of `key`. 
                - `input_layout`: BSH, BSND, BNSD, BNSD\_BSND, TND, NTD, NTD\_TND.
                - PSE, prefix, fake quantization, full quantization, and post-quantization are not supported.
                - When kv is a tensorlist, b in the shape of `key_rope` must be consistent with the tensorlist length; n and s must be equal to the n and s of each tensor in the tensorlist; and d is 64.
            <!-- end id45 -->

    <!-- npu="A3,910b" id46 -->
    - Comprehensive constraints on the `query`, `key`, and `value` inputs in TND, TND_NTD, and NTD_TND scenarios (<term>Atlas A2 products</term>, <term>Atlas A3 products</term>):
        - `actual_seq_qlen` and `actual_seq_kvlen` must both be provided, and the number of elements in these parameters is used as the batch size (the number of elements must be less than or equal to 4096). The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element.
        - When `D` of `query` is `512`:
            - `sparse_mode`: Valid values: `0`, `3`, `4`, or `9`.
            - `TND` and `TND_NTD` are supported.
            - PagedAttention can be enabled. In this case, the length of `actual_seq_kvlen` must be identical to the batch size of `key` or `value`, representing the actual sequence length of each batch, and each value must not exceed `KV_S`.
            - The `N` dimension of `query` must be `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`, while the `N` dimension of `key` and `value` must be `1`.
            - `query_rope` and `key_rope` must both be provided, and their `D` dimension must be `64`.
            - Left padding, tensorlist, PSE, prefix, fake-quantization, full quantization, post-quantization, and empty tensors are not supported.

        - When `D` of `query` is not `512`:
            - When `query_rope` and `key_rope` are omitted: in `TND` layouts, `Q_D` (`D` dimension of `query`), `K_D` (`D` dimension of `key`), and `V_D` (`D` dimension of `value`) must all be `128`, or `Q_D` and `K_D` must be `192` with `V_D` equal to `128` or `192`; in `NTD` layouts, `V_D` must not be `192`; and in `NTD_TND` layouts, `Q_D` and `K_D` must be `128` or `192`, and `V_D` must be `128`. When `query_rope` and `key_rope` are provided, `Q_D`, `K_D`, and `V_D` must all be `128`. In GQA and PagedAttention scenarios, `V_D` must not be `192`. In MHA scenarios, `Q_D`, `K_D`, and `V_D` must all be `64` or all be `128`, or when `Q_D` and `K_D` are `192`, `V_D` must be `128`.
            - `TND`, `NTD`, and `NTD_TND` layouts are supported.
            - In PagedAttention scenarios, `block_size` must be a multiple of 16 and less than or equal to 1024.
            - In MHA scenarios, only `torch.float16` and `torch.bfloat16` data types are supported. When the data type is `torch.float16`, `inner_precise` supports only values of `0` and `1`. When the data type is `torch.bfloat16`, `inner_precise` supports only a value of `0`. When `sparse_mode` is `0`, `atten_mask` must not be provided. When `sparse_mode` is `3` or `4`, an optimized `atten_mask` matrix must be provided. PagedAttention supports only the `BnBsH` format, where the KV cache layout is `(blockNum, blocksize, H)`, with `blockNum` indicating the number of blocks, `blocksize` the number of tokens per block, and `H` the hidden layer size.
            - Left padding, tensorlist, PSE, prefix, fake-quantization, and full quantization are not supported.
    <!-- end id46 -->
    <!-- npu="950" id47 -->
    - Comprehensive constraints on the `query`, `key`, and `value` inputs in TND scenarios (<term>Ascend 950PR&950DT products</term>):
        - `actual_seq_qlen` and `actual_seq_kvlen` must be passed, and the number of elements of this input parameter is used as the Batch value. The value of each element indicates the sum of the sequence lengths of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element.
        - Left padding, tensorlist, pse, and prefix are not supported.
    <!-- end id47 -->

### GQA fake-quantization + KV NZ format constraints<a name="en-us_topic_0000001832267082_section_gqa_nz_constraint"></a>

- The parameter constraints when KV is in $NZ$ format in GQA fake-quantization scenarios are as follows:
    - `input_layout` must be `BSH`, `BSND`, and `BNSD`.
    - `key` and `value` support only `NZ` layout inputs with shape `[blockNum, KV_N, D/32, blockSize, 32]`.
    - Only KV separation is supported.
    - Only the high-performance mode is supported.
    - `query_rope` and `key_rope` cannot be configured.
    - The supported combinations of `num_query_heads` and `num_key_value_heads` are (10, 1), (64, 8), (80, 8), and (128, 16).

    <!-- npu="A3,910b" id48 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
        - Only page_attention scenarios are supported, and `block_size` supports only 128 or 512;
        - The perchannel and pertoken modes are supported. The `query` data type is fixed to `torch.bfloat16`, and `key` and `value` are fixed to `torch.int8`. The D of `query`, `key`, and `value` supports only 128. The `query` Sequence Length supports only 1-16;
        - dtype of `dequant_scale_key` and `dequant_scale_value`: in perchannel mode, only the `torch.bfloat16` type is supported; in pertoken mode, only the `torch.float32` type is supported;
        - shape of `dequant_scale_key` and `dequant_scale_value`: in perchannel mode, when the layout is BSH, [H] must be passed; when the layout is BNSD, [KV_N,1,D] must be passed; when the output is BSND, [KV_N, D] must be passed. In pertoken mode, [B,KV_S] must be passed, and S must be greater than or equal to the second dimension of `block_table` multiplied by `block_size`;
        - When MTP is `0`, `sparse_mode` can be `0`, and `atten_mask` must not be provided. When MTP is greater than 0 and less than 16, `sparse_mode` can be `3`, and an optimized `atten_mask` matrix must be provided with shape `(2048, 2048)`, or `sparse_mode` can be `9`, requiring a tree mask where the recommended shape is `(B, Q_S, Q_S)` for the `BSH`, `BSND`, or `BNSD` layouts, and a 1D tensor shape `(∑Q_Si²,)` for the `TND` layout where the total length equals the sum of the squared query sequence lengths of all batches.
        - Configuring `dequant_offset_key` and `dequant_offset_value` is not supported.
        - Left padding, tensorlist, PSE, prefix, and post-quantization are not supported.
    <!-- end id48 -->
    <!-- npu="950" id49 -->
    - <term>Ascend 950PR&950DT products</term>:
        - Only the per-channel mode is supported. The query data type is fixed to `torch.bfloat16`, and key and value are fixed to `torch.int8`. The D of `query`, `key`, and `value` supports only 128. The `query` Sequence Length supports only 1-16;
        - shape of `dequant_scale_key` and `dequant_scale_value`: when the layout is BSH, [H] must be passed; when the layout is BNSD, [N,1,D] must be passed; when the output is BSND, [N, D] must be passed;
        - When MTP equals 0, `sparse_mode`=0 without mask passed is supported; when MTP is greater than 0 and less than 16, `sparse_mode`=3 with the optimized `atten_mask` matrix passed is supported, and the shape of the `atten_mask` matrix must be (2048*2048);
        - Tensorlist, PSE, page attention, and post-quantization are not supported.
    <!-- end id49 -->

### `learnable_sink` Constraints<a name="en-us_topic_0000001832267082_section_learnable_sink_constraint"></a>

- Parameter constraints on `learnable_sink`:

    <!-- npu="A3,910b" id50 -->
    - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
        - Only `TND` and `NTD_TND` layouts are supported.
        - The `D` dimension of `value` must be less than or equal to `128`.
    <!-- end id50 -->
    <!-- npu="950" id51 -->
    - <term>Ascend 950PR&950DT products</term>:
        - Only the case where D of `value` equals 64 or 128 is supported;
    <!-- end id51 -->
    - Only non-quantization scenarios are supported.
    - PSE, left padding, shared prefix, and post-quantization are not supported.

### Constraints for Q_S>1 (Full Inference)<a name="en-us_topic_0000001832267082_section_qs_gt1_constraint"></a>

- **When `Q_S` is greater than 1**:
    - Usage constraints for `query`, `key`, and `value` inputs:

        <!-- npu="A3,910b" id52 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: A B axis of less than or equal to 65536 is supported. When the D axis is not aligned to 32 bytes, only a B axis of less than or equal to 128 is supported.
        <!-- end id52 -->
        <!-- npu="950" id53 -->
        - <term>Ascend 950PR&950DT products</term>: A B axis of less than or equal to 65536 is supported.
        <!-- end id53 -->
        - The `N` dimension must be less than or equal to 256, and the `D` dimension must be less than or equal to 512. When `input_layout` is `BSH` or `BSND`, the product of `N` and `D` should be less than 65535.
        - The `S` dimension must be less than or equal to 20971520 (20M). In certain long-sequence scenarios, excessive computation may cause the PFA operator to time out (raising an aicore error with an `errorStr` of "timeout or trap error"). In such cases, slicing along the `S` dimension is recommended. (Note: The computation volume depends on `B`, `S`, `N`, and `D`. Larger values result in higher computation cost). Typical timeout-prone long-sequence scenarios (where the product of `B`, `S`, `N`, and `D` is large) include but are not limited to:
            - B=1, Q\_N=20, Q\_S=2097152, D=256, KV\_N=1, KV\_S=2097152.
            - B=1, Q\_N=2, Q\_S=20971520, D=256, KV\_N=2, KV\_S=20971520.
            - B=20, Q\_N=1, Q\_S=2097152, D=256, KV\_N=1, KV\_S=2097152.
            - B=1, Q\_N=10, Q\_S=2097152, D=512, KV\_N=1, KV\_S=2097152.

        - When the data types of `query`, `key`, or `value` include `torch.int8`, the `D` dimension must be a multiple of 32. When the data types are all `torch.float16` or `torch.bfloat16`, the `D` dimension must be a multiple of 16.
        - Constraints on the `D` dimension:

            <!-- npu="A3,910b" id54 -->
            - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: When the data types of `query`, `key`, or `value` include `torch.int8`, the `D` dimension must be a multiple of 32. When the data types of `query`, `key`, `value`, or `attentionOut` include `torch_npu.int4`, the `D` dimension must be a multiple of 64. When the data types are all `torch.float16` or `torch.bfloat16`, the `D` dimension must be a multiple of 16.
            <!-- end id54 -->
            <!-- npu="950" id55 -->
            - <term>Ascend 950PR&950DT products</term>: Non-quantization scenarios: the types of `query`, `key`, and `value` are all `torch.float16` or `torch.bfloat16`, and D axes from 1 to 512 are all supported. Fake-quantization scenarios: the type of `query` is `torch.float16` or `torch.bfloat16`, and the types of `key` and `value` are `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e4m3fn`/`torch.float4_e2m1fn_x2`/`torch_npu.int4` (`torch.int32`). When the types of `key` and `value` are `torch_npu.int4` (`torch.int32`), the D axis supports only 64 alignment (`torch.int32` supports only D aligned to 8). Full quantization scenarios: per-block full quantization supports D axes from 1 to 128, and MxFP8 full quantization supports only D of 64 or 128.
            <!-- end id55 -->

    - `actual_seq_qlen`:

        <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The valid Sequence Length of each batch in this parameter must not be greater than the Sequence Length of the corresponding batch in `query`. When the passed length of seqlen is 1, each batch uses the same seqlen; when the passed length is greater than or equal to Batch, the first Batch values of seqlen are used. Other lengths are not supported. When the `input_layout` of `query` is TND/NTD_TND, for comprehensive constraints, see [MLA scenario constraints](#mla-scenario-constraints).

        <term>Ascend 950PR&950DT products</term>: This parameter must be a non-negative number, and its meaning and interception conditions differ for different `input_layout` values. Generally, this parameter is optional, and its length is 1 or greater than or equal to the Batch value of `query`. The value in this parameter represents the actual length of each batch and must not be greater than Q_S. When `input_layout` is TND, this parameter must be passed, and the number of elements of this parameter is used as the Batch value. The value of each element indicates the sum of the seqlens of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and negative values must not occur.

    - `actual_seq_kvlen`:

        <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The valid Sequence Length of each batch in this parameter must not be greater than the Sequence Length of the corresponding batch in `key/value`. When the passed length of seqlenKv is 1, each batch uses the same seqlenKv; when the passed length is greater than or equal to Batch, the first Batch values of seqlenKv are used. Other lengths are not supported. When the `input_layout` of `key/value` is TND/NTD_TND, for comprehensive constraints, see [MLA scenario constraints](#mla-scenario-constraints).

        <term>Ascend 950PR&950DT products</term>: When passed in, this parameter must be a non-negative number, and its meaning and interception conditions differ for different `input_layout` values. Generally, this parameter is optional, and the valid seqlenKv of each batch in this parameter must not be greater than the seqlenKv of the corresponding batch in `key/value`. When the passed length of this parameter is 1, each batch uses the same seqlenKv; when the passed length is greater than or equal to Batch, the first Batch values of seqlenKv are used. Other lengths are not supported. When the `input_layout` of `key/value` is TND, this parameter must be passed, and the number of elements of this parameter equals the Batch value. The value of each element indicates the sum of the seqlenKv values of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and negative values must not occur.

    - `sparse_mode`:

        <!-- npu="A3,910b" id56 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
            - The `sparse_mode` parameter supports only values of `0`, `1`, `2`, `3`, `4`, and `9`. Other values result in an error.
        <!-- end id56 -->
        <!-- npu="950" id57 -->
        - <term>Ascend 950PR&950DT products</term>:
            - Currently only the scenarios with the values 0, 1, 2, 3, and 4 are supported. An error is reported for other values.
        <!-- end id57 -->
        - When `sparse_mode` is `0`, if `atten_mask` is `None`, the input parameters `pre_tokens` and `next_tokens` are ignored and internally set to `INT_MAX`.
        - When `sparse_mode` is `2`, `3`, or `4`, the shape of `atten_mask` must be `(S, S)`, `(1, S, S)`, or `(1, 1, S, S)`, where `S` must be `2048`. The provided `atten_mask` must be a lower triangular matrix. If `atten_mask` is omitted or has an invalid shape, an error is raised.
        - When `sparse_mode` is `1`, `2`, or `3`, the input parameters `pre_tokens` and `next_tokens` are ignored and assigned based on the relevant rules.

        <!-- npu="A3,910b" id58 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
            - When `sparse_mode` is `9`, this mode is supported only in MLA scenarios where `query_rope` and `key_rope` are both provided. `atten_mask` must not be `None`. When `input_layout` is `BSH`, `BSND`, or `BNSD`, the shape must be `(B, Q_S, Q_S)`; when `input_layout` is `TND`, the shape must be a 1D tensor `(∑Q_Si²,)` where the total length equals the sum of the squared query sequence lengths of all batches. Left padding, `pse_shift`, and `sharedPrefix` are not supported.
        <!-- end id58 -->

    - PagedAttention scenarios:
        - PagedAttention can be enabled only when `block_table` exists and is valid, and `key` and `value` are arranged in contiguous memory based on the indices in `block_table`. The supported data types for `key` and `value` are `torch.float16` and `torch.bfloat16`. In this case, `input_layout` of `key` and `value` is ignored. The `block_table` contains block IDs. Their validity is not verified and must be ensured by the user.
        - `block_size` is a user-defined parameter that affects PagedAttention performance. When PagedAttention is enabled, `block_size` must range from 128 to 512 and must be a multiple of 128. Generally, PagedAttention improves throughput but reduces performance.

        - In PagedAttention scenarios, when the input KV cache layout is `(blocknum, blocksize, H)` and the product of `KV_N` and `D` is greater than 65535, execution is blocked and an error is raised due to hardware constraints. This can be resolved by enabling GQA (to reduce `KV_N`) or by changing the KV cache layout to `(blocknum, KV_N, blocksize, D)`. When the `input_layout` of `query` is `BNSD` or `TND`, the KV cache supports both `(blocknum, blocksize, H)` and `(blocknum, KV_N, blocksize, D)` layouts. When it is `BSH` or `BSND`, only `(blocknum, blocksize, H)` is supported. The value of `blocknum` must not be less than the total number of blocks required across all batches, calculated from `actual_seq_kvlen` and `block_size`. The shapes of `key` and `value` must be identical.

        <!-- npu="A3,910b" id59 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Page attention does not support fake-quantization scenarios.
        <!-- end id59 -->
        <!-- npu="950" id60 -->
        - <term>Ascend 950PR&950DT products</term>: In page attention fake-quantization scenarios, `query` can be `torch.float16`/`torch.bfloat16`, and the dtype of `key` and `value` can be `torch.int8`/`torch_npu.hifloat8`/`torch.float8_e4m3fn`/`torch.float4_e2m1fn_x2`/`torch_npu.int4` (`torch.int32`). When the kv cache is 5D, the kv cache layout is (blocknum, KV_N, D/16, blocksize, 16). In addition, when the dtype of `key` and `value` is `torch.int32`, the kv cache layout is (blocknum, KV_N, D/2, blocksize, 2).
        <!-- end id60 -->
        - PagedAttention does not support fake-quantization or tensorlist.
        - In PagedAttention scenarios, `actual_seq_kvlen` must be provided.
        - In PagedAttention scenarios, `block_table` must be a 2D tensor. Its first dimension must equal `B`, and its second dimension must be no less than `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actual_seq_kvlen` across batches).

        <!-- npu="A3,910b" id61 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
            - In PagedAttention scenarios, two layout formats are supported, and the supported data types are `torch.float32` and `torch.bfloat16`. Cases where `query` is `torch.int8` are not supported.
        <!-- end id61 -->
        <!-- npu="950" id62 -->
        - <term>Ascend 950PR&950DT products</term>:
            - In page attention scenarios, two formats and `torch.float32`/`torch.bfloat16` are supported. Scenarios where the input `query` is `torch.int8` support only PA_NZ.
        <!-- end id62 -->
        - When PagedAttention is enabled, the input sequence length `KV_S` must be greater than or equal to the product of `maxBlockNumPerSeq` and `block_size` in the following scenarios:
            - When `atten_mask` is provided and its shape is identical to `(B, 1, Q_S, KV_S)`.
            - When `pse_shift` is provided and its shape is identical to `(B, Q_N, Q_S, KV_S)`.

            <!-- npu="950" id63 -->
            - <term>Ascend 950PR&950DT products</term>:
                - When fake-quantization per-token mode is enabled: the shapes of the input parameters `antiquantScale` and `antiquantOffset` are both (2, B, S).
                - When per-token combined with per-head mode is enabled: the shapes of the two parameters are both (B, N, S), the data type is fixed to `torch.float32`, and this is supported when the data types of key and value are `torch.int8` or `torch_npu.int4` (`torch.int32`).
            <!-- end id63 -->

    - The input parameters `quant_scale_out` and `quant_offset_out` support `pertensor` or `perchannel` quantization, and their data types can be `torch.float32` or `torch.bfloat16`. If `quant_offset_out` is provided, its data type and shape must be identical to those of `quant_scale_out`. When the input data type is `torch.bfloat16`, both `torch.float32` and `torch.bfloat16` are supported. Otherwise, only `torch.float32` is supported. In `perchannel` scenarios, when the output layout is `BSH`, the product of all dimensions of `quant_scale_out` must equal `H`. For other layouts, the product must equal `Q_N * D`. The recommended shape for `quant_scale_out` is `(1, 1, H)` or `(H,)` for the `BSH` output layout, `(1, Q_N, 1, D)` or `(Q_N, D)` for the `BNSD` output layout, and `(1, 1, Q_N, D)` or `(Q_N, D)` for the `BSND` output layout.
    - When the output data type is `torch.int8` and `quant_scale_out` and `quant_offset_out` are in `perchannel` mode, Ring Attention, and a non-32-byte-aligned `D` dimension are not supported.
    - When the output data type is `torch.int8`, scenarios where `sparse_mode` is `band` and `pre_tokens` or `next_tokens` are negative are not supported.
    - Usage constraints for the `pse_shift` feature:

        - Supported when the data type of `query` is `torch.float16`, `torch.bfloat16`, or `torch.int8`.
        - When `query`, `key`, and `value` are all `torch.float16` and `pse_shift` is provided, high-precision mode is enforced. The constraints are the same as those of high-precision mode.
        - `Q_S` must be greater than or equal to the sequence length (`S`) of `query`, and `KV_S` must be greater than or equal to the sequence length (`S`) of `key`.

        <!-- npu="950" id64 -->
        - <term>Ascend 950PR&950DT products</term>: There is no alignment restriction in non-quantization scenarios.
        <!-- end id64 -->

    - When the output data type is `torch.int8`, if `quant_offset_out` is a non-`None` and non-empty tensor, and `sparse_mode`, `pre_tokens`, and `next_tokens` meet the execution blocking conditions, certain matrix rows will be excluded from computation. This causes computation errors, and execution will be blocked:
        - When `sparse_mode` is `0`, if `atten_mask` is not `None` and, for each batch, `actual_seq_qlen-actual_seq_kvlen - pre_tokens > 0`, or `next_tokens < 0`, the execution will be blocked.
        - When `sparse_mode` is `1` or `2`, the execution will not be blocked.
        - When `sparse_mode` is `3`, if, for each batch, `actual_seq_kvlen - actual_seq_qlen < 0`, the execution will be blocked.
        - When `sparse_mode` is `4`, if `pre_tokens < 0` or, for each batch, `next_tokens + actual_seq_kvlen - actual_seq_qlen < 0`, the execution will be blocked.

    - Constraints on KV fake-quantization parameter separation:
        - When both fake-quantization parameters and KV-separated quantization parameters are provided, the KV-separated quantization parameters take precedence.    
        - `key_quant_mode` and `value_quant_mode` must have identical values.
        - `dequant_scale_key` and `dequant_scale_value` must either both be omitted or both be provided. Likewise, `dequant_offset_key` and `dequant_offset_value` must either both be omitted or both be provided.
        - When `dequant_scale_key` and `dequant_scale_value` are both provided, their shapes must be identical. When `dequant_offset_key` and `dequant_offset_value` are both provided, their shapes must be identical. 

        <!-- npu="A3,910b" id65 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
            - Only `pertoken` and `perchannel` modes are supported. In `pertoken` mode, the shapes of both parameters must be `(B, KV_S)`, and their data types must be `torch.float32`. In `perchannel` mode, the shapes of both parameters must be `(KV_N, D)` or `(H)`, and their data types must be `torch.bfloat16`, where `H` equals the product of `KV_N` and `D`.
            - When `dequant_scale_key` and `dequant_scale_value` are both provided, the sequence length `S` of `query` must be less than or equal to 16. The data type of `query` must be `torch.bfloat16`, the data types of `key` and `value` must be `torch.int8`, and the output data type must be `torch.bfloat16`. This configuration does not support tensorlist or PagedAttention.
        <!-- end id65 -->
        <!-- npu="950" id66 -->
        - <term>Ascend 950PR&950DT products</term>:
            - Nine modes are supported: per-channel, per-tensor, per-token, per-tensor combined with per-head, per-token combined with per-head, per-token using page attention mode to manage scale/offset, per-token combined with per-head using page attention mode to manage scale/offset, and `key` supporting per-channel combined with `value` supporting per-token and per-token-group.
            - When neither `dequant_scale_key` nor `dequant_scale_value` is empty, except in scenarios where `key_quant_mode` is 0 and `value_quant_mode` is 1, their shapes must be consistent. When neither `dequant_offset_key` nor `dequant_offset_value` is empty, except in scenarios where `key_quant_mode` is 0 and `value_quant_mode` is 1, their shapes must be consistent.
        <!-- end id66 -->
        - The following table describes quantization modes for scale and offset management.

            > [!NOTE]
            > The scale and offset parameters are `dequant_scale_key`, `dequant_scale_value`, `dequant_offset_key`, and `dequant_offset_value`.
            > When Q_S is greater than 1, different products support different quantization modes:
            > <!-- npu="A3,910b" id81 -->
            > - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only the per-channel and per-token modes are supported.
            > <!-- end id81 -->
            > <!-- npu="950" id82 -->
            > - <term>Ascend 950PR&950DT products</term>: All the quantization modes below are supported.
            > <!-- end id82 -->

            | Quantization Mode | scale and offset conditions in this scenario | `key` and `value` conditions in this scenario |
            | --- | --- | --- |
            | per-channel mode | The shapes of the two parameters support (N, 1, D), (N, D), (H), (1, N, 1, D), (1, N, D), and (1, H), and the data type is the same as that of `query`. | <li><term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only the `key` and `value` data types of `torch.int8` are supported.</li><li><term>Ascend 950PR&950DT products</term>: Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`), `torch.int8`, `torch_npu.hifloat8`, or `torch.float8_e4m3fn`. When the type is `torch_npu.hifloat8` or `torch.float8_e4m3fn`, `dequant_offset` is not supported.</li> |
            | per-token mode | The shapes of the two parameters are both (B, S), and the data type is fixed to `torch.float32`. | <li><term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Only the `key` and `value` data types of `torch.int8` are supported.</li><li><term>Ascend 950PR&950DT products</term>: Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`), `torch.int8`, or `torch.float8_e4m3fn`.</li> |
            | per-tensor mode | The shapes of the two parameters are both (1,), and the data type is the same as that of `query`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
            | per-tensor combined with per-head mode | The shapes of the two parameters are both (N,), and the data type is the same as that of `query`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
            | per-token combined with per-head mode | The shapes of the two parameters are both (B, N, S), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
            | per-token combined with the page attention mode | The shapes of the two parameters are both (blocknum, blocksize), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch.int8` or `torch.float8_e4m3fn`. |
            | per-token combined with per head and using the page attention mode | The shapes of the two parameters are both (blocknum, N, blocksize), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch.int8`. |
            | `key` supports per-channel combined with `value` supports per-token mode | For `key` supporting per-channel, the shapes of the two parameters can be (N, 1, D), (N, D), (H), (1, N, 1, D), (1, N, D), or (1, H), and the data type is the same as that of `query`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
            | For `value` supporting per-token, the shapes of the two parameters are both (B, S) and the data type is fixed to `torch.float32`. | | |
            | per-token-group mode | The shape of `dequant_scale` is (1, B, N, S, D/32), the data type is fixed to `torch.float8_e8m0fnu`, and `dequant_offset` is not supported. | Supported when the data types of `key` and `value` are `torch.float4_e2m1fn_x2`. |

    <!-- npu="950" id67 -->
    - MXFP8 full quantization:

        - <term>Ascend 950PR&950DT products</term>:
            - The data types of `query`, `key`, and `value` support `torch.float8_e4m3fn`.
            - `query_quant_mode`, `key_quant_mode`, and `value_quant_mode` are 6, 6, and 8, respectively.
            - The data types of `dequant_scale_query`, `dequant_scale_key`, and `dequant_scale_value` are fixed to `torch.float8_e8m0fnu`.
            - `input_layout` supports only TND, and both PageAttention and non-PageAttention scenarios are supported. The shape of `query` is (Q_T, Q_N, D). `key` and `value` are passed in according to the shape of the PageAttention scenario. The kv cache layout supports (blocknum, KV_N, blocksize, D) or (blocknum, KV_N, D/D0, blocksize, D0), where blocksize is fixed at 512.
            - When Q_S \* G > 80, the shape of `dequant_scale_query` is recommended to be (Q_T, Q_N, D/64, 2); when Q_S * G <= 80, the shape of the input parameter `dequant_scale_query` is recommended to be (KV_N, Q_T, G, D/64, 2). In non-PagedAttention scenarios, the shape of `dequant_scale_key` is (KV_T, KV_N, D/64, 2), and the shape of `dequant_scale_value` is (KV_T/64, KV_N, D, 2). In PagedAttention scenarios where the kv cache layout is BnNBsD, the shape of `dequant_scale_key` is (Bn, KV_N, Bs, D/64, 2), and the shape of `dequant_scale_value` is (Bn, KV_N, Bs/64, D, 2). When the kv cache layout is $NZ$, the shape of `dequant_scale_key` is (Bn, KV_N, Bs/16, D/64, 16, 2), and the shape of `dequant_scale_value` is (Bn, KV_N, D/16, Bs/64, 16, 2).
            - `sparse_mode` 0 and 3 are supported, and when `sparse_mode` is 0, the mask matrix must not be passed.
            - `inner_precise` 0 and 1 are supported.
            - softmax_lse is supported.
            - In PagedAttention scenarios during single-operator direct calls, when the kv cache layout is BnNBsD or $NZ$, `key`, `value`, `dequant_scale_key`, `dequant_scale_value`, and `key_rope` support non-contiguity on axis 0 and axis 1.
    <!-- end id67 -->

    <!-- npu="950" id68 -->
    - GQA FP8 full quantization:

        - <term>Ascend 950PR&950DT products</term>:
            - The data types of `query`, `key`, and `value` support `torch.float8_e4m3fn`.
            - `query_quant_mode`, `key_quant_mode`, and `value_quant_mode` are 3, 3, and 2, respectively.
            - The data types of `dequant_scale_query`, `dequant_scale_key`, and `dequant_scale_value` are fixed to `torch.float32`.
            - `input_layout` supports NTD_TND, and only PageAttention scenarios are supported. The shape of `query` is (Q_N, Q_T, D), and the shapes of the input parameters `key` and `value` are (blocknum, KV_N, blocksize, D), where blocksize is fixed at 128 and D is fixed at 128.
            - The shape of `dequant_scale_query` is (Q_N, Q_T), the shape of `dequant_scale_key` is (blocknum, KV_N, blocksize), and the shape of `dequant_scale_value` is (KV_N).
            - `sparse_mode` 0 and 3 are supported, and when `sparse_mode` is 0, the mask matrix must not be passed.
            - `inner_precise` 0 and 1 are supported.
            - softmax_lse is supported.
            - PScale static quantization is supported. The shape of PScale is (1), the data type is `torch.float32`, and it is optional. If not passed, the default value is 1.0.
            - During single-operator direct calls in PagedAttention scenarios, `key`, `value`, and `dequant_scale_key` support only non-contiguous scenarios on axis 0 and axis 1.
            - rope, left padding, tensorlist, prefix, and pse are not supported.
    <!-- end id68 -->

### Constraints for Q_S=1 (Incremental Inference)<a name="en-us_topic_0000001832267082_section_qs_eq1_constraint"></a>

- **When `Q_S` is equal to 1**:
    - Usage constraints for `query`, `key`, and `value` inputs:
        - The `B` dimension must be less than or equal to 65536, the `N` dimension must be less than or equal to 256, the `S` dimension must be less than or equal to 262144, and the `D` dimension must be less than or equal to 512.
        - Scenarios where the input data types of `query`, `key`, and `value` are all `torch.int8` are not supported.
        - In `torch_npu.int4` (`torch.int32`) fake-quantization scenarios, PyTorch graph-mode execution only supports inputs where KV `torch_npu.int4` values are packed into `torch.int32` tensors. (Using dynamic quantization to generate data in `torch_npu.int4` format is recommended, as each `torch.int32` value encapsulates eight `torch_npu.int4` values).
        - In `torch_npu.int4` (`torch.int32`) fake-quantization scenarios, when KV `torch_npu.int4` values are packed into `torch.int32` inputs, the `N`, `D`, or `H` dimensions of `key` and `value` must be one-eighth of their actual values. Additionally, `torch_npu.int4` fake-quantization requires the `D` dimension to be a multiple of 64, whereas the underlying `torch.int32` tensor requires the `D` dimension to be a multiple of 8.

        <!-- npu="950" id69 -->
        - <term>Ascend 950PR&950DT products</term>: When the input types of `key` and `value` are `torch.float4_e2m1fn_x2`/`torch_npu.int4` (`torch.int32`), the D axis of `query` and the D axes of `key` and `value` must be aligned to 64 (`torch.int32` supports only D of `key` and `value` aligned to 8).
        <!-- end id69 -->

    - `actual_seq_qlen`:

        <!-- npu="A3,910b" id70 -->
        - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: When `input_layout` of `query` is not `TND` and `Q_S` is 1, this parameter is ignored. For details about the overall constraints when `input_layout` of `query` is `TND` or `TND_NTD`, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
        <!-- end id70 -->
        <!-- npu="950" id71 -->
        - <term>Ascend 950PR&950DT products</term>: This parameter does not take effect when no rope parameter is input. It takes effect when a rope parameter is input. When passed in, it must be a non-negative number, and its meaning and interception conditions differ for different `input_layout` values. Generally, this parameter is optional, and its length is 1 or greater than or equal to the Batch value of `query`. The value in this parameter represents the actual length of each batch and must not be greater than Q_S. When `input_layout` is TND, this parameter must be passed, and the number of elements of this parameter is used as the Batch value. The value of each element indicates the sum of the seqlens of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and negative values must not occur.
        <!-- end id71 -->

    - `actual_seq_kvlen`:

        <!-- npu="A3,910b" id72 -->
        - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>: The valid sequence length of each batch in this parameter must be less than or equal to the sequence length of the corresponding batch in `key` and `value`. If the length of this parameter is 1, all batches use the same sequence length. If the length is greater than or equal to the batch size, only the first `batch_size` elements are used. Other lengths are not supported. For details about the overall constraints when `input_layout` of `key` and `query` is `TND` or `TND_NTD`, see [MLA Constraints](#en-us_topic_0000001832267082_section_mla_constraint).
        <!-- end id72 -->
        <!-- npu="950" id73 -->
        - <term>Ascend 950PR&950DT products</term>: This parameter must be a non-negative number, and its meaning and interception conditions differ for different `input_layout` values. Generally, this parameter is optional, and the valid Sequence Length of each batch in this parameter must not be greater than the seqlenKv of the corresponding batch in `key/value`. When the passed length of this parameter is 1, each batch uses the same seqlenKv; when the passed length is greater than or equal to Batch, the first Batch values of seqlenKv are used. Other lengths are not supported. When `input_layout` is TND/TND_NTD, this parameter must be passed. In non-PA scenarios, the b-th value indicates the accumulated length of the S axes of the first b batches, and the values must be arranged in increasing order (greater than or equal to the previous value). The number of elements of this parameter represents the total number of batches. In PA scenarios, its length equals the Batch value of `key/value`, it represents the actual length of each batch, and the values must not be greater than KV_S.
        <!-- end id73 -->

    - PagedAttention scenarios:
        - PagedAttention can be enabled only when `block_table` exists and is valid, and `key` and `value` are arranged in a contiguous memory space based on the indices in `block_table`. In this case, the `input_layout` parameter of `key` and `value` is ignored.

        <!-- npu="A3,910b" id74 -->
        - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
            - The data types of `key` and `value` can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
            - Scenarios where `query` is `torch.bfloat16` or `torch.float16` while `key` and `value` are `torch_npu.int4` (`torch.int32`) are not supported.
        <!-- end id74 -->
        <!-- npu="950" id75 -->
        - <term>Ascend 950PR&950DT products</term>: The data types of `key` and `value` can be `torch.float16`, `torch.bfloat16`, `torch.int8`, `torch_npu.int4` (`torch.int32`), `torch.float8_e4m3fn`, `torch_npu.hifloat8`, or `torch.float4_e2m1fn_x2`.
        <!-- end id75 -->
        - In this scenario, `block_size` is a user-defined parameter whose value affects the performance of page attention. Generally, page attention can improve throughput but reduces performance.

            <!-- npu="A3,910b" id76 -->
            - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: When the input types of `key` and `value` are `torch.float16` or `torch.bfloat16`, alignment to 16 is required; when the input type of `key` and `value` is `torch.int8`, alignment to 32 is required. The value 128 is recommended.
            <!-- end id76 -->
            <!-- npu="950" id77 -->
            - <term>Ascend 950PR&950DT products</term>: When the input types of `key` and `value` are `torch.float16`/`torch.bfloat16`, alignment to 16 is required; when the input types of `key` and `value` are `torch.int8`/`torch.float8_e4m3fn`/`torch_npu.hifloat8`, alignment to 32 is required; when the input types of `key` and `value` are `torch.float4_e2m1fn_x2`/`torch_npu.int4` (`torch.int32`), alignment to 64 is required.
            <!-- end id77 -->

        - The product of all dimensions of the tensors corresponding to `key` and `value` must not exceed the representable range of `torch.int32`.
        - In PagedAttention scenarios, `block_table` must be a 2D tensor. Its first dimension must equal `B`, and its second dimension must be no less than `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actual_seq_kvlen` across batches).

        <!-- npu="A3,910b" id78 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
            - In page attention scenarios, when the `input_layout` of `query` is BNSD or TND, the kv cache layout supports the two formats (blocknum, blocksize, H) and (blocknum, KV_N, blocksize, D); when the `input_layout` of `query` is BSH or BSND, the kv cache layout supports only the format (blocknum, blocksize, H). blocknum must not be less than the sum of the numbers of blocks of each batch calculated from `actual_seq_kvlen` and `block_size`. In addition, the shapes of `key` and `value` must be consistent.
        <!-- end id78 -->
        <!-- npu="950" id79 -->
        - <term>Ascend 950PR&950DT products</term>:
            - In PagedAttention scenarios, when `input_layout` of `query` is `BNSD` or `TND`, the KV cache supports three layouts: `(blocknum, blocksize, H)`, `(blocknum, KV_N, blocksize, D)`, and `(blocknum, KV_N, D/16, blocksize, 16)`. When `input_layout` of `query` is `BSH` or `BSND`, only two layouts are supported: `(blocknum, blocksize, H)` and `(blocknum, KV_N, D/16, blocksize, 16)`. The value of `blocknum` must not be less than the total number of blocks required across all batches, calculated from `actual_seq_kvlen` and `block_size`. The shapes of `key` and `value` must be identical.
            - In page attention fake-quantization scenarios, when the kv cache is 5D, the kv cache layout is (blocknum, KV_N, D/16, blocksize, 16). In addition, when the dtype of `key` and `value` is `torch.int32`, the kv cache layout is (blocknum, KV_N, D/2, blocksize, 2).
        <!-- end id79 -->
        - In PagedAttention scenarios, the `(blocknum, KV_N, blocksize, D)` layout format of the KV cache typically provides better performance than `(blocknum, blocksize, H)`, and is therefore recommended.
        - In PagedAttention scenarios, when the KV cache layout format is (blocknum, blocksize, H) and the product of `numKvHeads` and `headDim` exceeds 64K, hardware instruction constraints block execution and raise an error. This can be resolved by enabling GQA (to reduce `numKvHeads`) or by adjusting the KV cache layout format to `(blocknum, numKvHeads, blocksize, D)`.
        - In PagedAttention scenarios, the product of all dimensions of the tensors corresponding to `key` and `value` must not exceed the representable range of `torch.int32`.

    - Constraints on KV fake-quantization parameter separation:
        - Except for scenarios where `key_quant_mode` is 0 and `value_quant_mode` is 1, the values of `key_quant_mode` and `value_quant_mode` must be identical.  
        - `dequant_scale_key` and `dequant_scale_value` must either both be omitted or both be provided. Likewise, `dequant_offset_key` and `dequant_offset_value` must either both be omitted or both be provided.
        - Except for scenarios where `key_quant_mode` is 0 and `value_quant_mode` is 1, when `dequant_scale_key` and `dequant_scale_value` are both provided, their shapes must be identical; when `dequant_offset_key` and `dequant_offset_value` are both provided, their shapes must also be identical.
        - Post-quantization is not supported in `torch_npu.int4` (`torch.int32`) fake-quantization scenarios.
        - The following table describes quantization modes for scale and offset management.
    
            > [!NOTE]   
            > The scale and offset parameters are `dequant_scale_key`, `dequant_scale_value`, `dequant_offset_key`, and `dequant_offset_value`.
            > When Q_S equals 1, different products support different quantization modes:
            > <!-- npu="A3,910b" id83 -->
            > - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>: The per-channel mode, per-tensor mode, per-token mode, per-tensor combined with per-head mode, per-token combined with per-head mode, per-token combined with the page attention mode, per-token combined with per head and using the page attention mode, and the mode where `key` supports per-channel combined with value supporting per-token are supported.
            > <!-- end id83 -->
            > <!-- npu="950" id84 -->
            > - <term>Ascend 950PR&950DT products</term>: All the quantization modes below are supported.
            > <!-- end id84 -->

            | Quantization Mode | scale and offset conditions in this scenario | `key` and value conditions in this scenario |
            | --- | --- | --- |
            | per-channel mode | The shapes of the two parameters support (1, KV_N, 1, D), (1, KV_N, D), and (1, H), and the data type is the same as that of `query`. | <li><term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`.</li><li><term>Ascend 950PR&950DT products</term>: Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`), `torch.int8`, `torch_npu.hifloat8`, or `torch.float8_e4m3fn`. When the type is `torch_npu.hifloat8` or `torch.float8_e4m3fn`, `dequant_offset` is not supported.</li> |
            | per-tensor mode | The shapes of the two parameters are (1,), and the data type is the same as that of `query`. | Supported when the data types of `key` and `value` are `torch.int8`. |
            | per-token mode | The shapes of the two parameters are both (1, B, KV_S), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
            | per-tensor combined with per-head mode | The shapes of the two parameters are (KV_N,), and the data type is the same as that of `query`. | Supported when the data types of `key` and `value` are `torch.int8`. |
            | per-token combined with per-head mode | The shapes of the two parameters are both (B, KV_N, KV_S), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. |
            | per-token combined with the page attention mode | The shapes of the two parameters are (blocknum, blocksize), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch.int8`. |
            | per-token combined with per head and using the page attention mode | The shapes of the two parameters are (blocknum, KV_N, blocksize), and the data type is fixed to `torch.float32`. | Supported when the data types of `key` and `value` are `torch.int8`. |
            | `key` supports per-channel combined with `value` supports per-token mode | <li>For `key` supporting per-channel, the shapes of the two parameters can be (1, KV_N, 1, D), (1, KV_N, D), or (1, H), and the data type is the same as that of `query`.</li><li>For `value` supporting per-token, the shapes of the two parameters are both (1, B, KV_S) and the data type is fixed to `torch.float32`.</li> | <li><term>Atlas A2 products</term>, <term>Atlas A3 products</term>: Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`. When the data types of `key` and `value` are `torch.int8`, only the case where the dtype of query and the output is `torch.float16` is supported.</li><li><term>Ascend 950PR&950DT products</term>: Supported when the data types of `key` and `value` are `torch_npu.int4` (`torch.int32`) or `torch.int8`.</li> |
            | per-token-group mode | The shape of `dequant_scale` is (1, B, KV_N, KV_S, D/32), and the data type is fixed to `torch.float8_e8m0fnu`. `dequant_offset` is not supported. | Supported when the data types of `key` and `value` are `torch.float4_e2m1fn_x2`. |

    - Usage restrictions of the `pse_shift` function are as follows:
        - The data type of `pse_shift` must be consistent with the data type of `query`.

        <!-- npu="A3,910b" id80 -->
        - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
            - Only D-axis alignment is supported, that is, the D axis must be divisible by 16.
        <!-- end id80 -->

## Example<a name="en-us_topic_0000001832267082_section14459801435"></a>

### Example 1: MHA + non-quantization + BNSD (PromptFlashAttention)

**Scenario description**: `query/key/value` are all `torch.float16`, this is a non-quantization scenario with the BNSD layout. There are no BMM1/BMM2 quantization parameters, and no output quantization is performed.

**Key parameters**:

- `num_query_heads=8`: Required. Number of heads.
- `num_key_value_heads`: Optional. If empty, it indicates that the single-head size is the same as `num_query_heads`, that is, the single-head attention mechanism.
- `input_layout="BNSD"`: Required. Layout format of the input data.
- `softmax_scale=1/sqrt(D)`: Required. Prevents the dot product value from being too large and causing the softmax gradient to vanish.
- `pre_tokens=65535, next_tokens=65535`: Optional. Indicates that the attention range is not limited.

**Related constraints**: [General constraints](#general-constraints)

```python
import torch
import torch_npu
import math
# Generate random data and send it to the NPU
q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
softmax_scale = 1/math.sqrt(128.0)
actseqlen = [164]
actseqlenkv = [1024]

# Call the FIA operator
out, _ = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, 
         actual_seq_qlen = actseqlen, actual_seq_kvlen = actseqlenkv,
         num_query_heads = 8, input_layout = "BNSD", softmax_scale = softmax_scale, pre_tokens=65535, next_tokens=65535)

print(out)
```

The output `out` of the preceding code is similar to the following:

```text
tensor([[[[ 0.0219,  0.0201,  0.0049,  ...,  0.0118, -0.0011, -0.0140],
        [ 0.0294,  0.0256, -0.0081,  ...,  0.0267,  0.0067, -0.0117],
        [ 0.0285,  0.0296,  0.0011,  ...,  0.0150,  0.0056, -0.0062],
        ..
        [ 0.0177,  0.0194, -0.0060,  ...,  0.0226,  0.0029, -0.0039],
        [ 0.0180,  0.0186, -0.0067,  ...,  0.0204, -0.0045, -0.0164],
        [ 0.0176,  0.0288, -0.0091,  ...,  0.0304,  0.0033, -0.0173]]]],
        device='npu:0', dtype=torch.float16)
```

### Example 2: GQA + KV NZ + PageAttention

**Scenario description**: In the decode phase, Q_S=1 (token-by-token incremental inference). In GQA, KV_N=1 < Q_N=10. The KV cache is managed by PageAttention, `key`/`value` are `torch.int8` and are dequantized to `torch.float16` by fake quantization for computation. When the NZ format is selected, only the high-performance mode is supported.

**Key parameters**:

- `num_query_heads=10, num_key_value_heads=1`: Required. GQA supported combinations: (10,1)/(64,8)/(80,8)/(128,16).
- `block_size=128`: Required. Enables PageAttention. Only 128 or 512 is supported.
- `dequant_scale_value, dequant_scale_key`: Required when per-channel fake quantization is used. The dtype is fixed to `torch.bfloat16`.
- `key_quant_mode=0,value_quant_mode=0`: Required when per-channel fake quantization is used. Fixed to 0.
- `key, value`: Required. The type is `torch.int8`, and the NZ format is [blockNum, KV_N, D/32, blockSize, 32].
- `inner_precise=1`: Required. Under the KV NZ format constraint, only the high-performance mode is supported.

**Related constraints**: [GQA fake-quantization + KV NZ format constraints](#gqa-fake-quantization--kv-nz-format-constraints), [incremental inference constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint)

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 10, 1, 128
KV_N = 1            # Combination (10, 1)
block_size = 128     # only 128 or 512 is supported
KV_S = 1024          # key/value sequence length

# Calculate the number of blocks
block_num = (KV_S + block_size - 1) // block_size

# query: bfloat16, BNSD
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()

# key/value: int8, NZ format [blockNum, KV_N, D/32, blockSize, 32]
k = torch.randint(-128, 127, (block_num, KV_N, D // 32, block_size, 32), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (block_num, KV_N, D // 32, block_size, 32), dtype=torch.int8).npu()

# dequant_scale: perchannel mode, bfloat16
# When layout=BNSD, the shape is [KV_N, 1, D]
dequant_scale_key = torch.randn(KV_N, 1, D, dtype=torch.bfloat16).npu()
dequant_scale_value = torch.randn(KV_N, 1, D, dtype=torch.bfloat16).npu()

# block_table: page attention mapping table
block_table = torch.arange(block_num, dtype=torch.int32).reshape(B, block_num).npu()

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(q, k, v,
         dequant_scale_key=dequant_scale_key, dequant_scale_value=dequant_scale_value, 
         key_quant_mode=0, value_quant_mode=0, 
         block_table=block_table, actual_seq_kvlen=[KV_S],
         num_query_heads=Q_N, num_key_value_heads=KV_N,
         input_layout="BNSD", softmax_scale=softmax_scale, block_size=block_size,
         inner_precise=1 # High performance mode
         )

print(out)
```

### Example 3: MLA + TND constraints

**Scenario description**: MLA scenario, where the RoPE position encoding information is passed separately. TND layout, D=128.

**Key parameters**:

- `input_layout="TND"`: Required in TND scenarios.
- `actual_seq_qlen, actual_seq_kvlen`: Required in TND scenarios.
- `query_rope, key_rope`: Required. MLA mode parameters.

**Related constraints**: [MLA scenario constraints](#mla-scenario-constraints)

```python
import torch
import torch_npu
import math

Q_N, KV_N = 8, 8
D = 128
D_rope = 64

# TND scenario: T is the sum of the seqlens of all batches
# actual_seq_qlen indicates the cumulative seqlen of each batch
# Single-batch scenario: T = S
S_q = 164
S_kv = 1024

# query:  (T, Q_N, D)     → (164, 8, 128)
# key:    (T, KV_N, D)    → (1024, 8, 128)
# value:  (T, KV_N, D)    → (1024, 8, 128)
q = torch.randn(S_q, Q_N, D, dtype=torch.float16).npu()
k = torch.randn(S_kv, KV_N, D, dtype=torch.float16).npu()
v = torch.randn(S_kv, KV_N, D, dtype=torch.float16).npu()

# query_rope: (T, Q_N, 64)
# key_rope:   (T, KV_N, 64)
query_rope = torch.randn(S_q, Q_N, D_rope, dtype=torch.float16).npu()
key_rope   = torch.randn(S_kv, KV_N, D_rope, dtype=torch.float16).npu()

# In TND scenarios, actual_seq_qlen / actual_seq_kvlen must be passed
# The element is the sum of the S values of the current batch and all preceding batches (for a single batch, the S value is given directly)
actual_seq_qlen = [S_q]
actual_seq_kvlen = [S_kv]

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v,
    query_rope=query_rope,
    key_rope=key_rope,
    actual_seq_qlen=actual_seq_qlen,       # must be passed in TND scenarios
    actual_seq_kvlen=actual_seq_kvlen,     # must be passed in TND scenarios
    num_query_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="TND",                    # TND layout
    softmax_scale=softmax_scale,
    sparse_mode=0,                         # do not pass atten_mask
    inner_precise=1                        # only 0 or 1 is supported
)

print(out)  #tensor([[[...]]], device='npu:0', dtype=torch.float16)
```

### Example 4: learnable_sink + non-quantization constraints

**Scenario description**: Adds the **learnable_sink** parameter and notes the other constraint conditions.

**Key parameters**:

- `learnable_sink`: Required. The `learnable_sink` parameter constrains key parameters.
- `actual_seq_qlen, actual_seq_kvlen`: Required in TND scenarios.
- `input_layout="TND"`: Required. Only TND and NTD_TND scenarios are supported.

**Related constraints**: [learnable_sink constraints](#learnable_sink-constraints)

```python
import torch
import torch_npu
import math

S_q = 164
S_kv = 1024
Q_N, KV_N = 8, 8
D = 128               # D of value <= 128

# query/key/value: TND layout, bfloat16
q = torch.randn(S_q, Q_N, D, dtype=torch.bfloat16).npu()
k = torch.randn(S_kv, KV_N, D, dtype=torch.bfloat16).npu()
v = torch.randn(S_kv, KV_N, D, dtype=torch.bfloat16).npu()

# learnable_sink: shape=(Q_N,), dtype=bfloat16
learnable_sink = torch.randn(Q_N, dtype=torch.bfloat16).npu()

# In TND scenarios, actual_seq_qlen / actual_seq_kvlen must be passed
actual_seq_qlen = [S_q]
actual_seq_kvlen = [S_kv]

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v,
    learnable_sink=learnable_sink,          # Learnable Sink Token
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_kvlen=actual_seq_kvlen,
    num_query_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="TND",
    softmax_scale=softmax_scale,
    sparse_mode=0
)

print(out)
```

### Example 5: `torch.int8` post-quantization

**Scenario description**: The input is `torch.bfloat16` and the output is `torch.int8`. Except for the parameter names, this example is largely the same as the example of the same name for `torch_npu.npu_fused_infer_attention_score`.

v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()

- `quant_scale_out`: Required. For `torch.bfloat16` input, both `torch.float32` and `torch.bfloat16` are supported.
- `quant_offset_out`: Optional. The type and shape are the same as those of scale. If not passed, the default is 0.

softmax_scale = 1/math.sqrt(128.0)

```python
import torch
import torch_npu
import math

B, Q_N, Q_S, D = 1, 8, 164, 128
KV_N = 8
KV_S = 1024

# query/key/value: bfloat16, BNSD
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()
k = torch.randn(B, KV_N, KV_S, D, dtype=torch.bfloat16).npu()
v = torch.randn(B, KV_N, KV_S, D, dtype=torch.bfloat16).npu()

# quant_scale_out: required; for bfloat16 input, both float32 and bfloat16 are supported
# perchannel: for BNSD output, (1, Q_N, 1, D) is recommended
quant_scale_out = torch.randn(1, Q_N, 1, D, dtype=torch.float32).npu()
# quant_offset_out: optional; the type and shape are the same as those of scale; if not passed, the default is 0

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v,
    quant_scale_out=quant_scale_out,        # required for int8 output
    num_query_heads=Q_N,
    input_layout="BNSD",
    softmax_scale=softmax_scale,
    pre_tokens=65535,
    next_tokens=65535
)

print(out.dtype)  # torch.int8
print(out.shape)  # torch.Size([1, 8, 164, 128])
```

### Example 6: fake-quantization (KV split mode) + Decode

**Scenario description**: `dequant_scale_key` and `dequant_scale_value` are passed separately for KV. Compared with the example of the same name for `torch_npu.npu_fused_infer_attention_score`, the usage is the same and only the parameter names differ.

**Key parameters**: `key_quant_mode,value_quant_mode`: Required. The values must be consistent. When the value is 0, the perchannel mode is used.

**Related constraints**: [Q_S=1 (incremental inference) constraints](#en-us_topic_0000001832267082_section_qs_eq1_constraint)

```python
import torch
import torch_npu
import math


B, Q_N, Q_S, D = 1, 8, 1, 128       # Q_S=1 incremental inference
KV_N = 2                            
KV_S = 2048

# query: bfloat16, BNSD
q = torch.randn(B, Q_N, Q_S, D, dtype=torch.bfloat16).npu()
k = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()
v = torch.randint(-128, 127, (B, KV_N, KV_S, D), dtype=torch.int8).npu()

# dequant_scale: required, perchannel, dtype consistent with query
# shape = (1, KV_N, 1, D)
dequant_scale_key   = torch.randn(1, KV_N, 1, D, dtype=torch.bfloat16).npu()
dequant_scale_value = torch.randn(1, KV_N, 1, D, dtype=torch.bfloat16).npu()
# dequant_offset: optional, shape consistent with scale

softmax_scale = 1.0 / math.sqrt(D)

out, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q, k, v,
    dequant_scale_key=dequant_scale_key,            # KV split: key dequantization factor
    dequant_scale_value=dequant_scale_value,        # KV split: value dequantization factor
    key_quant_mode=0,                               # perchannel
    value_quant_mode=0,                             # perchannel, must be consistent with key
    num_query_heads=Q_N,
    num_key_value_heads=KV_N,
    input_layout="BNSD",
    softmax_scale=softmax_scale
)

print(out.dtype)   # torch.bfloat16
print(out.shape)   # torch.Size([1, 8, 1, 128])
```

### Example 7: general constraints + aclgraph mode

**Scenario description**: When `backend="npugraph_ex"` is explicitly passed, the [aclgraph mode](https://gitcode.com/Ascend/torchair/blob/26.2.0/docs/en/npugraph_ex/quick_start.md) is used.

**Key parameters**: `backend="npugraph_ex"`: Required by `torch.compile`, uses the aclgraph mode.

**Related constraints**: [General constraints](#general-constraints)

```python
import torch
import torch_npu
import math
import torchair as tng

import torch._dynamo
TORCHDYNAMO_VERBOSE=1
TORCH_LOGS="+dynamo"

# Configure logging and debug settings for graph capture
import logging
from torchair.core.utils import logger
logger.setLevel(logging.DEBUG)
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
        model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
        graph_output = model()
    single_op = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, num_query_heads = 8, input_layout = "BNSD", softmax_scale=softmax_scale, pre_tokens=65535, next_tokens=65535)
    print("single op output with mask:", single_op[0], single_op[0].shape)
    print("graph output with mask:", graph_output[0], graph_output[0].shape)

if __name__ == "__main__":
    MetaInfershape()
```

The output of the preceding code is similar to the following:

```text
single op output with mask: tensor([[[[-0.0417,  0.0780,  0.0827,  ...,  0.0655, -0.0575,  0.0035],
          [-0.0108, -0.0408, -0.0359,  ...,  0.0466,  0.0022,  0.0015],
          [ 0.0052, -0.0380,  0.1528,  ...,  0.0970,  0.0013,  0.0779],
          ...,
          [ 0.0104, -0.0696, -0.0266,  ..., -0.0652, -0.0572, -0.0585],
          [-0.0435, -0.0248, -0.0358,  ..., -0.0403,  0.0307, -0.0343],
          [ 0.0108, -0.0703, -0.0366,  ..., -0.0621,  0.0567, -0.0403]]]],
       device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])
graph output with mask: tensor([[[[-0.0417,  0.0780,  0.0827,  ...,  0.0655, -0.0575,  0.0035],
          [-0.0108, -0.0408, -0.0359,  ...,  0.0466,  0.0022,  0.0015],
          [ 0.0052, -0.0380,  0.1528,  ...,  0.0970,  0.0013,  0.0779],
          ...,
          [ 0.0104, -0.0696, -0.0266,  ..., -0.0652, -0.0572, -0.0585],
          [-0.0435, -0.0248, -0.0358,  ..., -0.0403,  0.0307, -0.0343],
          [ 0.0108, -0.0703, -0.0366,  ..., -0.0621,  0.0567, -0.0403]]]],
       device='npu:0', dtype=torch.float16) torch.Size([1, 8, 164, 128])

```
