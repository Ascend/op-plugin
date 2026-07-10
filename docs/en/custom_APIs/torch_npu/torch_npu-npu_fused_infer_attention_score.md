# torch_npu.npu_fused_infer_attention_score

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>   | √  |

## Function<a name="en-us_topic_0000001832267082_section14441124184110"></a>

- Description: Adapts to the `FlashAttention` operator in the incremental and full inference scenarios, supporting both full computation (`PromptFlashAttention`) and incremental computation (`IncreFlashAttention`). When `S` of the `query` matrix is `1`, the execution enters the `IncreFlashAttention` branch. Otherwise, it enters the `PromptFlashAttention` branch.
- Formula:

    $$
    attention\_out = softmax \left(scale * (query * key^\top) + atten\_mask \right) * value
    $$

## Prototype<a name="en-us_topic_0000001832267082_section45077510411"></a>

```python
torch_npu.npu_fused_infer_attention_score(query, key, value, *, pse_shift=None, atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, block_table=None, query_padding_size=None, kv_padding_size=None, key_antiquant_scale=None, key_antiquant_offset=None, value_antiquant_scale=None, value_antiquant_offset=None, key_shared_prefix=None, value_shared_prefix=None, actual_shared_prefix_len=None, query_rope=None, key_rope=None, key_rope_antiquant_scale=None, num_heads=1, scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout="BSH", num_key_value_heads=0, sparse_mode=0, inner_precise=0, block_size=0, antiquant_mode=0, softmax_lse_flag=False, key_antiquant_mode=0, value_antiquant_mode=0) -> (Tensor, Tensor)
```

## Parameters<a name="en-us_topic_0000001832267082_section112637109429"></a>

> [!NOTE]  
>
> - Dimension definitions for the `query`, `key`, and `value` parameters:<br>`B` (`Batch Size`) indicates the input sample batch size.<br>`S` (`Sequence Length`) indicates the input sample sequence length.<br>`H` (`Head Size`) indicates the hidden layer size.<br>`N` (`Head Num`) indicates the number of heads.<br> `D` (`Head Dim`) indicates the minimum unit size of the hidden layer, satisfying `D = H/N`.<br>`T` indicates the cumulative sum of the sequence lengths of all batch input samples.
> - `Q_S` and `S1` indicate the `S` dimension in the shape of `query`.<br>`KV_S` and `S2` indicate the `S` dimension in the shape of `key` and `value`.<br>`Q_N` indicates `num_query_heads`.<br>`KV_N` indicates `num_key_value_heads`.

- **`query`** (`Tensor`): Required. Query input of the attention structure. Non-contiguous tensors are not supported. The data type can be `float16` or `bfloat16`. The data layout can be ND.

- **`key`** (`Tensor`): Required. Key input of the attention structure. Non-contiguous tensors are not supported. The data type can be `float16`, `bfloat16`, `int8`, or `int4` (`int32`). The data layout can be ND.

- **`value`** (`Tensor`): Required. Value input of the attention structure. Non-contiguous tensors are not supported. The data type can be `float16`, `bfloat16`, `int8`, or `int4` (`int32`). The data layout can be ND.

- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`pse_shift`** (`Tensor`): Optional. Positional encoding parameter within the attention structure. The data type can be `float16` or `bfloat16`. The data type of this parameter and that of `query` must meet the data type deduction rules. Non-contiguous tensors are not supported. The data layout can be ND. This parameter can be set to `None` if this feature is not used.
    - When `Q_S` is greater than 1, if `pse_shift` is `float16`, `query` must be `float16` or `int8`. If `pse_shift` is `bfloat16`, `query` must be `bfloat16`. The input shape must be (B, Q_N, Q_S, KV_S) or (1, Q_N, Q_S, KV_S), where `Q_S` is the sequence length `S` in the `query` shape, and `KV_S` is the sequence length `S` in the `key` and `value` shapes. If `KV_S` of `pse_shift` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. There are no specific requirements for the padding values of the padded portion.
    - When `Q_S` is 1, if `pse_shift` is `float16`, `query` must be `float16`. If `pse_shift` is `bfloat16`, `query` must be `bfloat16`. The input shape must be (B, Q_N, 1, KV_S) or (1, Q_N, 1, KV_S), where `KV_S` is the sequence length `S` in the `key` and `value` shapes. If `KV_S` of `pse_shift` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. There are no specific requirements for the padding values of the padded portion.

- **`atten_mask`** (`Tensor`): Optional. Masks the results of `query` (Q) and `key` (K) to indicate whether to compute the correlation between tokens. The data type can be `bool`, `int8`, or `uint8`. Non-contiguous tensors are not supported. The data layout can be ND. This parameter can be set to `None` if this feature is not used.
    - When `sparse_mode` is `0` or `1`:
        - The input shape can be (1, Q_S, KV_S), (B, 1, Q_S, KV_S), or (1, 1, Q_S, KV_S).
        - When `input_layout` is `BSH`, `BSND`, `BNSD`, or `BNSD_BSND`, the `D` dimensions of `query`, `key`, and `value` are identical, and `query_rope` and `key_rope` are omitted, a shape of (B, KV_S) can be passed if `Q_S` is 1, and (Q_S, KV_S) can be passed if `Q_S` is greater than 1.
        - If `Q_S` or `KV_S` is not 16-byte or 32-byte aligned, it can be rounded up to the aligned value. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
    - When `sparse_mode` is `2`, `3`, or `4`, the input shape can be `(2048, 2048)`, `(1, 2048, 2048)`, or `(1, 1, 2048, 2048)`.

- **`actual_seq_lengths`** (`List[int]`): Optional. Valid sequence length (`seqlen`) of `query` in different batches. The data type can be `int64`. If `seqlen` is not specified, set this parameter to `None`, indicating that the value is identical to the sequence length `S` in the `query` shape.

    Restriction: The valid `seqlen` of each batch in this parameter must not exceed that of the corresponding batch in `query`. If the input length of `seqlen` is `1`, all batches use the same `seqlen`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlen` are used. Other lengths are not supported. When `input_layout` of `query` is `TND`, this parameter is required, and the number of elements in this parameter is used as the batch size. The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element, and cannot be negative.

- **`actual_seq_lengths_kv`** (`List[int]`): Optional. Valid sequence length of `key` and `value` across different batches. The data type can be `int64`. If this parameter is not specified or is set to `None`, the value is identical to the sequence length `S` in the `key` and `value` shapes. Constraints vary across different `Q_S` values. For details, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`dequant_scale1`** (`Tensor`): Optional. The data type can be `uint64` or `float32`. The data layout can be ND. This parameter indicates the dequantization factor after BMM1. `pertensor` mode is supported. This parameter can be set to `None` if this feature is not used.
- **`quant_scale1`** (`Tensor`): Optional. The data type can be `float32`. The data layout can be ND. This parameter indicates the quantization factor before BMM2. `pertensor` mode is supported. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`dequant_scale2`** (`Tensor`): Optional. The data type can be `uint64` or `float32`. The data layout can be ND. This parameter indicates the dequantization factor after BMM2. `pertensor` mode is supported. This parameter can be set to `None` if this feature is not used.
- **`quant_scale2`** (`Tensor`): Optional. The data type can be `float32` or `bfloat16`. The data layout can be ND. This parameter indicates the output quantization factor. `pertensor` and `perchannel` modes are supported. When the input data type is `bfloat16`, both `float32` and `bfloat16` are supported. Otherwise, only `float32` is supported. In `perchannel` mode, if the output layout is `BSH`, the product of all dimensions of `quant_scale2` must be identical to `H`. For other layouts, the product must equal `Q_N * D`. (You are advised to pass a shape of `(1, 1, H)` or `(H,)` when the output layout is `BSH`; `(1, Q_N, 1, D)` or `(Q_N, D)` when the output layout is `BNSD`; and `(1, 1, Q_N, D)` or `(Q_N, D)` when the output layout is `BSND`). This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`quant_offset2`** (`Tensor`): Optional. The data type can be `float32` or `bfloat16`. The data layout can be ND. This parameter indicates the output quantization offset. `pertensor` and `perchannel` modes are supported. If `quant_offset2` is provided, its data type and shape must be identical to those of `quant_scale2`. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`antiquant_scale`** (`Tensor`): Optional. The data type can be `float16` or `bfloat16`. The data layout can be ND. This parameter indicates the fake-quantization factor. `pertensor` and `perchannel` modes are supported. When `Q_S` is `1`, only `perchannel` is supported. When `Q_S` is greater than or equal to `2`, only `float16` is supported. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`antiquant_offset`** (`Tensor`): Optional. The data type can be `float16` or `bfloat16`. The data layout can be ND. This parameter indicates the fake-quantization offset. `pertensor` and `perchannel` modes are supported. When `Q_S` is `1`, only `perchannel` is supported. When `Q_S` is greater than or equal to `2`, only `float16` is supported. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`block_table`** (`Tensor`): Optional. The data type can be `int32`. The data layout can be ND. This parameter indicates the block mapping table used for KV storage in page attention. This parameter can be set to `None` if this feature is not used.
- **`query_padding_size`** (`Tensor`): Optional. The data type can be `int64`. The data layout can be ND. This parameter indicates whether the data of each batch in `query` is right-aligned and the number of right-aligned data records. It is valid only when `Q_S` is greater than 1. In other scenarios, it is invalid. The default value is `None`.
- **`kv_padding_size`** (`Tensor`): Optional. The data type can be `int64`. The data layout can be ND. This parameter indicates whether the data of each batch in `key` and `value` is right-aligned and the number of right-aligned data records. The default value is `None`.
- **`key_antiquant_scale`** (`Tensor`): Optional. The data layout can be `float16`, `bfloat16`, and `float32`. The data layout can be ND. This parameter indicates the dequantization factor of `value` when the KV fake-quantization parameters are separated. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214). The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with scale managed by page attention, and `pertoken` + `perhead` with scale managed by page attention.

- **`key_antiquant_offset`** (`Tensor`): Optional. The data type can be `float16`, `bfloat16`, or `float32`. The data layout can be ND. This parameter indicates the dequantization offset of `key` when the KV fake-quantization parameters are separated. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with offset managed by page attention, and `pertoken` + `perhead` with offset managed by page attention. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`value_antiquant_scale`** (`Tensor`): Optional. The data type can be `float16`, `bfloat16`, or `float32`. The data layout can be ND. This parameter indicates the dequantization factor of `value` when the KV fake-quantization parameters are separated. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214). The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with scale managed by page attention, and `pertoken` + `perhead` with scale managed by page attention.

- **`value_antiquant_offset`** (`Tensor`): Optional. The data type can be `float16`, `bfloat16`, or `float32`. The data layout can be ND. This parameter indicates the dequantization offset of `value` when the KV fake-quantization parameters are separated. The following modes are supported: `perchannel`, `pertensor`, `pertoken`, `pertensor` + `perhead`, `pertoken` + `perhead`, `pertoken` with offset managed by page attention, and `pertoken` + `perhead` with offset managed by page attention. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`key_shared_prefix`** (`Tensor`): Optional. Key prefix parameter in the attention structure. The data type can be `float16`, `bfloat16`, or `int8`. Non-contiguous tensors are not supported. The data layout can be ND. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`value_shared_prefix`** (`Tensor`): Optional. Value prefix parameter in the attention structure. The data type can be `float16`, `bfloat16`, or `int8`. Non-contiguous tensors are not supported. The data layout can be ND. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
- **`actual_shared_prefix_len`** (`List[int]`): Optional. Valid sequence length of `key_shared_prefix` and `value_shared_prefix`. The data type can be `int64`. If `seqlen` is not specified, this parameter can be set to `None`, indicating that the value is identical to the sequence length `S` in the `key_shared_prefix` and `value_shared_prefix` shapes. Restriction: The valid sequence length in this parameter must be less than or equal to the sequence length in `key_shared_prefix` and `value_shared_prefix`.
- **`query_rope`** (`Tensor`): Optional. RoPE information of the query in the Multi-head Latent Attention (MLA) structure. The data type can be `float16` or `bfloat16`. Non-contiguous tensors are not supported. The data layout can be ND.
- **`key_rope`** (`Tensor`): Optional. RoPE information of the key in the MLA structure. The data type can be `float16` or `bfloat16`. Non-contiguous tensors are not supported. The data layout can be ND.
- **`key_rope_antiquant_scale`** (`Tensor`): Optional. Reserved parameter, currently not used. Retain the default value.
- **`num_heads`** (`int`): Optional. Number of heads for `query`. The data type can be `int64`. In `BNSD` scenarios, this value must be identical to the shape of the N axis of `query`. Otherwise, an execution exception occurs.
- **`scale`** (`float`): Optional. Scaling factor, typically the reciprocal of the square root of `D`, used as a scalar value for Muls in the computation flow. The data type can be `float`. Its data type and the data type of `query` must meet the type deduction rules. The default value is `1.0`.
- **`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2147483647`. This parameter is invalid when `Q_S` is `1`.
- **`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. The data type can be `int64`. The default value is `2147483647`. This parameter is invalid when `Q_S` is `1`.
- **`input_layout`** (`string`): Optional. Data layout of the input `query`, `key`, and `value`. The default value is `"BSH"`.

    > [!NOTE]   
    > When a layout format contains an underscore (_), the portion to the left of the underscore represents the layout of the input `query`, and the portion to the right represents the layout of the output. The operator performs layout conversion internally.

    Supported layouts: `BSH`, `BSND`, `BNSD`, `BNSD_BSND` (when the input is `BNSD`, the output format is `BSND`; supported only when `Q_S` is greater than 1), `BSH_NBSD`, `BSND_NBSD`, `BNSD_NBSD` (when the output format is `NBSD`, supported only when `Q_S` is greater than 1 and less than or equal to 16), `TND`, `TND_NTD`, and `NTD_TND`. For details about the comprehensive constraints on `TND`-related scenarios, see [Constraints](#en-us_topic_0000001832267082_section12345537164214). `BNSD_BSND` indicates that the input is `BNSD` and the output format is `BSND`. It is supported only when `Q_S` is greater than 1.

- **`num_key_value_heads`** (`int`): Optional. Number of heads for `key` and `value`, used to support Grouped-Query Attention (GQA) scenarios. The data type can be `int64`. The default value is `0`, indicating that the number of heads for `key`, `value`, and `query` are identical. The value of `num_heads` must be divisible by `num_key_value_heads`, and the ratio of `num_heads` to `num_key_value_heads` cannot exceed 64. In `BSND`, `BNSD`, and `BNSD_BSND` (supported only when `Q_S` is greater than 1) scenarios, this value must also be identical to the shape of the N axis of `key` and `value`. Otherwise, an execution exception occurs.
- **`sparse_mode`** (`int`): Optional. Sparse mode. The data type can be `int64`. This parameter is invalid when `Q_S` is `1` and no RoPE input is provided. For details about the comprehensive constraints when `input_layout` is `TND`, `TND_NTD`, or `NTD_TND`, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
    - When `sparse_mode` is set to `0`, `defaultMask` mode is enabled. If `atten_mask` is omitted, no masking is performed, and `pre_tokens` and `next_tokens` are ignored (internally set to `INT_MAX`). If `atten_mask` is provided, a complete `atten_mask` matrix with shape `(S1, S2)` must be provided, indicating that the region between `pre_tokens` and `next_tokens` is computed.
    - When `sparse_mode` is set to `1`, `allMask` mode is enabled. A complete `atten_mask` matrix (S1 * S2) must be provided.
    - When `sparse_mode` is set to `2`, the mask in `leftUpCausal` mode is enabled. An optimized `atten_mask` matrix with shape `(2048, 2048)` must be provided.
    - When `sparse_mode` is set to `3`, the mask in `rightDownCausal` mode is enabled, corresponding to the lower-triangular scenario partitioned by the right vertex. An optimized `atten_mask` matrix with shape `(2048, 2048)` must be provided.
    - When `sparse_mode` is set to `4`, the mask in `band` mode is enabled. An optimized `atten_mask` matrix with shape `(2048, 2048)` must be provided.
    - When `sparse_mode` is set to `5`, `6`, `7`, or `8`, `prefix`, `global`, `dilated`, or `block_local` mode is enabled, respectively. Currently, these modes are not supported. The default value is `0`. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).

- **`inner_precise`** (`int`): Optional. Mode configuration flag with 4 valid options: `0`, `1`, `2`, and `3`. Each option is represented by a 2-bit value. Bit 0 specifies high-precision or high-performance mode, and bit 1 specifies whether to perform invalid row correction. The data type can be `int64`. When `Q_S` is greater than 1, if `sparse_mode` is `0` or `1` and a user-defined mask is provided, enabling invalid row correction is recommended. When `Q_S` is `1`, you can only set `inner_precise` to `0` or `1`. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).

    - `0`: enables high-precision mode and disables invalid row correction.
    - `1`: enables high-performance mode and disables invalid row correction.
    - `2`: enables high-precision mode and enables invalid row correction.
    - `3`: enables high-performance mode and enables invalid row correction.

    > [!NOTE]   
    > For `bfloat16` and `int8`, no distinction is made between high-precision and high-performance modes. Invalid row correction applies to `float16`, `bfloat16`, and `int8`. Currently, `0` and `1` are reserved values. If any row in the mask involved in computation consists entirely of `1`s, precision may degrade. In this case, you can set this parameter to `2` or `3` to enable invalid row correction and improve precision. However, this configuration may degrade performance.

- **`block_size`** (`int`): Optional. Maximum number of tokens in each block stored in the KV cache in page attention mode. The default value is `0`. The data type can be `int64`.
- **`antiquant_mode`** (`int`): Optional. Fake-quantization mode. A value of `0` indicates `perchannel` (which includes `pertensor`), and a value of `1` indicates `pertoken`. The default value is `0`.
    
    This parameter is invalid when `Q_S` is greater than or equal to `2`. If `Q_S` is `1`, passing a value other than `0` or `1` causes an execution exception.
    
- **`softmax_lse_flag`** (`bool`): Optional. Specifies whether to output `softmax_lse`, supporting S-axis tiling (splitting along the sequence length dimension). `True` outputs `softmax_lse`, and `False` disables it. The default value is `False`.
- **`key_antiquant_mode`** (`int`): Optional. Fake-quantization mode for `key`. The default value is `0`. Except for scenarios where `key_antiquant_mode` is `0` and `value_antiquant_mode` is `1`, this value must be identical to `value_antiquant_mode`. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).

    When `Q_S` is greater than or equal to `2`, only values of `0` and `1` are supported. When `Q_S` is `1`, the supported values are `0`, `1`, `2`, `3`, `4`, and `5`.

    - `0`: enables `perchannel` mode (which includes `pertensor`).
    - `1`: enables `pertoken` mode.
    - `2`: enables `pertensor` + `perhead` mode.
    - `3`: enables `pertoken` + `perhead` mode.
    - `4`: enables `pertoken` mode with scale or offset managed by page attention.
    - `5`: enables `pertoken` + `perhead` mode with scale or offset managed by page attention.

- **`value_antiquant_mode`** (`int`): Optional. Fake-quantization mode for `value`. Mode numbering is identical to that of `key_antiquant_mode`. The default value is `0`. Except for scenarios where `key_antiquant_mode` is `0` and `value_antiquant_mode` is `1`, this value must be identical to `key_antiquant_mode`. For details about the comprehensive constraints, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
    
    When `Q_S` is greater than or equal to `2`, only values of `0` and `1` are supported. When `Q_S` is `1`, the supported values are `0`, `1`, `2`, `3`, `4`, and `5`.

## Return Values<a name="en-us_topic_0000001832267082_section22231435517"></a>

- **`attention_out`** (`Tensor`): Output tensor in the formula. The data type can be `float16`, `bfloat16`, or `int8`. The data layout can be ND. Restriction: The `D` dimension of this return value must be identical to that of `value`, and all other dimensions must match those of the input parameter `query`.
- **`softmax_lse`** (`Tensor`): log-sum-exp result computed over the Query-Key product by using the Ring Attention algorithm. It is obtained by first computing `query` * `key`, subtracting the row-wise maximum (`softmax_max`), applying `exp`, summing the results (`softmax_sum`), taking the natural logarithm of `softmax_sum`, and finally adding `softmax_max`. The data type is `float32`. When `softmax_lse_flag` is `True`, the output shape is `(B, Q_N, Q_S, 1)` under standard layouts, and `(T, Q_N, 1)` when `input_layout` is `TND` or `NTD_TND`. When `softmax_lse_flag` is `False`, the output is a tensor of shape `[1]` with value `0`.

## Constraints<a name="en-us_topic_0000001832267082_section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Empty input handling: The operator checks whether `query` is empty internally. If `query` is empty, an empty value is returned. If `query` is a non-empty tensor but `key` and `value` are empty tensors (`S2` is `0`), `attention_out` returns a tensor filled with all `0`s matching its corresponding shape. If `attention_out` evaluates to an empty tensor, an empty value is returned.
- The corresponding tensors of `key` and `value` must have identical shapes. In non-contiguous scenarios, each tensor in the tensorlist of `key` and `value` must have a batch size of 1. The number of tensors must be equal to `B` in `query`, and the `N` and `D` dimensions must be identical.
- Comprehensive constraints on the number of `int8` quantization-related parameters and the input or output data formats:
    - For `int8` output: The input parameters `dequant_scale1`, `quant_scale1`, `dequant_scale2`, and `quant_scale2` must be provided simultaneously. `quant_offset2` is optional and defaults to `0` if not specified.
        - Atlas A2 training products/Atlas A2 inference products: The input data type is `int8`.
        - Atlas A3 training products/Atlas A3 inference products: The input data type is `int8`.

    - For `float16` output: The input parameters `dequant_scale1`, `quant_scale1`, and `dequant_scale2` must be provided simultaneously. If `quant_offset2` or `quant_scale2` is provided (not `None`), an error is raised and execution returns.
        - Atlas A2 training products/Atlas A2 inference products: The input data type is `int8`.
        - Atlas A3 training products/Atlas A3 inference products: The input data type is `int8`.

    - For `int8` output with full `float16` or `bfloat16` input: `quant_scale2` must be provided, and `quant_offset2` is optional (defaulting to `0` if omitted). If `dequant_scale1`, `quant_scale1`, or `dequant_scale2` is provided (not `None`), an error is raised and execution returns.
    - `quant_scale2` and `quant_offset2` support `pertensor` or `perchannel` modes, and their data types can be `float32` or `bfloat16`.

- Constraints on the `antiquant_scale` and `antiquant_offset` parameters:
    - `perchannel`, `pertensor`, and `pertoken` modes are supported.

        - `perchannel` mode: The shapes of both parameters must be: `(2, KV_N, 1, D)` in `BNSD` scenarios, `(2, KV_N, D)` in `BSND` scenarios, and `(2, H)` in `BSH` scenarios, where `N` is `num_key_value_heads`. The data types must be identical to that of `query`. `antiquant_mode` must be set to `0`. This mode is supported only when the data types of `key` and `value` are `int8`.
        - `pertensor` mode: The shapes of both parameters must be `(2,)`, and their data types must be identical to that of `query`. `antiquant_mode` must be set to `0`. The data types of `key` and `value` must be `int8`.

        - `pertoken` mode: The shapes of both parameters must be `(2, B, KV_S)`, and their data types must be `float32`. `antiquant_mode` must be set to `1`. The data types of `key` and `value` must be `int8`.

        The operator determines its execution mode based on the parameter shape. If `dim` is 1, it executes in `pertensor` mode. Otherwise, it executes in `perchannel` mode.

    - Symmetric quantization and asymmetric quantization are supported.
        - In asymmetric quantization mode, `antiquant_scale` and `antiquant_offset` must both be provided.
        - In symmetric quantization mode, `antiquant_offset` can be `None`. If `antiquant_offset` is omitted, symmetric quantization is performed. Otherwise, asymmetric quantization is performed.

- When `query_rope` and `key_rope` are provided (MLA scenario), the following constraints apply:
    - The data type and data format of `query_rope` must be identical to those of `query`.
    - The data type and data format of `key_rope` must be identical to those of `key`.
    - `query_rope` and `key_rope` must either both be omitted or both be provided. Partial configuration is not supported.
    - When `query_rope` and `key_rope` are non-empty, the `D` dimension of `query` must be `512` or `128`.
        - When `D` of `query` is `512`:
            - `sparse_mode`: Valid values: `0`, `3`, or `4`.
            - When `query_rope` is configured, the valid values of the `N` dimension of `query` are `1`, `2`, `4`, `8`, `16`, `32`, `64`, and `128`. The `D` dimension of `query_rope` must be `64`, and all other dimensions must match those of `query`.
            - When `key_rope` is configured: the `N` dimension must be `1` and the `D` dimension must be `512` for `key`. The `D` dimension of `key_rope` must be `64`, and all other dimensions must match those of `key`.
            - Supported data layouts for `key`, `value`, and `key_rope`: ND or `NZ`. When the data layout is `NZ`, the layout of the input parameters `key` and `value` is `[blockNum, KV_N, D/16, blockSize, 16]` for the `float16` or `bfloat16` data type, and `[blockNum, KV_N, D/32, blockSize, 32]` for the `int8` data type.
            - Supported `input_layout` values: `BSH`, `BSND`, `BNSD`, `BNSD_NBSD`, `BSND_NBSD`, `BSH_NBSD`, `TND`, and `TND_NTD`.
            - Page attention is supported: `block_size` must be divisible by 16 and cannot exceed 1024.
            - `softmax_lse`, left padding, tensorlist, PSE, prefix, fake quantization, full quantization, and post quantization are not supported.

        - When `D` of `query` is `128`:
            - Supported `input_layout` values: `BSH`, `BSND`, `TND`, `BNSD`, `NTD`, `BSH_BNSD`, `BSND_BNSD`, `BNSD_BSND`, and `NTD_TND`. 
            - When `query_rope` is configured: The `D` dimension of `query_rope` must be 64, and all other dimensions must match those of `query`. 
            - When `key_rope` is configured: The `D` dimension of `key_rope` must be `64`, and all other dimensions must match those of `key`. 
            - Left padding, tensorlist, PSE, prefix, fake quantization, full quantization, and post quantization are not supported.
            - Other constraints are the same as those when the layout is `TND` or `NTD_TND`.

    - Constraints on the input parameters `query`, `key`, and `value` in `TND`, `TND_NTD`, and `NTD_TND` layouts:
        - `actual_seq_lengths` and `actual_seq_lengths_kv` must both be provided, and the number of elements in these parameters is used as the batch size (the number of elements must be less than or equal to 4096). The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element.
        - When `D` of `query` is `512`:
            - `sparse_mode`: Valid values: `0`, `3`, or `4`.
            - `TND` and `TND_NTD` layouts are supported.
            - Page attention can be enabled. In this case, the length of `actual_seq_lengths_kv` must be identical to the batch size of `key` or `value`, representing the actual sequence length of each batch, and each value must not exceed `KV_S`.
            - The `N` dimension of `query` must be `1`, `2`, `4`, `8`, `16`, `32`, `64`, or `128`, while the `N` dimension of `key` and `value` must be `1`.
            - `query_rope` and `key_rope` must both be provided, and their `D` dimension must be `64`.
            - Left padding, tensorlist, PSE, prefix, fake quantization, and full quantization are not supported.

        - When `D` of `query` is not `512`:
            - When `query_rope` and `key_rope` are omitted: in `TND` layouts, `Q_D`, `K_D`, and `V_D` must all be `128`, or `Q_D` and `K_D` must be `192` with `V_D` equal to `128` or `192`; in `NTD` layouts, `V_D` must not be `192`; and in `NTD_TND` layouts, `Q_D` and `K_D` must be `128` or `192`, and `V_D` must be `128`. When `query_rope` and `key_rope` are provided, `Q_D`, `K_D`, and `V_D` must all be `128`.
            - `TND`, `NTD`, and `NTD_TND` layouts are supported.
            - In page attention scenarios, `block_size` must be a multiple of 16 and less than or equal to 1024.
            - Left padding, tensorlist, PSE, prefix, fake quantization, full quantization, and post quantization are not supported.

- Constraints on parameters in GQA fake-quantization scenarios when `key` and `value` are in `NZ` layout:
    - `perchannel` and `pertoken` modes are supported. The data type of `query` is fixed to `bfloat16`, and the data types of `key` and `value` are fixed to `int8`. The `D` dimension of `query`, `key`, and `value` must be `128`. The sequence length (`S`) of `query` must be 1 to 16.
    - `input_layout` must be `BSH`, `BSND`, and `BNSD`.
    - Only `page_attention` scenarios are supported, where `blockSize` must be `128` or `512`.
    - `key` and `value` support only `NZ` layout inputs, with a shape format of `[blockNum, KV_N, D/32, blockSize, 32]`.
    - Data type requirements for `key_antiquant_scale` and `value_antiquant_scale`: in `perchannel` mode, only `bfloat16` is supported; in `pertoken` mode, only `float32` is supported.
    - Shape requirements for `key_antiquant_scale` and `value_antiquant_scale`: in `perchannel` mode, when `input_layout` is `BSH`, the shape must be `[H]`, when `input_layout` is `BNSD`, the shape must be `[KV_N, 1, D]`, and when the output layout is `BSND`, the shape must be `[KV_N, D]`; in `pertoken` mode, the shape must be `[B, KV_S]`, where `S` must be greater than or equal to the second dimension of `block_table` multiplied by `block_size`.
    - Only KV separation is supported.
    - Only high-performance mode is supported.
    - When MTP is `0`, `sparse_mode` can be `0`, and `atten_mask` must not be provided. When MTP is greater than 0 and less than 16, `sparse_mode` can be `3`, and an optimized `atten_mask` matrix must be provided with shape ``(2048, 2048)``.
    - Configuring `key_antiquant_offset` and `value_antiquant_offset` is not supported.
    - Configuring `query_rope` and `key_rope` is not supported.
    - Left padding, tensorlist, PSE, prefix, and post-quantization are not supported.
    - Supported combinations of (`num_query_heads`, `num_key_value_heads`) are `(10, 1)`, `(64, 8)`, `(80, 8)`, and `(128, 16)`.
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
            - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: When the data types of `query`, `key`, or `value` include `int8`, the `D` dimension must be a multiple of 32. When the data types of `query`, `key`, `value`, or `attention_out` include `int4`, the `D` dimension must be a multiple of 64. When the data types are all `float16` or `bfloat16`, the `D` dimension must be a multiple of 16.

    - Constraints on `actual_seq_lengths`:
    
        Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The valid sequence length of each batch in this parameter must be less than or equal to the sequence length of the corresponding batch in `query`. If the input length of `seqlen` is `1`, all batches use the same `seqlen`. If the input length is greater than or equal to the batch size, the first *N* elements (where *N* equals the batch size) of `seqlen` are used. Other lengths are not supported. For details about the comprehensive constraints when `input_layout` of `query` is `TND` or `NTD_TND`, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
        
    - Constraints on `actual_seq_lengths_kv`:
    
        Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The valid sequence length of each batch in this parameter must be less than or equal to the sequence length of the corresponding batch in `key` and `value` If the length of this parameter is 1, all batches use the same sequence length. If the length is greater than or equal to the batch size, only the first `batch_size` elements are used. Other lengths are not supported. For details about the comprehensive constraints when `input_layout` of `key` or `value` is `TND` or `NTD_TND`, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
        
    - The `sparse_mode` parameter must be set to `0`, `1`, `2`, `3`, or `4`. If it is set to other values, an error is raised.
        
        - When `sparse_mode` is `0`, if `atten_mask` is `None`, or if `atten_mask` is provided in a left-padding scenario, the input parameters `pre_tokens` and `next_tokens` are ignored and internally set to `INT_MAX`.
        - When `sparse_mode` is `2`, `3`, or `4`, the shape of `atten_mask` must be `(S, S)`, `(1, S, S)`, or `(1, 1, S, S)`, where `S` must be `2048`. The provided `atten_mask` must be a lower triangular matrix. If `atten_mask` is omitted or has an invalid shape, an error is raised.       
        - When `sparse_mode` is `1`, `2`, or `3`, the input parameters `pre_tokens` and `next_tokens` are ignored and assigned based on the relevant rules.
        
    - Combined-parameter scenarios for KV Cache dequantization support only dequantizing `int8` to `float16`. If the product of the data ranges of `key` and `value` and `antiquant_scale` falls within the range of (-1, 1), accuracy is guaranteed in high-performance mode. Otherwise, high-precision mode must be enabled to ensure accuracy.
    - Page attention scenarios:
        - Page attention can be enabled only when `block_table` exists and is valid, and `key` and `value` are arranged in contiguous memory based on the indices in `block_table`. The supported data types for `key` and `value` are `float16` and `bfloat16`. In this case, `input_layout` of `key` and `value` is ignored. The `block_table` contains block IDs. Their validity is not verified and must be ensured by the user.
        - `block_size` is a user-defined parameter that affects page attention performance. When page attention is enabled, `block_size` must range from 128 to 512 and must be a multiple of 128. Generally, page attention improves throughput but reduces performance.
    
        - In page attention scenarios, when the input KV cache layout is `(blocknum, blocksize, H)` and the product of `KV_N` and `D` is greater than 65535, execution is blocked and an error is raised due to hardware constraints. This can be resolved by enabling GQA (to reduce `KV_N`) or by changing the KV cache layout to `(blocknum, KV_N, blocksize, D)`. When the `input_layout` of `query` is `BNSD` or `TND`, the KV cache supports both `(blocknum, blocksize, H)` and `(blocknum, KV_N, blocksize, D)` layouts. When it is `BSH` or `BSND`, only `(blocknum, blocksize, H)` is supported. The value of `blocknum` must not be less than the total number of blocks required across all batches, calculated using `actual_seq_lengths_kv` and `block_size`. The shapes of `key` and `value` must be identical.
        - Page attention does not support fake quantization, tensorlist, or left padding.
        - In page attention scenarios, `actual_seq_lengths_kv` must be provided.
        - In page attention scenarios, `block_table` must be a 2D tensor. Its first dimension must equal `B`, and its second dimension must be no less than `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actual_seq_lengths_kv` across batches).
        - In page attention scenarios, two formats are supported, and the supported data types are `float32` and `bfloat16`. Cases where `query` is `int8` are not supported.
        - When page attention is enabled, the input sequence length `KV_S` must be greater than or equal to the product of `maxBlockNumPerSeq` and `block_size` in the following scenarios:
            - When `atten_mask` is provided and its shape is identical to `(B, 1, Q_S, KV_S)`.
            - When `pse_shift` is provided and its shape is identical to `(B, Q_N, Q_S, KV_S)`.
    
    - Left padding for `query`:
        - In left padding for `query`, the start index for data transfer is calculated as: `Q_S - query_padding_size - actual_seq_lengths`. The end index is calculated as: `Q_S - query_padding_size`. The start index must not be less than 0, and the end index must not exceed `Q_S`. Otherwise, the result may be incorrect.
        - In left padding for `query`, if `kv_padding_size` is less than 0, it is set to `0`.
        - Left padding for `query` must be enabled together with the `actual_seq_lengths` parameter. Otherwise, right-padding for `query` is used by default.   
        - Left padding for `query` does not support page attention and cannot be enabled together with the `block_table` parameter.
        
    - Left padding for `key` and `value`:
        - Left padding for `key` and `value`, the start index for KV cache transfer is calculated as: `KV_S - kv_padding_size - actual_seq_lengths_kv`. The end index is calculated as: `KV_S - kv_padding_size`. The start index must not be less than 0, and the end index must not exceed `KV_S`. Otherwise, the result may be incorrect.
        - In left padding for `key` and `value`, if `kv_padding_size` is less than 0, it is set to `0`.
        - Left padding for `key` and `value` must be enabled together with the `actual_seq_lengths_kv` parameter. Otherwise, right padding for `key` and `value` is used by default.   
        - Left padding for `key` and `value` does not support page attention and cannot be enabled together with the `block_table` parameter.
        
    - The input parameters `quant_scale2` and `quant_offset2` support `pertensor` or `perchannel` quantization, and their data types can be `float32` or `bfloat16`. If `quant_offset2` is provided, its data type and shape must be identical to those of `quant_scale2`. When the input data type is `bfloat16`, both `float32` and `bfloat16` are supported. Otherwise, only `float32` is supported. In `perchannel` scenarios, when the output layout is `BSH`, the product of all dimensions of `quant_scale2` must equal `H`. For other layouts, the product must equal `N * D`. The recommended shape for `quant_scale2` is `(1, 1, H)` or `(H,)` for the `BSH` output layout, `(1, Q_N, 1, D)` or `(Q_N, D)` for the `BNSD` output layout, and `(1, 1, Q_N, D)` or `(Q_N, D)` for the `BSND` output layout.
    - When the output data type is `int8` and `quant_scale2` and `quant_offset2` are in `perchannel` mode, left padding, Ring Attention, and cases where the `D` dimension is not 32-byte aligned are not supported.
    - When the output data type is `int8`, scenarios where `sparse_mode` is `band` and `pre_tokens` or `next_tokens` are negative are not supported.
    - Usage constraints for the `pse_shift` feature:
        
        - Supported when the data type of `query` is `float16`, `bfloat16`, or `int8`.
        - When `query`, `key`, and `value` are all `float16` and `pse_shift` is provided, high-precision mode is enforced. The constraints are the same as those of high-precision mode.
        - `Q_S` must be greater than or equal to the sequence length (`S`) of `query`, and `KV_S` must be greater than or equal to the sequence length (`S`) of `key`. In prefix scenarios, `KV_S` must be greater than or equal to the sum of `actual_shared_prefix_len` and the sequence length (`S`) of `key`.
        
    - When the output data type is `int8`, if `quant_offset2` is a non-`None` and non-empty tensor, and `sparse_mode`, `pre_tokens`, and `next_tokens` meet the execution blocking conditions, certain matrix rows will be excluded from computation. This causes computation errors, and execution will be blocked:
        - When `sparse_mode` is `0`, if `atten_mask` is not `None` and, for each batch, `actual_seq_lengths - actual_seq_lengths_kv - pre_tokens > 0` or `next_tokens < 0`, the execution will be blocked.
        - When `sparse_mode` is `1` or `2`, the execution will not be blocked.
        - When `sparse_mode` is `3` and, for each batch, `actual_seq_lengths_kv - actual_seq_lengths < 0`, the execution will be blocked.
        - When `sparse_mode` is `4` and `pre_tokens < 0` or `batch next_tokens + actual_seq_lengths_kv - actual_seq_lengths < 0`, the execution will be blocked.
        
    - Constraints on prefix parameters:
        - `key_shared_prefix` and `value_shared_prefix` must either both be omitted or both be provided.
        - When `key_shared_prefix` and `value_shared_prefix` are both provided, the dimensions and data types of `key_shared_prefix`, `value_shared_prefix`, `key`, and `value` must be identical.
        - When `key_shared_prefix` and `value_shared_prefix` are both provided, the batch size (first dimension) of `key_shared_prefix` must be `1`. For `BNSD` and `BSND` layouts, the `N` and `D` dimensions must be identical to those of `key`. For the `BSH` layout, the `H` dimension must be identical to that of `key`. The same applies to `value_shared_prefix`. The `S` dimensions of `key_shared_prefix` and `value_shared_prefix` must be identical.
        - When `actual_shared_prefix_len` is provided, its shape must be `[1]`, and its value must not exceed the `S` dimension of `key_shared_prefix` and `value_shared_prefix`.
        - The sum of the prefix `S` and the `S` of `key` or `value` must satisfy the original `S` constraints for `key` or `value`.
        - Prefix is not supported in page attention, left padding, or tensorlist scenarios.
        - In prefix scenarios, the data types of `query`, `key`, and `value` cannot all be `int8`.
        - In prefix scenarios, when `sparse_mode` is `0` or `1` and `atten_mask` is provided, `S2` must be greater than or equal to the sum of `actual_shared_prefix_len` and the `S` of `key`.
        - In prefix scenarios, the input data types of `query`, `key`, and `value` cannot all be `int8`.
        
    - Constraints on KV fake-quantization parameter separation:
        - When both fake-quantization parameters and KV-separated quantization parameters are provided, the KV-separated quantization parameters take precedence.    
        - `key_antiquant_mode` and `value_antiquant_mode` must have identical values.
        - `key_antiquant_scale` and `value_antiquant_scale` must either both be omitted or both be provided. Likewise, `key_antiquant_offset` and `value_antiquant_offset` must either both be omitted or both be provided.
        - When `key_antiquant_scale` and `value_antiquant_scale` are both provided, their shapes must be identical. When `key_antiquant_offset` and `value_antiquant_offset` are both provided, their shapes must be identical. 
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products:
            - Only `pertoken` and `perchannel` modes are supported. In `pertoken` mode, the shapes of both parameters must be `(B, KV_S)`, and their data types must be `float32`. In `perchannel` mode, the shapes of both parameters must be `(KV_N, D)` or `(H)`, and their data types must be `bfloat16`.
            - When `key_antiquant_scale` and `value_antiquant_scale` are both provided, the `query` sequence length `S` must be less than or equal to `16`. The data type of `query` must be `bfloat16`, the data types of `key` and `value` must be `int8`, and the output data type must be `bfloat16`. Tensorlist, left padding, page attention, and prefix are not supported.
    
        - The following table describes quantization modes for scale and offset management.
    
            > [!NOTE]   
            > The scale and offset parameters are `key_antiquant_scale`, `value_antiquant_scale`, `key_antiquant_offset`, and `value_antiquant_offset`.

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
    
- **When `Q_S` is equal to 1**:
    - Usage constraints for `query`, `key`, and `value` inputs:
        - The `B` dimension must be less than or equal to 65536, the `N` dimension must be less than or equal to 256, the `S` dimension must be less than or equal to 262144, and the `D` dimension must be less than or equal to 512.
        - Scenarios where the input data types of `query`, `key`, and `value` are all `int8` are not supported.
        - In `int4` (`int32`) fake-quantization scenarios, PyTorch graph-mode execution only supports inputs where KV `int4` values are packed into `int32` tensors. (Using `dynamic_quant` to generate data in `int4` format is recommended, as each `int32` value encapsulates eight `int4` values).
        - In `int4` (`int32`) fake-quantization scenarios, when KV `int4` values are packed into `int32` inputs, the `N`, `D`, or `H` dimensions of `key` and `value` must be one-eighth of their actual values (the same rule applies to prefix parameters). Additionally, `int4` fake quantization requires the `D` dimension to be a multiple of 64, whereas the underlying `int32` tensor requires the `D` dimension to be a multiple of 8.

    - Constraints on `actual_seq_lengths`:
    
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: When `input_layout` of `query` is not `TND` and `Q_S` is 1, this parameter is ignored. For details about the comprehensive constraints when `input_layout` or `query` is `TND` or `TND_NTD`, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
        
    - Constraints on `actual_seq_lengths_kv`:
    
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The valid sequence length of each batch in this parameter must be less than or equal to the sequence length of the corresponding batch in `key` and `value` If the length of this parameter is 1, all batches use the same sequence length. If the length is greater than or equal to the batch size, only the first `batch_size` elements are used. Other lengths are not supported. For details about the comprehensive constraints when `input_layout` of `key` or `value` is `TND` or `TND_NTD`, see [Constraints](#en-us_topic_0000001832267082_section12345537164214).
        
    - Page attention scenarios:
        - Page attention can be enabled only when `block_table` exists and is valid, and `key` and `value` are arranged in a contiguous memory space based on the indices in `block_table`. In this case, the `input_layout` parameter of `key` and `value` is ignored.
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products:
            - The data types of `key` and `value` can be `float16`, `bfloat16`, or `int8`.
            - Scenarios where `query` is `bfloat16` or `float16` while `key` and `value` are `int4` (`int32`) are not supported.
    
        - In these scenarios, `block_size` is a user-defined parameter that affects page attention performance. `block_size` must be a non-zero value and must not exceed 512. When the input data types of `key` and `value` are `float16` or `bfloat16`, the `block_size` must be a multiple of 16; when the input data type is `int8`, it must be a multiple of 32. A value of 128 is recommended. Generally, page attention improves throughput but reduces performance.
        - The product of all dimensions of the tensors corresponding to `key` and `value` must not exceed the representable range of `int32`.
        - In page attention scenarios, `block_table` must be a 2D tensor. Its first dimension must equal `B`, and its second dimension must be no less than `maxBlockNumPerSeq` (the maximum number of blocks corresponding to `actual_seq_lengths_kv` across batches).
        - In page attention scenarios, when `input_layout` of `query` is `BNSD` or `TND`, the KV cache supports both `(blocknum, blocksize, H)` and `(blocknum, KV_N, blocksize, D)` layouts. When `input_layout` of `query` is `BSH` or `BSND`, only the `(blocknum, blocksize, H)` format is supported. The value of `blocknum` must not be less than the total number of blocks required across all batches, calculated using `actual_seq_lengths_kv` and `block_size`. The shapes of `key` and `value` must be identical.
        - In page attention scenarios, the `(blocknum, KV_N, blocksize, D)` layout format of the KV cache typically provides better performance than `(blocknum, blocksize, H)`, and is therefore recommended.
        - In page attention scenarios, when the KV cache layout format is `(blocknum, blocksize, H)` and the product of `numKvHeads` and `headDim` exceeds 64K, execution is blocked and an error is raised due to hardware instruction constraints. This can be resolved by enabling GQA (to reduce `numKvHeads`) or by adjusting the KV cache layout format to `(blocknum, numKvHeads, blocksize, D)`.
        - Page attention does not support tensorlist or left padding.
        - In page attention scenarios, the product of all dimensions of the tensors corresponding to `key` and `value` must not exceed the representable range of `int32`.
        - In page attention scenarios, when `atten_mask` is provided, if `sparse_mode` is not `2`, `3`, or `4`, the last dimension of `atten_mask` must be greater than or equal to the product of the second dimension of `block_table` and `block_size`.
        - In page attention scenarios, when `pse_shift` is provided, the last dimension of `pse_shift` must be greater than or equal to the product of the second dimension of `block_table` and `block_size`.
        - In page attention scenarios, the input sequence length `S` must be greater than or equal to the product of the second dimension of `block_table` and `block_size` in the following scenarios:
            - In `pertoken` fake quantization mode: the shapes of `antiquant_scale` and `antiquant_offset` must both be `(2, B, S)`.
            - In `pertoken` + `perhead` mode: The shapes of these parameters must be `(B, N, S)`, and their data types must be `float32`. The data types of `key` and `value` can be `int8` and `int4` (`int32`).
    
    - Left padding for `key` and `value`:
        - The start index for KV cache transfer is calculated as: `Smax - kv_padding_size - actual_seq_lengths`. The end index for KV cache transfer is calculated as: `Smax` - `kv_padding_size`. If the start index or end index for KV cache transfer is less than 0, the output tensor is filled with all `0`s.
        - If `kv_padding_size` is less than 0, it is set to `0`.
        - Left padding for `key` and `value` must be enabled together with the `actual_seq_lengths` parameter. Otherwise, right padding for `key` and `value` is used by default.
        - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: In left padding for `key` and `value`, configurations where `query` is `bfloat16` or `float16` while `key` and `value` are `int4` (`int32`) are not supported.
    
    - Constraints on KV fake-quantization parameter separation:
        - Except for scenarios where `key_antiquant_mode` is `0` and `value_antiquant_mode` is `1`, the values of `key_antiquant_mode` and `value_antiquant_mode` must be identical.  
        - `key_antiquant_scale` and `value_antiquant_scale` must either both be omitted or both be provided. Likewise, `key_antiquant_offset` and `value_antiquant_offset` must either both be omitted or both be provided.
        - Except for scenarios where `key_antiquant_mode` is `0` and `value_antiquant_mode` is `1`, when `key_antiquant_scale` and `value_antiquant_scale` are both provided, their shapes must be identical; when `key_antiquant_offset` and `value_antiquant_offset` are both provided, their shapes must also be identical.
        - Post-quantization is not supported in `int4` (`int32`) fake-quantization scenarios.
        - The following table describes quantization modes for scale and offset management.
    
            > [!NOTE]   
            > The scale and offset parameters are `key_antiquant_scale`, `value_antiquant_scale`, `key_antiquant_offset`, and `value_antiquant_offset`.
    
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
    scale = 1/math.sqrt(128.0)
    actseqlen = [164]
    actseqlenkv = [1024]
    
    # Call the FIA operator.
    out, _ = torch_npu.npu_fused_infer_attention_score(q, k, v, 
    actual_seq_lengths = actseqlen, actual_seq_lengths_kv = actseqlenkv,
    num_heads = 8, input_layout = "BNSD", scale = scale, pre_tokens=65535, next_tokens=65535)
    
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
    
    # Generate data
    q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    scale = 1/math.sqrt(128.0)
    
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self):
            return torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads = 8, input_layout = "BNSD", scale=scale, pre_tokens=65535, next_tokens=65535)
    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()
        single_op = torch_npu.npu_fused_infer_attention_score(q, k, v, num_heads = 8, input_layout = "BNSD", scale=scale, pre_tokens=65535, next_tokens=65535)
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
