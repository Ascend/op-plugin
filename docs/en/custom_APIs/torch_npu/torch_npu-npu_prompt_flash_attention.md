# torch_npu.npu_prompt_flash_attention

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id19 -->
- <term>Atlas inference products</term>: Supported
<!-- end id19 -->

## Function

- Function: full FA implementation.

- Formulas:

$$
atten\_out = softmax\left(scale \cdot (Q \cdot K) + atten\_mask\right) \cdot V
$$

## Prototype

```python
torch_npu.npu_prompt_flash_attention(query, key, value, *, pse_shift=None, padding_mask=None, atten_mask=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147483647, next_tokens=0, input_layout="BSH",num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0) -> Tensor
```

## Parameters

- **`query`** (`Tensor`): Required. Input $Q$ in the formulas. The data type must satisfy type deduction rules to remain identical to the data types of `key` and `value`. Non-contiguous tensors are not supported. The data layout can be ND.
    <!-- npu="310p" id20 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`.
    <!-- end id20 -->

    <!-- npu="910b" id3 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id3 -->
    <!-- npu="A3" id4 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id4 -->

- **`key`** (`Tensor`): Required. Input $K$ in the formulas. The data type must satisfy type deduction rules to remain identical to the data types of `query` and `value`. Non-contiguous tensors are not supported. The data layout can be ND.
    <!-- npu="310p" id21 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`.
    <!-- end id21 -->

    <!-- npu="910b" id5 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id5 -->
    <!-- npu="A3" id6 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id6 -->

- **`value`** (`Tensor`): Required. Input $V$ in the formulas. The data type must satisfy type deduction rules to remain identical to the data types of `query` and `key`. Non-contiguous tensors are not supported. The data layout can be ND.
    <!-- npu="310p" id22 -->
    - <term>Atlas inference products</term>: The data type can be `torch.float16`.
    <!-- end id22 -->

    <!-- npu="910b" id7 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id7 -->
    <!-- npu="A3" id8 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id8 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`pse_shift`** (`Tensor`): Optional. Non-contiguous tensors are not supported. The data layout can be ND. The input shape must be `(B, N, Q_S, KV_S)` or `(1, N, Q_S, KV_S)`, where `Q_S` represents `S` in the shape of `query`, and `KV_S` represents `S` in the shapes of `key` and `value`. If `KV_S` of `pse_shift` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. There are no specific requirements for the padding values of the padded portion. This parameter can be set to `None` if this feature is not used. For details about the overall constraints, see [Constraints](#section12345537164214).
    <!-- npu="310p" id23 -->
    - <term>Atlas inference products</term>: this parameter is not supported currently.
    <!-- end id23 -->

    <!-- npu="910b" id9 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`. When `pse_shift` is `torch.float16`, `query` must be `torch.float16` or `torch.int8`; when `pse_shift` is `torch.bfloat16`, `query` must be `torch.bfloat16`. When `query`, `key`, and `value` are `torch.float16` and `pse_shift` exists, the high-precision mode is used by default.
    <!-- end id9 -->
    <!-- npu="A3" id10 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float16` or `torch.bfloat16`. When `pse_shift` is `torch.float16`, `query` must be `torch.float16` or `torch.int8`; when `pse_shift` is `torch.bfloat16`, `query` must be `torch.bfloat16`. When `query`, `key`, and `value` are `torch.float16` and `pse_shift` exists, the high-precision mode is used by default.
    <!-- end id10 -->

- **`padding_mask`**: Reserved parameter, currently not used. Retain the default value.
- **`atten_mask`** (`Tensor`): Optional. Inverse triangular mask matrix where the lower triangle is all zeros and the upper triangle is negative infinity, $atten\_mask$ in the formula. The data type can be `torch.bool`, `torch.int8`, or `torch.uint8`. The data layout can be ND. Non-contiguous tensors are not supported. This parameter can be set to `None` if this feature is not used. Recommended shapes are `(Q_S, KV_S)`, `(B, Q_S, KV_S)`, `(1, Q_S, KV_S)`, `(B, 1, Q_S, KV_S)`, or `(1, 1, Q_S, KV_S)`, where `Q_S` represents `S` in the shape of `query`, and `KV_S` represents `S` in the shapes of `key` and `value`. In scenarios where `KV_S` of `atten_mask` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. Redundant elements must be padded with `1`. For details about the overall constraints, see [Constraints](#section12345537164214).
- **`actual_seq_lengths`** (`List[int]`): Optional. Valid sequence lengths of `query` across different batches. The data type can be `torch.int64`. If `seqlen` is not specified, set this parameter to `None`, indicating that the value is identical to the sequence length `S` in the `query` shape. Restriction: Valid sequence lengths in each batch must be less than or equal to the corresponding sequence length of that batch in `query`. When the input length of `seqlen` is `1`, each batch uses the same `seqlen`. When the input length of `seqlen` is greater than or equal to the batch count, the first batch number of elements in `seqlen` are selected. Other lengths are not supported.
    <!-- npu="310p" id24 -->
    - <term>Atlas inference products</term>: this parameter is not supported currently.
    <!-- end id24 -->

    <!-- npu="910b" id11 -->
    - <term>Atlas A2 products</term>: the TND layout is supported. When the `input_layout` of `query` is TND, this input parameter must be passed, and the number of its elements is used as the Batch value. The value of each element in this input parameter indicates the sum of the seqlens of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and negative values must not occur.
    <!-- end id11 -->
    <!-- npu="A3" id12 -->
    - <term>Atlas A3 products</term>: the TND layout is supported. When the `input_layout` of `query` is TND, this input parameter must be passed, and the number of its elements is used as the Batch value. The value of each element in this input parameter indicates the sum of the seqlens of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and negative values must not occur.
    <!-- end id12 -->

- **`deq_scale1`** (`Tensor`): Optional. Dequantization factor after BMM1. `pertensor` mode is supported. The data type can be `torch.uint64` or `torch.float32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used. Atlas inference accelerator cards: This parameter is not supported currently.
    <!-- npu="310p" id25 -->
    - <term>Atlas inference products</term>: this parameter is not supported currently.
    <!-- end id25 -->
- **`quant_scale1`** (`Tensor`): Optional. Quantization factor before BMM2. `pertensor` mode is supported. The data type can be `torch.float32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used. Atlas inference accelerator cards: This parameter is not supported currently.
    <!-- npu="310p" id26 -->
    - <term>Atlas inference products</term>: this parameter is not supported currently.
    <!-- end id26 -->
- **`deq_scale2`** (`Tensor`): Optional. Dequantization factor after BMM2. `pertensor` mode is supported. The data type can be `torch.uint64` or `torch.float32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used. Atlas inference accelerator cards: This parameter is not supported currently.
    - <term>Atlas inference products</term>: this parameter is not supported currently.
- **`quant_scale2`** (`Tensor`): Optional. Output quantization factor. The data layout can be ND. `pertensor` and `perchannel` modes are supported. This parameter can be set to `None` if this feature is not used.
    <!-- npu="310p" id27 -->
    - <term>Atlas inference products</term>: this parameter is not supported currently.
    <!-- end id27 -->

    <!-- npu="910b" id13 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float32` or `torch.bfloat16`. When the input is `torch.bfloat16`, both `torch.float32` and `torch.bfloat16` are supported; otherwise, only `torch.float32` is supported. In the perchannel format, when the output layout is $BSH$, the product of all dimensions of `quant_scale2` must be equal to $H$; for other layouts, the product must be equal to $N*D$. When the output layout is $BSH$, the shape of `quant_scale2` is recommended to be $(1, 1, H)$ or $(H,)$; when the output is $BNSD$, $(1, N, 1, D)$ or $(N, D)$ is recommended; when the output is $BSND$, $(1, 1, N, D)$ or $(N, D)$ is recommended.
    <!-- end id13 -->
    <!-- npu="A3" id14 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float32` or `torch.bfloat16`. When the input is `torch.bfloat16`, both `torch.float32` and `torch.bfloat16` are supported; otherwise, only `torch.float32` is supported. In the perchannel format, when the output layout is $BSH$, the product of all dimensions of `quant_scale2` must be equal to $H$; for other layouts, the product must be equal to $N*D$. When the output layout is $BSH$, the shape of `quant_scale2` is recommended to be $(1, 1, H)$ or $(H,)$; when the output is $BNSD$, $(1, N, 1, D)$ or $(N, D)$ is recommended; when the output is $BSND$, $(1, 1, N, D)$ or $(N, D)$ is recommended.
    <!-- end id14 -->

- **`quant_offset2`** (`Tensor`): Optional. Output quantization offset. `pertensor` and `perchannel` modes are supported. The data layout can be ND. If `quant_offset2` is provided, its data type and shape must be identical to those of `quant_scale2`. This parameter can be set to `None` if this feature is not used.
    <!-- npu="310p" id28 -->
    - <term>Atlas inference products</term>: this parameter is not supported currently.
    <!-- end id28 -->

    <!-- npu="910b" id15 -->
    - <term>Atlas A2 products</term>: The data type can be `torch.float32` or `torch.bfloat16`.
    <!-- end id15 -->
    <!-- npu="A3" id16 -->
    - <term>Atlas A3 products</term>: The data type can be `torch.float32` or `torch.bfloat16`.
    <!-- end id16 -->

- **`num_heads`** (`int`): Optional. Head count of `query`. Pass this value as a Python `int`.
- **`scale_value`** (`float`): Optional. Scaling factor serving as the scalar value of Muls in the computation flow, $scale$ in the formula. The value is typically the reciprocal of the square root of $d$. The data type can be `float`. Its data type and the data type of `query` must meet the type deduction rules. The default value is `1.0`.
- **`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2147483647`. Only the default value `2147483647` is supported for Atlas inference accelerator cards.
    <!-- npu="310p" id29 -->
    - <term>Atlas inference products</term>: only the default value 2147483647 is supported.
    <!-- end id29 -->
- **`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `0`. Only `0` and `2147483647` are supported for Atlas inference accelerator cards.
    <!-- npu="310p" id30 -->
    - <term>Atlas inference products</term>: only 0 and 2147483647 are supported.
    <!-- end id30 -->
- **`input_layout`** (`str`): Optional. Data layout configuration for the input `query`, `key`, and `value`. Supported values are `"BSH"`, `"BSND"`, `"BNSD"`, or `"BNSD_BSND"`. When the input layout is `"BNSD"`, the output layout is `"BSND"`. The default value is `"BSH"`.
- **`num_key_value_heads`** (`int`): Optional. Number of heads for `key` and `value`, used to support Grouped-Query Attention (GQA) scenarios. GQA is a compromise between Multi-Head Attention (MHA) and Multi-Query Attention (MQA): multiple query heads share the same set of key/value heads, thereby significantly reducing KV Cache memory usage and memory access overhead while maintaining model performance close to that of MHA. Pass this value as a Python `int`. The default value is `0`, indicating that the head counts of `key`, `value`, and `query` are identical. Restrictions: `num_heads` must be divisible by `num_key_value_heads`. The ratio of `num_heads` to `num_key_value_heads` must be less than or equal to `64`. In `"BSND"`, `"BNSD"`, or `"BNSD_BSND"` layouts, this value must be identical to the shape value of the n-axis in `key`/`value`. If this condition is not satisfied, the system raises an error. Only the default value `0` is supported for Atlas inference accelerator cards.
    <!-- npu="310p" id32 -->
    - <term>Atlas inference products</term>: only the default value `0` is supported.
    <!-- end id32 -->
- **`actual_seq_lengths_kv`** (`List[int]`): Optional. Valid sequence lengths of `key`/`value` across different batches. The data type can be `torch.int64`. Restrictions: Valid sequence lengths in each batch within this parameter must be less than or equal to the sequence length of the corresponding batch in `key`/`value`. When the input length of `seqlenKV` is `1`, each batch uses the same `seqlenKV`. When the input length is greater than or equal to the batch count, the first batch number of elements are selected. Other lengths are not supported.
    <!-- npu="310p" id31 -->
    - <term>Atlas inference products</term>: this parameter is not supported currently.
    <!-- end id31 -->

    <!-- npu="910b" id17 -->
    - <term>Atlas A2 products</term>: the TND layout is supported. When the input_layout of key/value is TND, this input parameter must be passed, and the number of its elements is used as the Batch value. The value of each element in this input parameter indicates the sum of the seqlens of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and no negative values are allowed.
    <!-- end id17 -->
    <!-- npu="A3" id18 -->
    - <term>Atlas A3 products</term>: the TND layout is supported. When the input_layout of key/value is TND, this input parameter must be passed, and the number of its elements is used as the Batch value. The value of each element in this input parameter indicates the sum of the seqlens of the current batch and all preceding batches. Therefore, the value of the latter element must be greater than or equal to that of the former element, and no negative values are allowed.
    <!-- end id18 -->

- **`sparse_mode`** (`int`): Optional. Sparsification mode. The data type can be `int64`. The default value is `0`. For details about overall constraints, see [Constraints](#section12345537164214). Only the default value `0` is supported for Atlas inference accelerator cards.

    <!-- npu="310p" id33 -->
    - <term>Atlas inference products</term>: only the default value `0` is supported.
    <!-- end id33 -->
    - When `sparse_mode` is 0, it indicates the `defaultMask` mode. If `atten_mask` is not passed, no mask operation is performed, and `pre_tokens` and `next_tokens` are ignored (set to `INT_MAX` internally); if it is passed, the complete `atten_mask` matrix $(S1 * S2)$ must be passed, indicating that the part between `pre_tokens` and `next_tokens` must be computed. The case where an entire row of the part of the input mask matrix participating in the computation is 1 is not supported.
    - When `sparse_mode` is 1, it indicates `allMask`. The case where an entire row of the part of the input mask matrix participating in the computation is 1 is not supported.
    - When `sparse_mode` is 2, it indicates the `leftUpCausal` mask mode, and the optimized `atten_mask` matrix (2048*2048) must be passed.
    - When `sparse_mode` is 3, it indicates the `rightDownCausal` mask mode, which corresponds to the lower-triangular scenario divided by the left vertex, and the optimized `atten_mask` matrix (2048*2048) must be passed.
    - When `sparse_mode` is 4, it indicates the `band` mask mode, and the optimized `atten_mask` matrix (2048*2048) must be passed.
    - When `sparse_mode` is 5, 6, 7, or 8, it indicates `prefix`, `global`, `dilated`, or `block_local` respectively. None of them are supported currently.

## Return Values

`Tensor`

$atten\_out$ in the formulas, representing the final computation result. When `input_layout` is $BNSD\_BSND$, the shape of the input `query` is $BNSD$ and the output shape is $BSND$. In other cases, the shape is the same as the shape of `query`.

## Constraints<a name="section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Handling of empty input parameters: the operator must determine whether the parameter `query` is empty internally and return directly if it is empty. If the parameter `query` is not an empty Tensor but the parameters `key` and `value` are empty Tensors (that is, $S2$ is 0), the output of the corresponding shape is filled with all zeros (filling `atten_out`). When `atten_out` is an empty Tensor, the framework handles it.
- Usage constraints for the `query`, `key`, and `value` inputs are as follows:

    <a name="zh-cn_topic_0000001798619409_table382212695610"></a>
    <table><thead align="left"><tr id="en-us_topic_0000001798619409_row1882282685619"><th class="cellrowborder" valign="top" width="30.570000000000004%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001798619409_p182262616569"><a name="en-us_topic_0000001798619409_p182262616569"></a><a name="en-us_topic_0000001798619409_p182262616569"></a>Product Model</p>
    </th>
    <th class="cellrowborder" valign="top" width="69.43%" id="mcps1.1.3.1.2"><p id="en-us_topic_0000001798619409_p1682262645612"><a name="en-us_topic_0000001798619409_p1682262645612"></a><a name="en-us_topic_0000001798619409_p1682262645612"></a>Axis Constraints</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="en-us_topic_0000001798619409_row88221926155613"><td class="cellrowborder" valign="top" width="30.570000000000004%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001798619409_p282310268569"><a name="en-us_topic_0000001798619409_p282310268569"></a><a name="en-us_topic_0000001798619409_p282310268569"></a><span id="en-us_topic_0000001798619409_ph999933035615"><a name="en-us_topic_0000001798619409_ph999933035615"></a><a name="en-us_topic_0000001798619409_ph999933035615"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term15651172142210_16"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term15651172142210_16"></a><term>Atlas inference products</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.43%" headers="mcps1.1.3.1.2 "><a name="en-us_topic_0000001798619409_ul849244711561"></a><a name="en-us_topic_0000001798619409_ul849244711561"></a><ul id="en-us_topic_0000001798619409_ul849244711561"><li>The B-axis value can be less than or equal to 128. </li><li>The N-axis value can be less than or equal to 256. </li><li>The S-axis value can be less than or equal to 65535. </li><li>The D-axis value can be less than or equal to 512.</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000001798619409_row1782312260569"><td class="cellrowborder" valign="top" width="30.570000000000004%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001798619409_p128231426175620"><a name="en-us_topic_0000001798619409_p128231426175620"></a><a name="en-us_topic_0000001798619409_p128231426175620"></a><span id="en-us_topic_0000001798619409_ph12807852115620"><a name="en-us_topic_0000001798619409_ph12807852115620"></a><a name="en-us_topic_0000001798619409_ph12807852115620"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term11962195213215_7"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term11962195213215_7"></a><term>Atlas A2 products</term></span></p>
    <p id="en-us_topic_0000001798619409_p16629103195810"><a name="en-us_topic_0000001798619409_p16629103195810"></a><a name="en-us_topic_0000001798619409_p16629103195810"></a><span id="en-us_topic_0000001798619409_ph1464135665612"><a name="en-us_topic_0000001798619409_ph1464135665612"></a><a name="en-us_topic_0000001798619409_ph1464135665612"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term1253731311225_7"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term1253731311225_7"></a><term>Atlas A3 products</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.43%" headers="mcps1.1.3.1.2 "><a name="en-us_topic_0000001798619409_ul103175710575"></a><a name="en-us_topic_0000001798619409_ul103175710575"></a><ul id="en-us_topic_0000001798619409_ul103175710575"><li>The B-axis value must be less than or equal to 65536 (64K). When the D-axis is not 32-byte aligned, only values less than or equal to 128 are supported. </li><li>The N-axis value can be less than or equal to 256. </li><li>The S-axis value can be less than or equal to 20971520 (20M). In long-sequence scenarios, an excessive computation workload can cause the PFA operator execution to time out, which raises an AI Core error (errorStr: "timeout or trap error"). In this configuration, S-axis splitting is recommended. The computation workload is affected by the B, S, N, and D axes. Larger values cause a higher computation workload. Typical long-sequence scenarios that can cause a timeout (scenarios where the product of B, S, N, and D is large) include:<a name="en-us_topic_0000001798619409_ul103175718575"></a><a name="en-us_topic_0000001798619409_ul103175718575"></a><ul id="en-us_topic_0000001798619409_ul103175718575"><li>B=1, Q<sub>N</sub>=20, Q<sub>S</sub>=1048576, D = 256, KV<sub>N</sub>=1, KV<sub>S</sub>=1048576. </li><li>B=1, Q<sub>N</sub>=2, Q<sub>S</sub>=10485760, D = 256, KV<sub>N</sub>=2, KV<sub>S</sub>=10485760. </li><li>B=20, Q<sub>N</sub>=1, Q<sub>S</sub>=1048576, D = 256, KV<sub>N</sub>=1, KV<sub>S</sub>=1048576. </li><li>B=1, Q<sub>N</sub>=10, Q<sub>S</sub>=1048576, D = 512, KV<sub>N</sub>=1, KV<sub>S</sub>=1048576.</li></ul>
    </li><li>The D axis must be less than or equal to 512. When <code>input_layout</code> is BSH or BSND, N*D must be less than 65535.</li></ul>
    </td>
    </tr>
    </tbody>
    </table>

- The parameter `sparse_mode` currently supports only the values `0, 1, 2, 3, 4`. Passing other values causes an error.
    - When `sparse_mode` is 0, if `atten_mask` is a null pointer, the input parameters `pre_tokens` and `next_tokens` are ignored (set to `INT_MAX` internally).
    - When `sparse_mode` is 2, 3, or 4, the shape of `atten_mask` must be $(S, S)$, $(1, S, S)$, or $(1, 1, S, S)$, where the value of $S$ must be fixed to 2048. The user must ensure that the passed `atten_mask` is lower-triangular; if `atten_mask` is not passed or the passed shape is incorrect, an error is reported.
    - When `sparse_mode` is 1, 2, or 3, the input parameters `pre_tokens` and `next_tokens` are ignored and assigned according to the related rules.

- Comprehensive restrictions on the number of `torch.int8` quantization-related input parameters and the input and output data formats:
    - Scenario where the input is `torch.int8` and the output is `torch.int8`: the input parameters `deq_scale1`, `quant_scale1`, `deq_scale2`, and `quant_scale2` must all exist, and `quant_offset2` is optional and defaults to `0` when not passed.
    - Scenario where the input is `torch.int8` and the output is `torch.float16`: the input parameters `deq_scale1`, `quant_scale1`, and `deq_scale2` must all exist; if the input parameter `quant_offset2` or `quant_scale2` exists (that is, is not `None`), an error is reported and returned.
    - Scenario where the input is `torch.float16` or `torch.bfloat16` and the output is `torch.int8`: the input parameter `quant_scale2` must exist, and `quant_offset2` is optional and defaults to `0` when not passed; if the input parameter `deq_scale1`, `quant_scale1`, or `deq_scale2` exists (that is, is not `None`), an error is reported and returned.
    - The input parameters `quant_offset2` and `quant_scale2` support the pertensor and perchannel formats and the `torch.float32` and `torch.bfloat16` data types. If `quant_offset2` is passed, its type and shape information must be the same as those of `quant_scale2`. When the input is `torch.bfloat16`, both `torch.float32` and `torch.bfloat16` are supported; otherwise, only `torch.float32` is supported. In the perchannel format, when the output layout is $BSH$, the product of all dimensions of `quant_scale2` must be equal to $H$; for other layouts, the product must be equal to $N*D$. When the output layout is $BSH$, the shape of `quant_scale2` is passed as $(1, 1, H)$ or $(H,)$; when the output is $BNSD$, $(1, N, 1, D)$ or $(N, D)$ is recommended; when the output is $BSND$, $(1, 1, N, D)$ or $(N, D)$ is recommended. In the pertensor format, aligning the $D$ axis to 32 bytes is recommended.
    - In the perchannel format, the input parameters `quant_scale2` and `quant_offset2` do not support left padding, Ring Attention (a distributed attention computation scheme for ultra-long sequences that splits the KV sequence across devices and completes the attention computation by passing KV blocks in a ring between devices), or scenarios where $D$ is not 32-byte aligned.
    - When the output is `torch.int8`, the scenario where sparse is `band` and `pre_tokens`/`next_tokens` are negative is not supported currently.

- Usage restrictions of the `pse_shift` function are as follows:
    - This function is supported when the data type of `query` is `torch.float16`, `torch.bfloat16`, or `torch.int8`.
    - When the data types of `query`, `key`, and `value` are `torch.float16` and `pse_shift` exists, the high-precision mode is forcibly used, and the corresponding restrictions are inherited from those of the high-precision mode.
    - $Q\_S$ must be greater than or equal to the $S$ length of `query`, and $KV\_S$ must be greater than or equal to the $S$ length of `key`.

- When the output is `torch.int8`, a non-null pointer and a non-empty `Tensor` value are passed for the input parameter `quant_offset2`, and `sparse_mode`, `pre_tokens`, and `next_tokens` meet the following conditions, some rows of the matrix may be all
    - `sparse_mode=0`: if `atten_mask` is a non-null pointer and for each batch `actual_seq_lengths-actual_seq_lengths_kv-pre_tokens>0` or `next_tokens<0`, the interception condition is met.
    - `sparse_mode=1` or `2`: the interception condition is never met.
    - `sparse_mode=3`: for each batch, `actual_seq_lengths_kv-actual_seq_lengths<0`, and the interception condition is met.
    - `sparse_mode=4`: when `pre_tokens<0` or for each batch `next_tokens+actual_seq_lengths_kv-actual_seq_lengths<0`, the interception condition is met.

- Separation of the kv pseudo-quantization parameters is not supported currently.
- The scenario of dimension misalignment is not supported currently.

## Example

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import math
    >>>
    >>> # Generate random data and send it to the npu
    >>> q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    >>> k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    >>> v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    >>> scale = 1/math.sqrt(128.0)
    >>> actseqlen = [164]
    >>> actseqlenkv = [1024]
    >>> atten_mask = torch.triu(torch.ones((164, 1024), dtype=torch.bool), diagonal=1).npu()
    >>>
    >>> # Call the PFA operator
    >>> out = torch_npu.npu_prompt_flash_attention(q, k, v,
    ... atten_mask = atten_mask, actual_seq_lengths = actseqlen, actual_seq_lengths_kv = actseqlenkv,
    ... num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)
    >>> print(out.shape)
    torch.Size([1, 8, 164, 128])
    >>> print(out.dtype)
    torch.float16
    ```

- Graph mode call

    ```python
    # Graph mode approach
    import torch
    import torch_npu
    import math

    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    import torch._dynamo
    TORCHDYNAMO_VERBOSE=1
    TORCH_LOGS="+dynamo"

    # Print macro that supports graph mode
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    from torch.library import Library, impl

    # Data generation
    q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    scale = 1/math.sqrt(128.0)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self):
            return torch_npu.npu_prompt_flash_attention(q, k, v, num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)

    def MetaInfershape():
        with torch.no_grad():
            model = Model()
            model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
            graph_output = model()

        single_op = torch_npu.npu_prompt_flash_attention(q, k, v, num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)
        print("single op output with mask:", single_op, single_op.shape)
        print("graph output with mask:", graph_output, graph_output.shape)

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
