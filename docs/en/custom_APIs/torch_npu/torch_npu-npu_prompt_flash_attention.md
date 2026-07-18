# torch_npu.npu_prompt_flash_attention

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |
|<term>Atlas inference accelerator cards</term> | √   |

## Function

- Description: Implements full FlashAttention (FA).

- Formula:

$$
atten\_out = softmax\left(scale \cdot (Q \cdot K) + atten\_mask\right) \cdot V
$$

## Prototype

```python
torch_npu.npu_prompt_flash_attention(query, key, value, *, pse_shift=None, padding_mask=None, atten_mask=None, actual_seq_lengths=None, deq_scale1=None, quant_scale1=None, deq_scale2=None, quant_scale2=None, quant_offset2=None, num_heads=1, scale_value=1.0, pre_tokens=2147483647, next_tokens=0, input_layout="BSH",num_key_value_heads=0, actual_seq_lengths_kv=None, sparse_mode=0) -> Tensor
```

## Parameters

- **`query`** (`Tensor`): Required. Input $Q$ in the formulas. The data type must satisfy type deduction rules to remain identical to the data types of `key` and `value`. Non-contiguous tensors are not supported. The data layout can be ND.
    - Atlas inference accelerator cards: The data type can be `float16`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16`, `bfloat16`, or `int8`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float16`, `bfloat16`, or `int8`.

- **`key`** (`Tensor`): Required. Input $K$ in the formulas. The data type must satisfy type deduction rules to remain identical to the data types of `query` and `value`. Non-contiguous tensors are not supported. The data layout can be ND.
    - Atlas inference accelerator cards: The data type can be `float16`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16`, `bfloat16`, or `int8`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float16`, `bfloat16`, or `int8`.

- **`value`** (`Tensor`): Required. Input $V$ in the formulas. The data type must satisfy type deduction rules to remain identical to the data types of `query` and `key`. Non-contiguous tensors are not supported. The data layout can be ND.
    - Atlas inference accelerator cards: The data type can be `float16`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16`, `bfloat16`, or `int8`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float16`, `bfloat16`, or `int8`.

- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).

- **`pse_shift`** (`Tensor`): Optional. Non-contiguous tensors are not supported. The data layout can be ND. The input shape must be `(B, N, Q_S, KV_S)` or `(1, N, Q_S, KV_S)`, where `Q_S` represents `S` in the shape of `query`, and `KV_S` represents `S` in the shapes of `key` and `value`. If `KV_S` of `pse_shift` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. There are no specific requirements for the padding values of the padded portion. This parameter can be set to `None` if this feature is not used. For details about the comprehensive constraints, see [Constraints](#section12345537164214).
    - Atlas inference accelerator cards: This parameter is not supported currently.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16` or `bfloat16`. When `pse_shift` is `float16`, `query` must be `float16` or `int8`. When `pse_shift` is `bfloat16`, `query` must be `bfloat16`. High-precision mode is enabled by default when `query`, `key`, and `value` are `float16` and `pse_shift` is provided.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float16` or `bfloat16`. When `pse_shift` is `float16`, `query` must be `float16` or `int8`. When `pse_shift` is `bfloat16`, `query` must be `bfloat16`. High-precision mode is enabled by default when `query`, `key`, and `value` are `float16` and `pse_shift` is provided.
- **`padding_mask`**: Reserved parameter, currently not used. Retain the default value.
- **`atten_mask`** (`Tensor`): Optional. Inverse triangular mask matrix where the lower triangle is all zeros and the upper triangle is negative infinity, $atten\_mask$ in the formula. The data type can be `bool`, `int8`, or `uint8`. The data layout can be ND. Non-contiguous tensors are not supported. This parameter can be set to `None` if this feature is not used. Recommended shapes are `(Q_S, KV_S)`, `(B, Q_S, KV_S)`, `(1, Q_S, KV_S)`, `(B, 1, Q_S, KV_S)`, or `(1, 1, Q_S, KV_S)`, where `Q_S` represents `S` in the shape of `query`, and `KV_S` represents `S` in the shapes of `key` and `value`. In scenarios where `KV_S` of `atten_mask` is not 32-byte aligned, padding to 32 bytes is recommended to improve performance. Redundant elements must be padded with `1`. For details about the comprehensive constraints, see [Constraints](#section12345537164214).
- **`actual_seq_lengths`** (`List[int]`): Optional. Valid sequence lengths of `query` across different batches. The data type can be `int64`. If `seqlen` is not specified, set this parameter to `None`, indicating that the value is identical to the sequence length `S` in the `query` shape. Restriction: Valid sequence lengths in each batch must be less than or equal to the corresponding sequence length of that batch in `query`. When the input length of `seqlen` is `1`, each batch uses the same `seqlen`. When the input length of `seqlen` is greater than or equal to the batch count, the first batch number of elements in `seqlen` are selected. Other lengths are not supported.
    - Atlas inference accelerator cards: This parameter is not supported currently.
    - Atlas A2 training products/Atlas A2 inference products: The TND layout is supported. When the `input_layout` of `query` is TND, this parameter must be provided, and the number of elements in this parameter defines the batch size. The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element, and cannot be negative.
    - Atlas A3 training products/Atlas A3 inference products: The TND layout is supported. When the `input_layout` of `query` is TND, this parameter must be provided, and the number of elements in this parameter defines the batch size. The value of each element in this parameter indicates the sum of sequence lengths of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element, and cannot be negative.
- **`deq_scale1`** (`Tensor`): Optional. Dequantization factor after BMM1. `pertensor` mode is supported. The data type can be `uint64` or `float32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used. Atlas inference accelerator cards: This parameter is not supported currently.
- **`quant_scale1`** (`Tensor`): Optional. Quantization factor before BMM2. `pertensor` mode is supported. The data type can be `float32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used. Atlas inference accelerator cards: This parameter is not supported currently.
- **`deq_scale2`** (`Tensor`): Optional. Dequantization factor after BMM2. `pertensor` mode is supported. The data type can be `uint64` or `float32`. The data layout can be ND. This parameter can be set to `None` if this feature is not used. Atlas inference accelerator cards: This parameter is not supported currently.
- **`quant_scale2`** (`Tensor`): Optional. Output quantization factor. The data layout can be ND. `pertensor` and `perchannel` modes are supported. This parameter can be set to `None` if this feature is not used.
    - Atlas inference accelerator cards: This parameter is not supported currently.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float32` or `bfloat16`. When the input data type is `bfloat16`, both `float32` and `bfloat16` are supported. Otherwise, only `float32` is supported. In `perchannel` mode, when the output layout is `BSH`, the product of all dimensions of `quant_scale2` must be identical to `H`. For other layouts, the product must be identical to `N * D`. The recommended shape is `(1, 1, H)` or `(H,)` for the `BSH` layout, `(1, N, 1, D)` or `(N, D)` for the `BNSD` layout, and `(1, 1, N, D)` or `(N, D)` for the `BSND` layout.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float32` or `bfloat16`. When the input data type is `bfloat16`, both `float32` and `bfloat16` are supported. Otherwise, only `float32` is supported. In `perchannel` mode, when the output layout is `BSH`, the product of all dimensions of `quant_scale2` must be identical to `H`. For other layouts, the product must be identical to `N * D`. The recommended shape is `(1, 1, H)` or `(H,)` for the `BSH` layout, `(1, N, 1, D)` or `(N, D)` for the `BNSD` layout, and `(1, 1, N, D)` or `(N, D)` for the `BSND` layout.

- **`quant_offset2`** (`Tensor`): Optional. Output quantization offset. `pertensor` and `perchannel` modes are supported. The data layout can be ND. If `quant_offset2` is provided, its data type and shape must be identical to those of `quant_scale2`. This parameter can be set to `None` if this feature is not used.
    - Atlas inference accelerator cards: This parameter is not supported currently.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float32` or `bfloat16`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float32` or `bfloat16`.

- **`num_heads`** (`List[int]`): Optional. Head count of `query`. The data type can be `int64`.
- **`scale_value`** (`float`): Optional. Scaling factor serving as the scalar value of Muls in the computation flow, $scale$ in the formula. The value is typically the reciprocal of the square root of $d$. The data type can be `float`. Its data type and the data type of `query` must meet the type deduction rules. The default value is `1.0`.
- **`pre_tokens`** (`int`): Optional. Number of preceding tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `2147483647`. Only the default value `2147483647` is supported for Atlas inference accelerator cards.
- **`next_tokens`** (`int`): Optional. Number of subsequent tokens to associate in attention computation for sparse computation. The data type can be `int64`. The default value is `0`. Only `0` and `2147483647` are supported for Atlas inference accelerator cards.
- **`input_layout`** (`str`): Optional. Data layout configuration for the input `query`, `key`, and `value`. Supported values are `"BSH"`, `"BSND"`, `"BNSD"`, or `"BNSD_BSND"`. When the input layout is `"BNSD"`, the output layout is `"BSND"`. The default value is `"BSH"`.
- **`num_key_value_heads`**: Optional. Number of heads for `key` and `value`, used to support Grouped-Query Attention (GQA) scenarios. The data type can be `int64`. The default value is `0`, indicating that the head counts of `key`, `value`, and `query` are identical. Restrictions: `num_heads` must be divisible by `num_key_value_heads`. The ratio of `num_heads` to `num_key_value_heads` must be less than or equal to `64`. In `"BSND"`, `"BNSD"`, or `"BNSD_BSND"` layouts, this value must be identical to the shape value of the n-axis in `key`/`value`. If this condition is not satisfied, the system raises an error. Only the default value `0` is supported for Atlas inference accelerator cards.
- **`actual_seq_lengths_kv`** (`int`): Optional. Valid sequence lengths of `key`/`value` across different batches. The data type can be `int64`. Restrictions: Valid sequence lengths in each batch within this parameter must be less than or equal to the sequence length of the corresponding batch in `key`/`value`. When the input length of `seqlenKV` is `1`, each batch uses the same `seqlenKV`. When the input length is greater than or equal to the batch count, the first batch number of elements are selected. Other lengths are not supported.
    - Atlas inference accelerator cards: This parameter is not supported currently.
    - Atlas A2 training products/Atlas A2 inference products: The TND layout is supported. When the `input_layout` of `key`/`value` is TND, this parameter must be provided, and the number of elements in this parameter defines the batch size. The value of each element in this parameter indicates the sum of `seqlenKV` values of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element, and cannot be negative.
    - Atlas A3 training products/Atlas A3 inference products: The TND layout is supported. When the `input_layout` of `key`/`value` is TND, this parameter must be provided, and the number of elements in this parameter defines the batch size. The value of each element in this parameter indicates the sum of `seqlenKV` values of the current batch and all previous batches. Therefore, the value of each element must be greater than or equal to that of the previous element, and cannot be negative.
- **`sparse_mode`** (`int`): Optional. Sparsification mode. The data type can be `int64`. The default value is `0`. For details about comprehensive constraints, see [Constraints](#section12345537164214). Only the default value `0` is supported for Atlas inference accelerator cards.
    - `0` enables `defaultMask` mode. If `atten_mask` is omitted, no mask computation is executed, and `pre_tokens` and `next_tokens` are ignored (internally set to `INT_MAX`). If `atten_mask` is provided, the complete `atten_mask` matrix with shape `(S1, S2)` must be provided, indicating that the region between `pre_tokens` and `next_tokens` is involved in attention computation. The configuration where an entire row in the computation region of the input mask matrix is `1` is not supported.
    - `1` enables `allMask` mode. The configuration where an entire row in the computation region of the input mask matrix is `1` is not supported.
    - `2` enables `leftUpCausal` mask mode. This parameter requires an optimized `atten_mask` matrix with shape `(2048, 2048)`.
    - `3` enables `rightDownCausal` mask mode. This parameter requires an optimized `atten_mask` matrix with shape `(2048, 2048)`. This mode corresponds to lower-triangular scenarios partitioned from the upper-left vertex.
    - `4` enables `band` mask mode. This parameter requires an optimized `atten_mask` matrix with shape `(2048, 2048)`.
    - `5`, `6`, `7`, or `8` enables `prefix`, `global`, `dilated`, and `block_local` modes, respectively. Currently, these configurations are not supported.

## Return Values

`Tensor`

Final computation result, $atten\_out$ in the formula. When `input_layout` is `"BNSD_BSND"`, the shape of the input `query` is `BNSD` and the output shape is `BSND`. In all other configurations, the output shape must be identical to the shape of `query`.

## Constraints<a name="section12345537164214"></a>

- This API can be used in inference scenarios.
- This API supports graph mode.
- When this API is used together with PyTorch, ensure that the CANN package versions match the PyTorch package versions.
- Empty input handling: The operator checks whether `query` is empty internally. If `query` is empty, an empty value is returned. When `query` is a non-empty tensor and `key` and `value` are empty tensors (that is, `S2` is `0`), an output tensor with the corresponding shape is filled with zeros (`atten_out`). When `atten_out` is an empty tensor, it is handled by the framework.
- Usage constraints for `query`, `key`, and `value` inputs:

    <a name="en-us_topic_0000001798619409_table382212695610"></a>
    <table><thead align="left"><tr id="en-us_topic_0000001798619409_row1882282685619"><th class="cellrowborder" valign="top" width="30.570000000000004%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001798619409_p182262616569"><a name="en-us_topic_0000001798619409_p182262616569"></a><a name="en-us_topic_0000001798619409_p182262616569"></a>Product Model</p>
    </th>
    <th class="cellrowborder" valign="top" width="69.43%" id="mcps1.1.3.1.2"><p id="en-us_topic_0000001798619409_p1682262645612"><a name="en-us_topic_0000001798619409_p1682262645612"></a><a name="en-us_topic_0000001798619409_p1682262645612"></a>Axis Constraints</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="en-us_topic_0000001798619409_row88221926155613"><td class="cellrowborder" valign="top" width="30.570000000000004%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001798619409_p282310268569"><a name="en-us_topic_0000001798619409_p282310268569"></a><a name="en-us_topic_0000001798619409_p282310268569"></a><span id="en-us_topic_0000001798619409_ph999933035615"><a name="en-us_topic_0000001798619409_ph999933035615"></a><a name="en-us_topic_0000001798619409_ph999933035615"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term15651172142210_16"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term15651172142210_16"></a>Atlas inference accelerator cards</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.43%" headers="mcps1.1.3.1.2 "><a name="en-us_topic_0000001798619409_ul849244711561"></a><a name="en-us_topic_0000001798619409_ul849244711561"></a><ul id="en-us_topic_0000001798619409_ul849244711561"><li>The B-axis value can be less than or equal to 128. </li><li>The N-axis value can be less than or equal to 256. </li><li>The S-axis value can be less than or equal to 65536 (64K). </li><li>The D-axis value can be less than or equal to 512.</li></ul>
    </td>
    </tr>
    <tr id="en-us_topic_0000001798619409_row1782312260569"><td class="cellrowborder" valign="top" width="30.570000000000004%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001798619409_p128231426175620"><a name="en-us_topic_0000001798619409_p128231426175620"></a><a name="en-us_topic_0000001798619409_p128231426175620"></a><span id="en-us_topic_0000001798619409_ph12807852115620"><a name="en-us_topic_0000001798619409_ph12807852115620"></a><a name="en-us_topic_0000001798619409_ph12807852115620"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term11962195213215_7"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term11962195213215_7"></a>Atlas A2 training products/Atlas A2 inference products</span></p>
    <p id="en-us_topic_0000001798619409_p16629103195810"><a name="en-us_topic_0000001798619409_p16629103195810"></a><a name="en-us_topic_0000001798619409_p16629103195810"></a><span id="en-us_topic_0000001798619409_ph1464135665612"><a name="en-us_topic_0000001798619409_ph1464135665612"></a><a name="en-us_topic_0000001798619409_ph1464135665612"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term1253731311225_7"></a><a name="en-us_topic_0000001798619409_en-us_topic_0000001312391781_term1253731311225_7"></a>Atlas A3 training products/Atlas A3 inference products</span></p>
    </td>
    <td class="cellrowborder" valign="top" width="69.43%" headers="mcps1.1.3.1.2 "><a name="en-us_topic_0000001798619409_ul103175710575"></a><a name="en-us_topic_0000001798619409_ul103175710575"></a><ul id="en-us_topic_0000001798619409_ul103175710575"><li>The B-axis value must be less than or equal to 65536 (64K). When the D-axis is not 32-byte aligned, only values less than or equal to 128 are supported. </li><li>The N-axis value can be less than or equal to 256. </li><li>The S-axis value must be less than or equal to 20971520 (20M). In long-sequence scenarios, an excessive computation workload can cause the PFA operator execution to time out, which raises an AI Core error (errorStr: "timeout or trap error"). In this configuration, S-axis splitting is recommended. The computation workload is affected by the B, S, N, and D axes. Larger values cause a higher computation workload. Typical long-sequence scenarios that can cause a timeout (scenarios where the product of B, S, N, and D is large) include:<a name="en-us_topic_0000001798619409_ul103175718575"></a><a name="en-us_topic_0000001798619409_ul103175718575"></a><ul id="en-us_topic_0000001798619409_ul103175718575"><li>B=1, Q<sub>N</sub>=20, Q<sub>S</sub>=1048576, D = 256, KV<sub>N</sub>=1, KV<sub>S</sub>=1048576. </li><li>B=1, Q<sub>N</sub>=2, Q<sub>S</sub>=10485760, D = 256, KV<sub>N</sub>=2, KV<sub>S</sub>=10485760. </li><li>B=20, Q<sub>N</sub>=1, Q<sub>S</sub>=1048576, D = 256, KV<sub>N</sub>=1, KV<sub>S</sub>=1048576. </li><li>B=1, Q<sub>N</sub>=10, Q<sub>S</sub>=1048576, D = 512, KV<sub>N</sub>=1, KV<sub>S</sub>=1048576.</li></ul>
    </li><li>The D-axis value can be less than or equal to 512. When <code>input_layout</code> is <code>"BSH"</code> or <code>"BSND"</code>, the product of <code>N</code> and <code>D</code> must be less than 65535.</li></ul>
    </td>
    </tr>
    </tbody>
    </table>

- The `sparse_mode` parameter must be set to `0`, `1`, `2`, `3`, or `4`. If it is set to other values, an error is raised.
    - When `sparse_mode` is `0`, if `atten_mask` is a null pointer, the input parameters `pre_tokens` and `next_tokens` are ignored and internally set to `INT_MAX`.
    - When `sparse_mode` is `2`, `3`, or `4`, the shape of `atten_mask` must be `(S, S)`, `(1, S, S)`, or `(1, 1, S, S)`, where `S` must be `2048`. The provided `atten_mask` must be a lower triangular matrix. If `atten_mask` is omitted or has an invalid shape, an error is raised.
    - When `sparse_mode` is `1`, `2`, or `3`, the input parameters `pre_tokens` and `next_tokens` are ignored. The values are assigned based on the corresponding rules.

- Comprehensive constraints on the number of `int8` quantization-related parameters and the input or output data formats:
    - When the input is `int8` and the output is `int8`, the input parameters `deq_scale1`, `quant_scale1`, `deq_scale2`, and `quant_scale2` must all be provided. `quant_offset2` is optional. If omitted, the default value `0` is used.
    - When the input is `int8` and the output is `float16`, the input parameters `deq_scale1`, `quant_scale1`, and `deq_scale2` must all be provided. If `quant_offset2` or `quant_scale2` is provided, an error is raised and returned.
    - When the input is `float16` or `bfloat16` and the output is `int8`, the input parameter `quant_scale2` must be provided. `quant_offset2` is optional. If omitted, the default value `0` is used. If `deq_scale1`, `quant_scale1`, or `deq_scale2` is provided (not set to `None`), an error is raised and returned.
    - The input parameters `quant_offset2` and `quant_scale2` support the `pertensor` and `perchannel` modes. The data type can be `float32` or `bfloat16`. If `quant_offset2` is provided, its data type and shape must be identical to those of `quant_scale2`. When the input data type is `bfloat16`, both `float32` and `bfloat16` are supported. Otherwise, only `float32` is supported. In `perchannel` mode, when the output layout is `BSH`, the product of all dimensions of `quant_scale2` must be identical to `H`. For other layouts, the product must be identical to `N * D`. The recommended shape is `(1, 1, H)` or `(H,)` for the `BSH` layout, `(1, N, 1, D)` or `(N, D)` for the `BNSD` layout, and `(1, 1, N, D)` or `(N, D)` for the `BSND` layout. In `pertensor` mode, alignment of the D-axis to 32 bytes is recommended.
    - In `perchannel` mode, the input parameters `quant_scale2` and `quant_offset2` do not support left padding, Ring Attention, or scenarios where the $D$ dimension is not 32-byte aligned.
    - When the output is `int8`, the scenario where the sparse mode is `"band"` and `pre_tokens` or `next_tokens` is negative is not supported.

- Usage constraints for the `pse_shift` feature:
    - Supported when the data type of `query` is `float16`, `bfloat16`, or `int8`.
    - When `query`, `key`, and `value` are all `float16` and `pse_shift` is provided, high-precision mode is enforced. The constraints are the same as those of high-precision mode.
    - `Q_S` must be greater than or equal to the sequence length (`S`) of `query`, and `KV_S` must be greater than or equal to the sequence length (`S`) of `key`.

- When the output is `int8`, the input parameter `quant_offset2` is provided as a non-null pointer and a non-empty `Tensor`, and `sparse_mode`, `pre_tokens`, and `next_tokens` meet the following conditions, some matrix rows do not participate in computation. This causes computation errors, and execution will be blocked:
    - When `sparse_mode` is `0`, if `atten_mask` is a non-null pointer and, for each batch, `actual_seq_lengths - actual_seq_lengths_kv - pre_tokens > 0` or `next_tokens < 0`, the execution will be blocked.
    - When `sparse_mode` is `1` or `2`, the execution will not be blocked.
    - When `sparse_mode` is `3`, if, for each batch, `actual_seq_lengths_kv - actual_seq_lengths < 0`, the execution will be blocked.
    - When `sparse_mode` is `4`, if `pre_tokens < 0` or, for each batch, `next_tokens + actual_seq_lengths_kv - actual_seq_lengths < 0`, the execution will be blocked.

- KV fake-quantization parameter separation is not supported.
- Scenarios with unaligned dimensions are not supported.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import math
    >>>
    >>> # Generate random data and send it to the NPU.
    >>> q = torch.randn(1, 8, 164, 128, dtype=torch.float16).npu()
    >>> k = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    >>> v = torch.randn(1, 8, 1024, 128, dtype=torch.float16).npu()
    >>> scale = 1/math.sqrt(128.0)
    >>> actseqlen = [164]
    >>> actseqlenkv = [1024]
    >>>
    >>> # Call the PFA operator.
    >>> out = torch_npu.npu_prompt_flash_attention(q, k, v,
    ... actual_seq_lengths = actseqlen, actual_seq_lengths_kv = actseqlenkv,
    ... num_heads = 8, input_layout = "BNSD", scale_value=scale, pre_tokens=65535, next_tokens=65535)
    >>> out.shape
    torch.Size([1, 8, 164, 128])
    >>> out.dtype
    torch.float16
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
