# torch_npu.contrib.module.LinearQuant

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    √     |
| <term>Atlas inference products</term>    |    √     |

## Function

Encapsulates the `torch_npu.npu_quant_matmul` API to perform matrix multiplication computations for A8W8 and A4W4 quantized operators.

## Prototype

```python
torch_npu.contrib.module.LinearQuant(in_features, out_features, *, bias=True, offset=False, pertoken_scale=False, device=None, dtype=None, output_dtype=None)
```

## Parameters

**Computation Parameters**

- **`in_features`** (`int`): Value of the K dimension in the matrix multiplication computation.
- **`out_features`** (`int`): Value of the N dimension in the matrix multiplication computation.
- **`bias`** (`bool`): Specifies whether to include bias in the computation. If set to `False`, `bias` is excluded from the quantized matrix multiplication computation.
- **`offset`** (`bool`): Specifies whether to include offset in the computation. If set to `False`, `offset` is excluded from the quantized matrix multiplication computation.
- **`pertoken_scale`** (`bool`): Optional. Specifies whether to include `pertoken_scale` parameters in the computation. If set to `False`, `pertoken_scale` is excluded from the quantized matrix multiplication computation. <term>Atlas inference products</term>: Currently, this parameter is not supported.
- **`device`**: The default value is `None`. **Reserved parameter, currently not used.**
- **`dtype`**: The default value is `None`. **Reserved parameter, currently not used.**
- **`output_dtype`** (`ScalarType`):Data type of the output tensor. The default value is `None`, indicating that the data type of the output tensor is `int8`.
    - <term>Atlas inference products</term>: The input data type can be `int8` or `float16`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The input data type can be `int8`, `float16`, `bfloat16`, or `int32`.

**Computation Input**

**`x1`** (`Tensor`): The data layout can be ND. This parameter must have 2 to 6 dimensions.

- <term>Atlas inference products</term>: The data type can be `int8`.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `int8` or `int32`. `int32` indicates that this API performs `int4` matrix multiplication computations, where the `int32` data type carries the `int4` data, and each individual `int32` element stores eight `int4` values.

## Variable Description

- **`weight`** (`Tensor`): The data type must be identical to that of `x1`. The data layout can be ND. The shape must have 2 to 6 dimensions. When the data type is `int32`, the shape must have 2 dimensions.
    - <term>Atlas inference products</term>: The data type can be `int8`. You must call `torchair.experimental.inference.use_internal_format_weight` or `torch_npu.npu_format_cast` to configure the high-performance data layout for `weight` with shape `(batch, n, k)`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `int8` or `int32`. `int32` matches the behavior of `x1`, indicating `int4` matrix multiplication computations. You must call `torch_npu.npu_format_cast` to configure the high-performance data layout for `weight` with shape `(batch, n, k)`. However, using this module method is not recommended. Use `torch_npu.npu_quant_matmul` instead.

- **`scale`** (`Tensor`): Scale for quantized computation. The data layout can be ND. This parameter must be 1D with shape `(t,)`, where `t = 1` or `t = n`, and `n` matches the `n` dimension of `weight`. If an `int64` `scale` is required, call `torch_npu.npu_trans_quant_param` in advance to obtain the `int64` `scale`.
    - <term>Atlas inference products</term>: The data type can be `float32` or `int64`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `float32`, `int64,` or `bfloat16`.

- **`offset`** (`Tensor`): Optional. Offset for quantized computation. The data type can be `float32`. The data layout can be ND. This parameter must be 1D with shape `(t,)`, where `t = 1` or `t = n`, and `n` matches the `n` dimension of `weight`.
- **`pertoken_scale`** (`Tensor`): Optional. Per-token scale tensor for quantized computation. The data type can be `float32`. The data layout can be ND. This parameter must be 1D with shape `(m,)`, where `m` matches the `m` dimension of `weight`. <term>Atlas inference products</term>: Currently, this parameter is not supported.
- **`bias`** (`Tensor`): Optional. Bias in matrix multiplication. The data layout can be ND. This parameter must be 1D with shape `(n,)` or 3D with shape `(batch, 1, n)`, where `n` matches the `n` dimension of `weight`. The `batch` value must match the `batch` value deduced after broadcasting `x1` and `weight`. When the output tensor has 2, 4, 5, or 6 dimensions, `bias` must be a 1D tensor. When the output tensor has 3 dimensions, `bias` can be a 1D or 3D tensor.
    - <term>Atlas inference products</term>: The data type can be `int32`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The data type can be `int32`, `bfloat16`, `float16`, or `float32`.

- **`output_dtype`** (`ScalarType`): Optional. Data type of the output tensor. The default value is `None`, indicating that the data type of the output tensor is `int8`.
    - <term>Atlas inference products</term>: The input data type can be `int8` or `float16`.
    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The input data type can be `int8`, `float16`, `bfloat16`, or `int32`.

## Return Values

`Tensor`

Output tensor representing the computation result of quantized matrix multiplication.

- If `output_dtype` is `"int8"` or `None`, the output data type is `int8`.
- If `output_dtype` is `"float16"`, the output data type is `float16`.
- If `output_dtype` is `"bfloat16"`, the output data type is `bfloat16`.
- If `output_dtype` is `"int32"`, the output data type is `int32`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- `x1`, `weight`, and `scale` must be provided.
- The last dimension of `x1` and `weight` must be less than or equal to 65535.
- Additional constraints for `int4` computation:

    When the data types of both `x1` and `weight` are `int32`, each `int32` element stores eight `int4` values. The last dimension of the input shape must be reduced by a factor of 8 relative to the original `int4` shape. The size of the last dimension of the original `int4` tensors must be a multiple of `8`. For example, when performing `int4` matrix multiplication with input shapes `(m, k)` and `(k, n)`, the inputs must be `int32` tensors with shapes `(m, k//8)` and `(k, n//8)`, where both `k` and `n` must be multiples of `8`. `x1` accepts only contiguous tensors with shape `(m, k//8)`, and `weight` accepts only contiguous tensors with shape `(n, k//8)`.

    > [!NOTE]  
    > A contiguous data layout means that all adjacent elements in a tensor are stored in contiguous memory locations, including across row boundaries. If `Tensor.is_contiguous()` returns `True`, the tensor layout is considered contiguous.

- The following table describes the supported data type combinations for the input parameters and variables.

    **Table 1** <term>Atlas inference products</term>

    <a name="en-us_topic_0000002021380113_table75025595916"></a>
    <table><thead align="left"><tr id="en-us_topic_0000002021380113_row13503185919911"><th class="cellrowborder" valign="top" width="11.32113211321132%" id="mcps1.2.8.1.1"><p id="en-us_topic_0000002021380113_p35036591098"><a name="en-us_topic_0000002021380113_p35036591098"></a><a name="en-us_topic_0000002021380113_p35036591098"></a><code>x1</code> (Input Parameter)</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.871187118711871%" id="mcps1.2.8.1.2"><p id="en-us_topic_0000002021380113_p450320598916"><a name="en-us_topic_0000002021380113_p450320598916"></a><a name="en-us_topic_0000002021380113_p450320598916"></a><code>weight</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.991399139913991%" id="mcps1.2.8.1.3"><p id="en-us_topic_0000002021380113_p125032592912"><a name="en-us_topic_0000002021380113_p125032592912"></a><a name="en-us_topic_0000002021380113_p125032592912"></a><code>scale</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.931293129312932%" id="mcps1.2.8.1.4"><p id="en-us_topic_0000002021380113_p750375912919"><a name="en-us_topic_0000002021380113_p750375912919"></a><a name="en-us_topic_0000002021380113_p750375912919"></a><code>offset</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="15.41154115411541%" id="mcps1.2.8.1.5"><p id="en-us_topic_0000002021380113_p16503175913919"><a name="en-us_topic_0000002021380113_p16503175913919"></a><a name="en-us_topic_0000002021380113_p16503175913919"></a><code>bias</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.001700170017003%" id="mcps1.2.8.1.6"><p id="en-us_topic_0000002021380113_p1150395911911"><a name="en-us_topic_0000002021380113_p1150395911911"></a><a name="en-us_topic_0000002021380113_p1150395911911"></a><code>pertoken_scale</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.47174717471747%" id="mcps1.2.8.1.7"><p id="en-us_topic_0000002021380113_p15503135912911"><a name="en-us_topic_0000002021380113_p15503135912911"></a><a name="en-us_topic_0000002021380113_p15503135912911"></a><code>output_dtype</code> (Input Parameter or Variable)</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="en-us_topic_0000002021380113_row4503125913910"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000002021380113_p150320591294"><a name="en-us_topic_0000002021380113_p150320591294"></a><a name="en-us_topic_0000002021380113_p150320591294"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000002021380113_p1450395915920"><a name="en-us_topic_0000002021380113_p1450395915920"></a><a name="en-us_topic_0000002021380113_p1450395915920"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.991399139913991%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000002021380113_p115031259296"><a name="en-us_topic_0000002021380113_p115031259296"></a><a name="en-us_topic_0000002021380113_p115031259296"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.931293129312932%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000002021380113_p1050319597913"><a name="en-us_topic_0000002021380113_p1050319597913"></a><a name="en-us_topic_0000002021380113_p1050319597913"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.41154115411541%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000002021380113_p85035591799"><a name="en-us_topic_0000002021380113_p85035591799"></a><a name="en-us_topic_0000002021380113_p85035591799"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.001700170017003%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000002021380113_p750312591699"><a name="en-us_topic_0000002021380113_p750312591699"></a><a name="en-us_topic_0000002021380113_p750312591699"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.47174717471747%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p550365911917"><a name="en-us_topic_0000002021380113_p550365911917"></a><a name="en-us_topic_0000002021380113_p550365911917"></a>float16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row750310595920"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000002021380113_p1750315599914"><a name="en-us_topic_0000002021380113_p1750315599914"></a><a name="en-us_topic_0000002021380113_p1750315599914"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000002021380113_p1850317591393"><a name="en-us_topic_0000002021380113_p1850317591393"></a><a name="en-us_topic_0000002021380113_p1850317591393"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.991399139913991%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000002021380113_p95034598912"><a name="en-us_topic_0000002021380113_p95034598912"></a><a name="en-us_topic_0000002021380113_p95034598912"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.931293129312932%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000002021380113_p14503359395"><a name="en-us_topic_0000002021380113_p14503359395"></a><a name="en-us_topic_0000002021380113_p14503359395"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.41154115411541%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000002021380113_p25033591296"><a name="en-us_topic_0000002021380113_p25033591296"></a><a name="en-us_topic_0000002021380113_p25033591296"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.001700170017003%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000002021380113_p115035592090"><a name="en-us_topic_0000002021380113_p115035592090"></a><a name="en-us_topic_0000002021380113_p115035592090"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.47174717471747%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p75033595916"><a name="en-us_topic_0000002021380113_p75033595916"></a><a name="en-us_topic_0000002021380113_p75033595916"></a>int8</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row183492120464"><td class="cellrowborder" colspan="7" valign="top" headers="mcps1.2.8.1.1 mcps1.2.8.1.2 mcps1.2.8.1.3 mcps1.2.8.1.4 mcps1.2.8.1.5 mcps1.2.8.1.6 mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p16486127124614"><a name="en-us_topic_0000002021380113_p16486127124614"></a><a name="en-us_topic_0000002021380113_p16486127124614"></a>Note: None indicates that the input parameter or variable is <code>False</code></p>
    </td>
    </tr>
    </tbody>
    </table>

    **Table 2** <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>

    <a name="en-us_topic_0000002021380113_table2504155910917"></a>
    <table><thead align="left"><tr id="en-us_topic_0000002021380113_row35048591395"><th class="cellrowborder" valign="top" width="11.32113211321132%" id="mcps1.2.8.1.1"><p id="en-us_topic_0000002021380113_p185044592912"><a name="en-us_topic_0000002021380113_p185044592912"></a><a name="en-us_topic_0000002021380113_p185044592912"></a><code>x1</code> (Input Parameter)</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.871187118711871%" id="mcps1.2.8.1.2"><p id="en-us_topic_0000002021380113_p1750415591899"><a name="en-us_topic_0000002021380113_p1750415591899"></a><a name="en-us_topic_0000002021380113_p1750415591899"></a><code>weight</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="14.161416141614161%" id="mcps1.2.8.1.3"><p id="en-us_topic_0000002021380113_p125041959693"><a name="en-us_topic_0000002021380113_p125041959693"></a><a name="en-us_topic_0000002021380113_p125041959693"></a><code>scale</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.921292129212922%" id="mcps1.2.8.1.4"><p id="en-us_topic_0000002021380113_p145043591797"><a name="en-us_topic_0000002021380113_p145043591797"></a><a name="en-us_topic_0000002021380113_p145043591797"></a><code>offset</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="15.251525152515253%" id="mcps1.2.8.1.5"><p id="en-us_topic_0000002021380113_p1250455915910"><a name="en-us_topic_0000002021380113_p1250455915910"></a><a name="en-us_topic_0000002021380113_p1250455915910"></a><code>bias</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.461746174617463%" id="mcps1.2.8.1.6"><p id="en-us_topic_0000002021380113_p125045596917"><a name="en-us_topic_0000002021380113_p125045596917"></a><a name="en-us_topic_0000002021380113_p125045596917"></a><code>pertoken_scale</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.01170117011701%" id="mcps1.2.8.1.7"><p id="en-us_topic_0000002021380113_p6504959094"><a name="en-us_topic_0000002021380113_p6504959094"></a><a name="en-us_topic_0000002021380113_p6504959094"></a><code>output_dtype</code> (Input Parameter or Variable)</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="en-us_topic_0000002021380113_row1850445912918"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000002021380113_p105042591492"><a name="en-us_topic_0000002021380113_p105042591492"></a><a name="en-us_topic_0000002021380113_p105042591492"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000002021380113_p1550412593911"><a name="en-us_topic_0000002021380113_p1550412593911"></a><a name="en-us_topic_0000002021380113_p1550412593911"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000002021380113_p35044591897"><a name="en-us_topic_0000002021380113_p35044591897"></a><a name="en-us_topic_0000002021380113_p35044591897"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000002021380113_p2231164184310"><a name="en-us_topic_0000002021380113_p2231164184310"></a><a name="en-us_topic_0000002021380113_p2231164184310"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000002021380113_p175042059793"><a name="en-us_topic_0000002021380113_p175042059793"></a><a name="en-us_topic_0000002021380113_p175042059793"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000002021380113_p15627147184415"><a name="en-us_topic_0000002021380113_p15627147184415"></a><a name="en-us_topic_0000002021380113_p15627147184415"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p1950435916910"><a name="en-us_topic_0000002021380113_p1950435916910"></a><a name="en-us_topic_0000002021380113_p1950435916910"></a>float16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row1650414599917"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000002021380113_p135041599910"><a name="en-us_topic_0000002021380113_p135041599910"></a><a name="en-us_topic_0000002021380113_p135041599910"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000002021380113_p950413591090"><a name="en-us_topic_0000002021380113_p950413591090"></a><a name="en-us_topic_0000002021380113_p950413591090"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000002021380113_p2504105916914"><a name="en-us_topic_0000002021380113_p2504105916914"></a><a name="en-us_topic_0000002021380113_p2504105916914"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000002021380113_p2050415914911"><a name="en-us_topic_0000002021380113_p2050415914911"></a><a name="en-us_topic_0000002021380113_p2050415914911"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000002021380113_p18504105920916"><a name="en-us_topic_0000002021380113_p18504105920916"></a><a name="en-us_topic_0000002021380113_p18504105920916"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000002021380113_p550512597915"><a name="en-us_topic_0000002021380113_p550512597915"></a><a name="en-us_topic_0000002021380113_p550512597915"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p1750585915914"><a name="en-us_topic_0000002021380113_p1750585915914"></a><a name="en-us_topic_0000002021380113_p1750585915914"></a>int8</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row175059594912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000002021380113_p550515591592"><a name="en-us_topic_0000002021380113_p550515591592"></a><a name="en-us_topic_0000002021380113_p550515591592"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000002021380113_p115052059396"><a name="en-us_topic_0000002021380113_p115052059396"></a><a name="en-us_topic_0000002021380113_p115052059396"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000002021380113_p050514591297"><a name="en-us_topic_0000002021380113_p050514591297"></a><a name="en-us_topic_0000002021380113_p050514591297"></a>float32/bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000002021380113_p1750595915910"><a name="en-us_topic_0000002021380113_p1750595915910"></a><a name="en-us_topic_0000002021380113_p1750595915910"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000002021380113_p650595920913"><a name="en-us_topic_0000002021380113_p650595920913"></a><a name="en-us_topic_0000002021380113_p650595920913"></a>int32/bfloat16/float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000002021380113_p1050518593911"><a name="en-us_topic_0000002021380113_p1050518593911"></a><a name="en-us_topic_0000002021380113_p1050518593911"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p19505959695"><a name="en-us_topic_0000002021380113_p19505959695"></a><a name="en-us_topic_0000002021380113_p19505959695"></a>bfloat16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row950520592912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000002021380113_p2505175915915"><a name="en-us_topic_0000002021380113_p2505175915915"></a><a name="en-us_topic_0000002021380113_p2505175915915"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000002021380113_p15051859596"><a name="en-us_topic_0000002021380113_p15051859596"></a><a name="en-us_topic_0000002021380113_p15051859596"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000002021380113_p750515599915"><a name="en-us_topic_0000002021380113_p750515599915"></a><a name="en-us_topic_0000002021380113_p750515599915"></a>float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000002021380113_p85051595912"><a name="en-us_topic_0000002021380113_p85051595912"></a><a name="en-us_topic_0000002021380113_p85051595912"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000002021380113_p1150511596912"><a name="en-us_topic_0000002021380113_p1150511596912"></a><a name="en-us_topic_0000002021380113_p1150511596912"></a>int32/float16/float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000002021380113_p1505135911914"><a name="en-us_topic_0000002021380113_p1505135911914"></a><a name="en-us_topic_0000002021380113_p1505135911914"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p850516591099"><a name="en-us_topic_0000002021380113_p850516591099"></a><a name="en-us_topic_0000002021380113_p850516591099"></a>float16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row750514591599"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000002021380113_p105051659594"><a name="en-us_topic_0000002021380113_p105051659594"></a><a name="en-us_topic_0000002021380113_p105051659594"></a>int32</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000002021380113_p1550525916920"><a name="en-us_topic_0000002021380113_p1550525916920"></a><a name="en-us_topic_0000002021380113_p1550525916920"></a>int32</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000002021380113_p750512591997"><a name="en-us_topic_0000002021380113_p750512591997"></a><a name="en-us_topic_0000002021380113_p750512591997"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000002021380113_p250517591991"><a name="en-us_topic_0000002021380113_p250517591991"></a><a name="en-us_topic_0000002021380113_p250517591991"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000002021380113_p19505195914919"><a name="en-us_topic_0000002021380113_p19505195914919"></a><a name="en-us_topic_0000002021380113_p19505195914919"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000002021380113_p550510599919"><a name="en-us_topic_0000002021380113_p550510599919"></a><a name="en-us_topic_0000002021380113_p550510599919"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p55058595918"><a name="en-us_topic_0000002021380113_p55058595918"></a><a name="en-us_topic_0000002021380113_p55058595918"></a>float16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row550595915912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000002021380113_p6505175913912"><a name="en-us_topic_0000002021380113_p6505175913912"></a><a name="en-us_topic_0000002021380113_p6505175913912"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000002021380113_p55051559192"><a name="en-us_topic_0000002021380113_p55051559192"></a><a name="en-us_topic_0000002021380113_p55051559192"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000002021380113_p75054591913"><a name="en-us_topic_0000002021380113_p75054591913"></a><a name="en-us_topic_0000002021380113_p75054591913"></a>float32/bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000002021380113_p45064591697"><a name="en-us_topic_0000002021380113_p45064591697"></a><a name="en-us_topic_0000002021380113_p45064591697"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000002021380113_p7506259691"><a name="en-us_topic_0000002021380113_p7506259691"></a><a name="en-us_topic_0000002021380113_p7506259691"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000002021380113_p115061359097"><a name="en-us_topic_0000002021380113_p115061359097"></a><a name="en-us_topic_0000002021380113_p115061359097"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p250625911912"><a name="en-us_topic_0000002021380113_p250625911912"></a><a name="en-us_topic_0000002021380113_p250625911912"></a>int32</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row8523104511"><td class="cellrowborder" colspan="7" valign="top" headers="mcps1.2.8.1.1 mcps1.2.8.1.2 mcps1.2.8.1.3 mcps1.2.8.1.4 mcps1.2.8.1.5 mcps1.2.8.1.6 mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p1899351015458"><a name="en-us_topic_0000002021380113_p1899351015458"></a><a name="en-us_topic_0000002021380113_p1899351015458"></a>Note: None indicates that the input parameter or variable is <code>False</code>.</p>
    </td>
    </tr>
    </tbody>
    </table>

## Examples

- Single-operator call
    - Code sample for scenarios with `int8` inputs:

        ```python
        import torch
        import torch_npu
        import logging
        import os
        from torch_npu.contrib.module import LinearQuant
        x1 = torch.randint(-1, 1, (1, 512), dtype=torch.int8).npu()
        x2 = torch.randint(-1, 1, (128, 512), dtype=torch.int8).npu()
        scale = torch.randn(1, dtype=torch.float32).npu()
        offset = torch.randn(128, dtype=torch.float32).npu()
        bias = torch.randint(-1,1, (128,), dtype=torch.int32).npu()
        in_features = 512
        out_features = 128
        output_dtype = torch.int8
        model = LinearQuant(in_features, out_features, bias=True, offset=True, output_dtype=output_dtype)
        model = model.npu()
        model.weight.data = x2
        model.scale.data = scale
        model.offset.data = offset
        model.bias.data = bias
        # Internal npu_trans_quant_param call
        output = model(x1)
        ```

    - Code sample for scenarios with `int32` inputs (supported only on the following products):

        - <term>Atlas A2 training products/Atlas A2 inference products</term>
        - <term>Atlas A3 training products/Atlas A3 inference products</term>

        ```python
        import torch
        import torch_npu
        import logging
        import os
        from torch_npu.contrib.module import LinearQuant
        # Uses int32 to carry int4 data. The actual int4 shape is x1: (1, 512) and x2: (128, 512).
        x1 = torch.randint(-1, 1, (1, 64), dtype=torch.int32).npu()
        x2 = torch.randint(-1, 1, (128, 64), dtype=torch.int32).npu()
        scale = torch.randn(1, dtype=torch.float32).npu()
        bias = torch.randint(-1,1, (128,), dtype=torch.int32).npu()
        in_features = 512
        out_features = 128
        output_dtype = torch.float16
        model = LinearQuant(in_features, out_features, bias=True, offset=False, output_dtype=output_dtype)
        model = model.npu()
        model.weight.data = x2
        model.scale.data = scale
        model.bias.data = bias
        output = model(x1)
        ```

- Code sample for graph mode call (supported only on the following products):

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    from torch_npu.contrib.module import LinearQuant
    import logging
    from torchair.core.utils import logger
    logger.setLevel(logging.DEBUG)
    import os
    import numpy as np
    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    x1 = torch.randint(-1, 1, (1, 512), dtype=torch.int8).npu()
    x2 = torch.randint(-1, 1, (128, 512), dtype=torch.int8).npu()
    scale = torch.randn(1, dtype=torch.float32).npu()
    offset = torch.randn(128, dtype=torch.float32).npu()
    bias = torch.randint(-1,1, (128,), dtype=torch.int32).npu()
    in_features = 512
    out_features = 128
    output_dtype = torch.int8
    model = LinearQuant(in_features, out_features, bias=True, offset=True, output_dtype=output_dtype)
    model = model.npu()
    model.weight.data = x2
    model.scale.data = scale
    model.offset.data = offset
    if output_dtype != torch.bfloat16:
        # Enable the data layout function for high-bandwidth x2
        tng.experimental.inference.use_internal_format_weight(model)
    model.bias.data = bias
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1)
    ```
