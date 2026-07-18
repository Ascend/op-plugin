# torch_npu.contrib.module.LinearA8W8Quant

> [!NOTICE]  
> This API is planned for deprecation. Use `torch_npu.contrib.module.LinearQuant` instead.

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    √     |
| <term>Atlas inference products</term>    |    √     |

## Function

Encapsulates the `torch_npu.npu_quant_matmul` API to perform matrix multiplication computations for the A8W8 quantized operator.

## Prototype

```python
torch_npu.contrib.module.LinearA8W8Quant(in_features, out_features, *, bias=True, offset=False, pertoken_scale=False, output_dtype=None)
```

## Parameters

**Computation Parameters**

- **`in_features`** (`int`): Value of the K dimension in the matrix multiplication computation.
- **`out_features`** (`int`): Value of the N dimension in the matrix multiplication computation.
- **`bias`** (`bool`): Specifies whether to include bias in the computation. If set to `False`, `bias` is excluded from the quantized matrix multiplication computation.
- **`offset`** (`bool`): Specifies whether to include offset in the computation. If set to `False`, `offset` is excluded from the quantized matrix multiplication computation.
- **`pertoken_scale`** (`bool`): Specifies whether to include `pertoken_scale` parameters in the computation. If set to `False`, `pertoken_scale` is excluded from the quantized matrix multiplication computation. Atlas inference products: Currently, this parameter is not supported.
- **`output_dtype`** (`ScalarType`): Data type of the output tensor. The default value is `None`, indicating that the data type of the output tensor is `int8`.
    - Atlas inference products: The input data type can be `int8` or `float16`.
    - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The input data type can be `int8`, `float16`, or `bfloat16`.

**Computation Input**

**`x1`** (`Tensor`): The data type can be `int8`. The data layout can be ND. The shape must have 2 to 6 dimensions.

## Variable Description

- **`weight`** (`Tensor`): Weight tensor used for matrix multiplication. The data type can be `int8`. The data layout can be ND. This parameter must be 2D to 6D with shape `(batch, n, k)`.
    - Atlas inference products: You must call `torchair.experimental.inference.use_internal_format_weight` or `torch_npu.npu_format_cast` to configure the high-performance data layout for `weight` with shape `(batch, n, k)`.
    - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: You must call `torch_npu.npu_format_cast` to configure the high-performance data layout for `weight` with shape `(batch, n, k)`. However, using this module method is not recommended. Use `torch_npu.npu_quant_matmul` instead.

- **`scale`** (`Tensor`): Scale for quantized computation. The data layout can be ND. This parameter must be 1D with shape `(t,)`, where `t = 1` or `t = n`, and `n` matches the `n` dimension of `weight`. If an `int64` `scale` is required, call `torch_npu.npu_trans_quant_param` in advance to obtain the `int64` `scale`.
    - Atlas inference products: The data type can be `float32` or `int64`.
    - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The data type can be `float32`, `int64`, or `bfloat16`.

- **`offset`** (`Tensor`): Optional. Offset for quantized computation. The data type can be `float32`. The data layout can be ND. This parameter must be 1D with shape `(t,)`, where `t = 1` or `t = n`, and `n` matches the `n` dimension of `weight`.
- **`pertoken_scale`** (`Tensor`): Optional. Per-token scale tensor for quantized computation. The data type can be `float32`. The data layout can be ND. This parameter must be 1D with shape `(m,)`, where `m` matches the `m` dimension of `x1`. This parameter can be non-empty only when the output data type is `float16` or `bfloat16`. Atlas inference products: Currently, this parameter is not supported.
- **`bias`** (`Tensor`): Optional. Bias in matrix multiplication. The data layout can be ND. This parameter must be 1D with shape `(n,)` or 3D with shape `(batch, 1, n)`, where `n` matches the `n` dimension of `weight`. The `batch` value must match the `batch` value deduced after broadcasting `x1` and `weight`. When the output tensor has 2, 4, 5, or 6 dimensions, `bias` must be a 1D tensor. When the output tensor has 3 dimensions, `bias` can be a 1D or 3D tensor.
    - Atlas inference products: The data type can be `int32`.
    - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The data type can be `int32`, `bfloat16`, `float16`, or `float32`.

- **`output_dtype`** (`ScalarType`): Optional. Data type of the output tensor. The default value is `None`, indicating that the data type of the output tensor is `int8`.
    - Atlas inference products: The input data type can be `int8` or `float16`.
    - Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products: The input data type can be `int8`, `float16`, or `bfloat16`.

## Return Values

`Tensor`

Output tensor representing the computation result of quantized matrix multiplication.

- If `output_dtype` is `"float16"`, the output data type is `float16`.
- If `output_dtype` is `"int8"` or `None`, the output data type is `int8`.
- If `output_dtype` is `"bfloat16"`, the output data type is `bfloat16`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- `x1`, `weight`, and `scale` must be provided.
- The last dimension of `x1` and `weight` must be less than or equal to 65535.
- The following table describes the supported data type combinations for the input parameters and variables.

    **Table 1** Atlas inference products

    <a name="en-us_topic_0000001778938168_table75025595916"></a>
    <table><thead align="left"><tr id="en-us_topic_0000001778938168_row13503185919911"><th class="cellrowborder" valign="top" width="11.32113211321132%" id="mcps1.2.8.1.1"><p id="en-us_topic_0000001778938168_p35036591098"><a name="en-us_topic_0000001778938168_p35036591098"></a><a name="en-us_topic_0000001778938168_p35036591098"></a><code>x1</code> (Input Parameter)</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.871187118711871%" id="mcps1.2.8.1.2"><p id="en-us_topic_0000001778938168_p450320598916"><a name="en-us_topic_0000001778938168_p450320598916"></a><a name="en-us_topic_0000001778938168_p450320598916"></a><code>weight</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.991399139913991%" id="mcps1.2.8.1.3"><p id="en-us_topic_0000001778938168_p125032592912"><a name="en-us_topic_0000001778938168_p125032592912"></a><a name="en-us_topic_0000001778938168_p125032592912"></a><code>scale</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.931293129312932%" id="mcps1.2.8.1.4"><p id="en-us_topic_0000001778938168_p750375912919"><a name="en-us_topic_0000001778938168_p750375912919"></a><a name="en-us_topic_0000001778938168_p750375912919"></a><code>offset</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="15.41154115411541%" id="mcps1.2.8.1.5"><p id="en-us_topic_0000001778938168_p16503175913919"><a name="en-us_topic_0000001778938168_p16503175913919"></a><a name="en-us_topic_0000001778938168_p16503175913919"></a><code>bias</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.001700170017003%" id="mcps1.2.8.1.6"><p id="en-us_topic_0000001778938168_p1150395911911"><a name="en-us_topic_0000001778938168_p1150395911911"></a><a name="en-us_topic_0000001778938168_p1150395911911"></a><code>pertoken_scale</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.47174717471747%" id="mcps1.2.8.1.7"><p id="en-us_topic_0000001778938168_p15503135912911"><a name="en-us_topic_0000001778938168_p15503135912911"></a><a name="en-us_topic_0000001778938168_p15503135912911"></a><code>output_dtype</code> (Input Parameter or Variable)</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="en-us_topic_0000001778938168_row4503125913910"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000001778938168_p150320591294"><a name="en-us_topic_0000001778938168_p150320591294"></a><a name="en-us_topic_0000001778938168_p150320591294"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000001778938168_p1450395915920"><a name="en-us_topic_0000001778938168_p1450395915920"></a><a name="en-us_topic_0000001778938168_p1450395915920"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.991399139913991%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000001778938168_p115031259296"><a name="en-us_topic_0000001778938168_p115031259296"></a><a name="en-us_topic_0000001778938168_p115031259296"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.931293129312932%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000001778938168_p1050319597913"><a name="en-us_topic_0000001778938168_p1050319597913"></a><a name="en-us_topic_0000001778938168_p1050319597913"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.41154115411541%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000001778938168_p85035591799"><a name="en-us_topic_0000001778938168_p85035591799"></a><a name="en-us_topic_0000001778938168_p85035591799"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.001700170017003%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000001778938168_p750312591699"><a name="en-us_topic_0000001778938168_p750312591699"></a><a name="en-us_topic_0000001778938168_p750312591699"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.47174717471747%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000001778938168_p550365911917"><a name="en-us_topic_0000001778938168_p550365911917"></a><a name="en-us_topic_0000001778938168_p550365911917"></a>float16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000001778938168_row750310595920"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000001778938168_p1750315599914"><a name="en-us_topic_0000001778938168_p1750315599914"></a><a name="en-us_topic_0000001778938168_p1750315599914"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000001778938168_p1850317591393"><a name="en-us_topic_0000001778938168_p1850317591393"></a><a name="en-us_topic_0000001778938168_p1850317591393"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.991399139913991%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000001778938168_p95034598912"><a name="en-us_topic_0000001778938168_p95034598912"></a><a name="en-us_topic_0000001778938168_p95034598912"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.931293129312932%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000001778938168_p14503359395"><a name="en-us_topic_0000001778938168_p14503359395"></a><a name="en-us_topic_0000001778938168_p14503359395"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.41154115411541%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000001778938168_p25033591296"><a name="en-us_topic_0000001778938168_p25033591296"></a><a name="en-us_topic_0000001778938168_p25033591296"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.001700170017003%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000001778938168_p115035592090"><a name="en-us_topic_0000001778938168_p115035592090"></a><a name="en-us_topic_0000001778938168_p115035592090"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.47174717471747%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000001778938168_p75033595916"><a name="en-us_topic_0000001778938168_p75033595916"></a><a name="en-us_topic_0000001778938168_p75033595916"></a>int8</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000001778938168_row183492120464"><td class="cellrowborder" colspan="7" valign="top" headers="mcps1.2.8.1.1 mcps1.2.8.1.2 mcps1.2.8.1.3 mcps1.2.8.1.4 mcps1.2.8.1.5 mcps1.2.8.1.6 mcps1.2.8.1.7 "><p id="en-us_topic_0000001778938168_p16486127124614"><a name="en-us_topic_0000001778938168_p16486127124614"></a><a name="en-us_topic_0000001778938168_p16486127124614"></a>Note: None indicates that the input parameter or variable is <code>False</code>.</p>
    </td>
    </tr>
    </tbody>
    </table>

    **Table 2** Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products

    <a name="en-us_topic_0000001778938168_table2504155910917"></a>
    <table><thead align="left"><tr id="en-us_topic_0000001778938168_row35048591395"><th class="cellrowborder" valign="top" width="11.32113211321132%" id="mcps1.2.8.1.1"><p id="en-us_topic_0000001778938168_p185044592912"><a name="en-us_topic_0000001778938168_p185044592912"></a><a name="en-us_topic_0000001778938168_p185044592912"></a><code>x1</code> (Input Parameter)</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.871187118711871%" id="mcps1.2.8.1.2"><p id="en-us_topic_0000001778938168_p1750415591899"><a name="en-us_topic_0000001778938168_p1750415591899"></a><a name="en-us_topic_0000001778938168_p1750415591899"></a><code>weight</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="14.161416141614161%" id="mcps1.2.8.1.3"><p id="en-us_topic_0000001778938168_p125041959693"><a name="en-us_topic_0000001778938168_p125041959693"></a><a name="en-us_topic_0000001778938168_p125041959693"></a><code>scale</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.921292129212922%" id="mcps1.2.8.1.4"><p id="en-us_topic_0000001778938168_p145043591797"><a name="en-us_topic_0000001778938168_p145043591797"></a><a name="en-us_topic_0000001778938168_p145043591797"></a><code>offset</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="15.251525152515253%" id="mcps1.2.8.1.5"><p id="en-us_topic_0000001778938168_p1250455915910"><a name="en-us_topic_0000001778938168_p1250455915910"></a><a name="en-us_topic_0000001778938168_p1250455915910"></a><code>bias</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.461746174617463%" id="mcps1.2.8.1.6"><p id="en-us_topic_0000001778938168_p125045596917"><a name="en-us_topic_0000001778938168_p125045596917"></a><a name="en-us_topic_0000001778938168_p125045596917"></a><code>pertoken_scale</code> (Variable)</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.01170117011701%" id="mcps1.2.8.1.7"><p id="en-us_topic_0000001778938168_p6504959094"><a name="en-us_topic_0000001778938168_p6504959094"></a><a name="en-us_topic_0000001778938168_p6504959094"></a><code>output_dtype</code> (Input Parameter or Variable)</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="en-us_topic_0000001778938168_row1850445912918"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000001778938168_p105042591492"><a name="en-us_topic_0000001778938168_p105042591492"></a><a name="en-us_topic_0000001778938168_p105042591492"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000001778938168_p1550412593911"><a name="en-us_topic_0000001778938168_p1550412593911"></a><a name="en-us_topic_0000001778938168_p1550412593911"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000001778938168_p35044591897"><a name="en-us_topic_0000001778938168_p35044591897"></a><a name="en-us_topic_0000001778938168_p35044591897"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000001778938168_p2231164184310"><a name="en-us_topic_0000001778938168_p2231164184310"></a><a name="en-us_topic_0000001778938168_p2231164184310"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000001778938168_p175042059793"><a name="en-us_topic_0000001778938168_p175042059793"></a><a name="en-us_topic_0000001778938168_p175042059793"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000001778938168_p15627147184415"><a name="en-us_topic_0000001778938168_p15627147184415"></a><a name="en-us_topic_0000001778938168_p15627147184415"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000001778938168_p1950435916910"><a name="en-us_topic_0000001778938168_p1950435916910"></a><a name="en-us_topic_0000001778938168_p1950435916910"></a>float16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000001778938168_row1650414599917"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000001778938168_p135041599910"><a name="en-us_topic_0000001778938168_p135041599910"></a><a name="en-us_topic_0000001778938168_p135041599910"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000001778938168_p950413591090"><a name="en-us_topic_0000001778938168_p950413591090"></a><a name="en-us_topic_0000001778938168_p950413591090"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000001778938168_p2504105916914"><a name="en-us_topic_0000001778938168_p2504105916914"></a><a name="en-us_topic_0000001778938168_p2504105916914"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000001778938168_p2050415914911"><a name="en-us_topic_0000001778938168_p2050415914911"></a><a name="en-us_topic_0000001778938168_p2050415914911"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000001778938168_p18504105920916"><a name="en-us_topic_0000001778938168_p18504105920916"></a><a name="en-us_topic_0000001778938168_p18504105920916"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000001778938168_p550512597915"><a name="en-us_topic_0000001778938168_p550512597915"></a><a name="en-us_topic_0000001778938168_p550512597915"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000001778938168_p1750585915914"><a name="en-us_topic_0000001778938168_p1750585915914"></a><a name="en-us_topic_0000001778938168_p1750585915914"></a>int8</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000001778938168_row175059594912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000001778938168_p550515591592"><a name="en-us_topic_0000001778938168_p550515591592"></a><a name="en-us_topic_0000001778938168_p550515591592"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000001778938168_p115052059396"><a name="en-us_topic_0000001778938168_p115052059396"></a><a name="en-us_topic_0000001778938168_p115052059396"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000001778938168_p050514591297"><a name="en-us_topic_0000001778938168_p050514591297"></a><a name="en-us_topic_0000001778938168_p050514591297"></a>float32/bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000001778938168_p1750595915910"><a name="en-us_topic_0000001778938168_p1750595915910"></a><a name="en-us_topic_0000001778938168_p1750595915910"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000001778938168_p650595920913"><a name="en-us_topic_0000001778938168_p650595920913"></a><a name="en-us_topic_0000001778938168_p650595920913"></a>int32/bfloat16/float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000001778938168_p1050518593911"><a name="en-us_topic_0000001778938168_p1050518593911"></a><a name="en-us_topic_0000001778938168_p1050518593911"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000001778938168_p19505959695"><a name="en-us_topic_0000001778938168_p19505959695"></a><a name="en-us_topic_0000001778938168_p19505959695"></a>bfloat16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000001778938168_row950520592912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="en-us_topic_0000001778938168_p2505175915915"><a name="en-us_topic_0000001778938168_p2505175915915"></a><a name="en-us_topic_0000001778938168_p2505175915915"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="en-us_topic_0000001778938168_p15051859596"><a name="en-us_topic_0000001778938168_p15051859596"></a><a name="en-us_topic_0000001778938168_p15051859596"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="en-us_topic_0000001778938168_p750515599915"><a name="en-us_topic_0000001778938168_p750515599915"></a><a name="en-us_topic_0000001778938168_p750515599915"></a>float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="en-us_topic_0000001778938168_p85051595912"><a name="en-us_topic_0000001778938168_p85051595912"></a><a name="en-us_topic_0000001778938168_p85051595912"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="en-us_topic_0000001778938168_p1150511596912"><a name="en-us_topic_0000001778938168_p1150511596912"></a><a name="en-us_topic_0000001778938168_p1150511596912"></a>int32/float16/float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="en-us_topic_0000001778938168_p1505135911914"><a name="en-us_topic_0000001778938168_p1505135911914"></a><a name="en-us_topic_0000001778938168_p1505135911914"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="en-us_topic_0000001778938168_p850516591099"><a name="en-us_topic_0000001778938168_p850516591099"></a><a name="en-us_topic_0000001778938168_p850516591099"></a>float16</p>
    </td>
    </tr>
    <tr id="en-us_topic_0000001778938168_row8523104511"><td class="cellrowborder" colspan="7" valign="top" headers="mcps1.2.8.1.1 mcps1.2.8.1.2 mcps1.2.8.1.3 mcps1.2.8.1.4 mcps1.2.8.1.5 mcps1.2.8.1.6 mcps1.2.8.1.7 "><p id="en-us_topic_0000001778938168_p1899351015458"><a name="en-us_topic_0000001778938168_p1899351015458"></a><a name="en-us_topic_0000001778938168_p1899351015458"></a>Note: None indicates that the input parameter or variable is <code>False</code>.</p>
    </td>
    </tr>
    </tbody>
    </table>

## Examples

- Single-operator call

    ```python
    # int8 input
    import torch
    import torch_npu
    import logging
    import os
    from torch_npu.contrib.module import LinearA8W8Quant
    x1 = torch.randint(-1, 1, (1, 512), dtype=torch.int8).npu()
    x2 = torch.randint(-1, 1, (128, 512), dtype=torch.int8).npu()
    scale = torch.randn(1, dtype=torch.float32).npu()
    offset = torch.randn(128, dtype=torch.float32).npu()
    bias = torch.randint(-1,1, (128,), dtype=torch.int32).npu()
    in_features = 512
    out_features = 128
    output_dtype = torch.int8
    model = LinearA8W8Quant(in_features, out_features, bias=True, offset=True, output_dtype=output_dtype)
    model = model.npu()
    model.weight.data = x2
    model.scale.data = scale
    model.offset.data = offset
    model.bias.data = bias
    output = model(x1)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    from torch_npu.contrib.module import LinearA8W8Quant
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
    model = LinearA8W8Quant(in_features, out_features, bias=True, offset=True, output_dtype=output_dtype)
    model = model.npu()
    model.weight.data = x2
    model.scale.data = scale
    model.offset.data = offset
    if output_dtype != torch.bfloat16:
        # Include the `npu_trans_quant_param` functionality and enable high-bandwidth x2 data layout for Atlas inference products
        tng.experimental.inference.use_internal_format_weight(model)
    model.bias.data = bias
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1)
    ```
