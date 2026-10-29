# torch_npu.contrib.module.LinearQuant

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->
<!-- npu="310p" id3 -->
- <term>Atlas inference products</term>: Supported
<!-- end id3 -->

## Function

LinearQuant is a wrapper class for the torch_npu.npu_quant_matmul API. It performs the matrix multiplication computation of the A8W8 and A4W4 quantized operators.

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
- **`pertoken_scale`** (`bool`): Optional. Specifies whether to include `pertoken_scale` parameters in the computation. If set to `False`, `pertoken_scale` is excluded from the quantized matrix multiplication computation. Atlas inference products: Currently, this parameter is not supported.
    <!-- npu="310p" id4 -->
    - <term>Atlas inference products</term>: pertoken_scale is not supported currently.
    <!-- end id4 -->
- **`device`**: The default value is `None`. **Reserved parameter, currently not used.**
- **`dtype`**: The default value is `None`. **Reserved parameter, currently not used.**
- **`output_dtype`** (`ScalarType`): Data type of the output tensor. The default value is `None`, indicating that the data type of the output tensor is `torch.int8`.

    <!-- npu="310p" id5 -->
    - <term>Atlas inference products</term>: the inputs `torch.int8` and `torch.float16` are supported.
    <!-- end id5 -->
    <!-- npu="A3,910b" id6 -->
    - <term>Atlas A2 products</term>/<term>Atlas A3 products</term>: the inputs `torch.int8`, `torch.float16`, `torch.bfloat16`, and `torch.int32` are supported.
    <!-- end id6 -->

**Computation Inputs**

**x1** (`Tensor`): The data layout can be $ND$, and the shape is at least 2D and at most 6D.

<!-- npu="310p" id7 -->
- <term>Atlas inference products</term>: the data type can be `torch.int8`.
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A2 products</term>/<term>Atlas A3 products</term>: the data types can be `torch.int8` and `torch.int32`. `torch.int32` indicates that this API is used to perform `torch_npu.int4` type matrix multiplication computation. The `torch.int32` type carries `torch_npu.int4` data, and each `torch.int32` value stores eight `torch_npu.int4` values.
<!-- end id8 -->

## Variable Description

- weight (`Tensor`): the data type must be the same as that of `x1`. The data layout can be $ND$, and the shape must be in the range of 2D to 6D. When the data type is `torch.int32`, the shape must be 2D.

    <!-- npu="310p" id9 -->
    - <term>Atlas inference products</term>: the data type can be `torch.int8`. Call torchair.experimental.inference.use_internal_format_weight or torch_npu.npu_format_cast to implement the high-performance data layout of weight (batch, n, k).
    <!-- end id9 -->
    <!-- npu="A3,910b" id10 -->
    - <term>Atlas A2 products</term>/<term>Atlas A3 products</term>: the data types can be `torch.int8` and `torch.int32` (the same as `x1`, indicating the computation of `torch_npu.int4` data). Call torch_npu.npu_format_cast to implement the high-performance data layout of weight (batch, n, k). However, using this module is not recommended; torch_npu.npu_quant_matmul is recommended.
    <!-- end id10 -->

- scale (`Tensor`): scale of the quantization computation. The data layout can be $ND$, and the shape must be 1D (t,), where t=1 or n, and n is the same as the n of `weight`. To pass a scale of the `torch.int64` data type, call the torch_npu.npu_trans_quant_param API in advance to obtain a scale of the `torch.int64` data type.

    <!-- npu="310p" id11 -->
    - <term>Atlas inference products</term>: the data types can be `torch.float32` and `torch.int64`.
    <!-- end id11 -->
    <!-- npu="A3,910b" id12 -->
    - <term>Atlas A2 products</term>/<term>Atlas A3 products</term>: the data types can be `torch.float32`, `torch.int64`, and `torch.bfloat16`.
    <!-- end id12 -->

- offset (`Tensor`): offset of the quantization computation. Optional. The data type can be `torch.float32`, the data layout can be $ND$, and the shape must be 1D (t,), where t=1 or n, and n is the same as the n of `weight`.

- pertoken_scale (`Tensor`): Optional. Pertoken of the quantization computation. The data type can be `torch.float32`, the data layout can be $ND$, and the shape must be 1D (m,), where m is the same as the m of `x1`.
    <!-- npu="310p" id13 -->
    - <term>Atlas inference products</term>: pertoken_scale is not supported currently.
    <!-- end id13 -->
- bias (`Tensor`): Optional. Bias in the matrix multiplication. The data layout can be $ND$, and the shape can be 1D (n,) or 3D (batch, 1, n), where n is the same as the n of `weight`, and the batch value must be equal to the batch value derived from broadcasting x1 and weight. When the output is 2D, 4D, 5D, or 6D, the bias shape is 1D; when the output is 3D, the bias shape is 1D or 3D.

    <!-- npu="310p" id14 -->
    - <term>Atlas inference products</term>: the data type can be `torch.int32`.
    <!-- end id14 -->
    <!-- npu="A3,910b" id15 -->
    - <term>Atlas A2 products</term>/<term>Atlas A3 products</term>: The data type can be `torch.int32`, `torch.bfloat16`, `torch.float16`, or `torch.float32`.
    <!-- end id15 -->

- output_dtype (`ScalarType`): Optional. Data type of the output Tensor. The default value is None, indicating that the output Tensor data type is `torch.int8`.

    <!-- npu="310p" id16 -->
    - <term>Atlas inference products</term>: the inputs `torch.int8` and `torch.float16` are supported.
    <!-- end id16 -->
    <!-- npu="A3,910b" id17 -->
    - <term>Atlas A2 products</term>/<term>Atlas A3 products</term>: the inputs `torch.int8`, `torch.float16`, `torch.bfloat16`, and `torch.int32` are supported.
    <!-- end id17 -->

## Return Values

`Tensor`

Represents the computation result of the quantized matmul:

- If output_dtype is `torch.int8` or None, the output data type is `torch.int8`.
- If output_dtype is `torch.float16`, the output data type is `torch.float16`.
- If output_dtype is `torch.bfloat16`, the output data type is `torch.bfloat16`.
- If output_dtype is `torch.int32`, the output data type is `torch.int32`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- `x1`, `weight`, and `scale` must not be empty.
- The size of the last dimension of the shapes of `x1` and `weight` must not exceed 65535.
- **Additional constraints for `torch_npu.int4` type computation**:

    When the data types of `x1` and `weight` are both `torch.int32`, each `torch.int32` value stores eight `torch_npu.int4` values. For the input shape, the last dimension of the shape must be 1/8 of the last dimension of the shape when the data is of the original `torch_npu.int4` type. The last dimension of the `torch_npu.int4` data shape must be a multiple of 8. For example, to perform the (m, k) multiplied by (k, n) `torch_npu.int4` type matrix multiplication computation, the inputs must be of the `torch.int32` type with the shapes (m, k//8) and (n, k//8), where both k and n must be multiples of 8. `x1` can accept only data with the shape (m, k//8) and a contiguous data layout, and `weight` can accept only data with the shape (n, k//8) and a contiguous data layout.

    > [!NOTE]  
    > A contiguous data layout means that all adjacent values in the array are contiguous in memory addresses, including when lines are wrapped. If the value returned by Tensor.is_contiguous is true, the tensor data layout is contiguous.

- The data type combinations supported between input parameters or variables are as follows:

    <!-- npu="310p" id20 -->
    **Table 1** <term>Atlas inference products</term>

    <a name="zh-cn_topic_0000002021380113_table75025595916"></a>
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
    <tbody><tr id="zh-cn_topic_0000002021380113_row4503125913910"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p150320591294"><a name="zh-cn_topic_0000002021380113_p150320591294"></a><a name="zh-cn_topic_0000002021380113_p150320591294"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p1450395915920"><a name="zh-cn_topic_0000002021380113_p1450395915920"></a><a name="zh-cn_topic_0000002021380113_p1450395915920"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="13.991399139913991%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p115031259296"><a name="zh-cn_topic_0000002021380113_p115031259296"></a><a name="zh-cn_topic_0000002021380113_p115031259296"></a><code>torch.int64</code>/<code>torch.float32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.931293129312932%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p1050319597913"><a name="zh-cn_topic_0000002021380113_p1050319597913"></a><a name="zh-cn_topic_0000002021380113_p1050319597913"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.41154115411541%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p85035591799"><a name="zh-cn_topic_0000002021380113_p85035591799"></a><a name="zh-cn_topic_0000002021380113_p85035591799"></a><code>torch.int32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.001700170017003%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p750312591699"><a name="zh-cn_topic_0000002021380113_p750312591699"></a><a name="zh-cn_topic_0000002021380113_p750312591699"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.47174717471747%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p550365911917"><a name="zh-cn_topic_0000002021380113_p550365911917"></a><a name="zh-cn_topic_0000002021380113_p550365911917"></a><code>torch.float16</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row750310595920"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p1750315599914"><a name="zh-cn_topic_0000002021380113_p1750315599914"></a><a name="zh-cn_topic_0000002021380113_p1750315599914"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p1850317591393"><a name="zh-cn_topic_0000002021380113_p1850317591393"></a><a name="zh-cn_topic_0000002021380113_p1850317591393"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="13.991399139913991%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p95034598912"><a name="zh-cn_topic_0000002021380113_p95034598912"></a><a name="zh-cn_topic_0000002021380113_p95034598912"></a><code>torch.int64</code>/<code>torch.float32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.931293129312932%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p14503359395"><a name="zh-cn_topic_0000002021380113_p14503359395"></a><a name="zh-cn_topic_0000002021380113_p14503359395"></a><code>torch.float32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.41154115411541%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p25033591296"><a name="zh-cn_topic_0000002021380113_p25033591296"></a><a name="zh-cn_topic_0000002021380113_p25033591296"></a><code>torch.int32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.001700170017003%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p115035592090"><a name="zh-cn_topic_0000002021380113_p115035592090"></a><a name="zh-cn_topic_0000002021380113_p115035592090"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.47174717471747%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p75033595916"><a name="zh-cn_topic_0000002021380113_p75033595916"></a><a name="zh-cn_topic_0000002021380113_p75033595916"></a><code>torch.int8</code></p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row183492120464"><td class="cellrowborder" colspan="7" valign="top" headers="mcps1.2.8.1.1 mcps1.2.8.1.2 mcps1.2.8.1.3 mcps1.2.8.1.4 mcps1.2.8.1.5 mcps1.2.8.1.6 mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p16486127124614"><a name="en-us_topic_0000002021380113_p16486127124614"></a><a name="en-us_topic_0000002021380113_p16486127124614"></a>Note: None indicates that the input parameter or variable is <code>False</code></p>
    </td>
    </tr>
    </tbody>
    </table>

    <!-- end id20 -->
    <!-- npu="A3,910b" id21 -->
    **Table 2** <term>Atlas A2 products</term> and <term>Atlas A3 products</term>

    <a name="zh-cn_topic_0000002021380113_table2504155910917"></a>
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
    <tbody><tr id="zh-cn_topic_0000002021380113_row1850445912918"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p105042591492"><a name="zh-cn_topic_0000002021380113_p105042591492"></a><a name="zh-cn_topic_0000002021380113_p105042591492"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p1550412593911"><a name="zh-cn_topic_0000002021380113_p1550412593911"></a><a name="zh-cn_topic_0000002021380113_p1550412593911"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p35044591897"><a name="zh-cn_topic_0000002021380113_p35044591897"></a><a name="zh-cn_topic_0000002021380113_p35044591897"></a><code>torch.int64</code>/<code>torch.float32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p2231164184310"><a name="zh-cn_topic_0000002021380113_p2231164184310"></a><a name="zh-cn_topic_0000002021380113_p2231164184310"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p175042059793"><a name="zh-cn_topic_0000002021380113_p175042059793"></a><a name="zh-cn_topic_0000002021380113_p175042059793"></a><code>torch.int32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p15627147184415"><a name="zh-cn_topic_0000002021380113_p15627147184415"></a><a name="zh-cn_topic_0000002021380113_p15627147184415"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p1950435916910"><a name="zh-cn_topic_0000002021380113_p1950435916910"></a><a name="zh-cn_topic_0000002021380113_p1950435916910"></a><code>torch.float16</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row1650414599917"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p135041599910"><a name="zh-cn_topic_0000002021380113_p135041599910"></a><a name="zh-cn_topic_0000002021380113_p135041599910"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p950413591090"><a name="zh-cn_topic_0000002021380113_p950413591090"></a><a name="zh-cn_topic_0000002021380113_p950413591090"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p2504105916914"><a name="zh-cn_topic_0000002021380113_p2504105916914"></a><a name="zh-cn_topic_0000002021380113_p2504105916914"></a><code>torch.int64</code>/<code>torch.float32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p2050415914911"><a name="zh-cn_topic_0000002021380113_p2050415914911"></a><a name="zh-cn_topic_0000002021380113_p2050415914911"></a><code>torch.float32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p18504105920916"><a name="zh-cn_topic_0000002021380113_p18504105920916"></a><a name="zh-cn_topic_0000002021380113_p18504105920916"></a><code>torch.int32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p550512597915"><a name="zh-cn_topic_0000002021380113_p550512597915"></a><a name="zh-cn_topic_0000002021380113_p550512597915"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p1750585915914"><a name="zh-cn_topic_0000002021380113_p1750585915914"></a><a name="zh-cn_topic_0000002021380113_p1750585915914"></a><code>torch.int8</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row175059594912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p550515591592"><a name="zh-cn_topic_0000002021380113_p550515591592"></a><a name="zh-cn_topic_0000002021380113_p550515591592"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p115052059396"><a name="zh-cn_topic_0000002021380113_p115052059396"></a><a name="zh-cn_topic_0000002021380113_p115052059396"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p050514591297"><a name="zh-cn_topic_0000002021380113_p050514591297"></a><a name="zh-cn_topic_0000002021380113_p050514591297"></a><code>torch.float32</code>/<code>torch.bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p1750595915910"><a name="zh-cn_topic_0000002021380113_p1750595915910"></a><a name="zh-cn_topic_0000002021380113_p1750595915910"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p650595920913"><a name="zh-cn_topic_0000002021380113_p650595920913"></a><a name="zh-cn_topic_0000002021380113_p650595920913"></a><code>torch.int32</code>/<code>torch.bfloat16</code>/<code>torch.float32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p1050518593911"><a name="zh-cn_topic_0000002021380113_p1050518593911"></a><a name="zh-cn_topic_0000002021380113_p1050518593911"></a><code>torch.float32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p19505959695"><a name="zh-cn_topic_0000002021380113_p19505959695"></a><a name="zh-cn_topic_0000002021380113_p19505959695"></a><code>torch.bfloat16</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row950520592912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p2505175915915"><a name="zh-cn_topic_0000002021380113_p2505175915915"></a><a name="zh-cn_topic_0000002021380113_p2505175915915"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p15051859596"><a name="zh-cn_topic_0000002021380113_p15051859596"></a><a name="zh-cn_topic_0000002021380113_p15051859596"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p750515599915"><a name="zh-cn_topic_0000002021380113_p750515599915"></a><a name="zh-cn_topic_0000002021380113_p750515599915"></a><code>torch.float32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p85051595912"><a name="zh-cn_topic_0000002021380113_p85051595912"></a><a name="zh-cn_topic_0000002021380113_p85051595912"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p1150511596912"><a name="zh-cn_topic_0000002021380113_p1150511596912"></a><a name="zh-cn_topic_0000002021380113_p1150511596912"></a><code>torch.int32</code>/<code>torch.float16</code>/<code>torch.float32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p1505135911914"><a name="zh-cn_topic_0000002021380113_p1505135911914"></a><a name="zh-cn_topic_0000002021380113_p1505135911914"></a><code>torch.float32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p850516591099"><a name="zh-cn_topic_0000002021380113_p850516591099"></a><a name="zh-cn_topic_0000002021380113_p850516591099"></a><code>torch.float16</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row750514591599"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p105051659594"><a name="zh-cn_topic_0000002021380113_p105051659594"></a><a name="zh-cn_topic_0000002021380113_p105051659594"></a><code>torch.int32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p1550525916920"><a name="zh-cn_topic_0000002021380113_p1550525916920"></a><a name="zh-cn_topic_0000002021380113_p1550525916920"></a><code>torch.int32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p750512591997"><a name="zh-cn_topic_0000002021380113_p750512591997"></a><a name="zh-cn_topic_0000002021380113_p750512591997"></a><code>torch.int64</code>/<code>torch.float32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p250517591991"><a name="zh-cn_topic_0000002021380113_p250517591991"></a><a name="zh-cn_topic_0000002021380113_p250517591991"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p19505195914919"><a name="zh-cn_topic_0000002021380113_p19505195914919"></a><a name="zh-cn_topic_0000002021380113_p19505195914919"></a><code>torch.int32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p550510599919"><a name="zh-cn_topic_0000002021380113_p550510599919"></a><a name="zh-cn_topic_0000002021380113_p550510599919"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p55058595918"><a name="zh-cn_topic_0000002021380113_p55058595918"></a><a name="zh-cn_topic_0000002021380113_p55058595918"></a><code>torch.float16</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row550595915912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p6505175913912"><a name="zh-cn_topic_0000002021380113_p6505175913912"></a><a name="zh-cn_topic_0000002021380113_p6505175913912"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p55051559192"><a name="zh-cn_topic_0000002021380113_p55051559192"></a><a name="zh-cn_topic_0000002021380113_p55051559192"></a><code>torch.int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p75054591913"><a name="zh-cn_topic_0000002021380113_p75054591913"></a><a name="zh-cn_topic_0000002021380113_p75054591913"></a><code>torch.float32</code>/<code>torch.bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p45064591697"><a name="zh-cn_topic_0000002021380113_p45064591697"></a><a name="zh-cn_topic_0000002021380113_p45064591697"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p7506259691"><a name="zh-cn_topic_0000002021380113_p7506259691"></a><a name="zh-cn_topic_0000002021380113_p7506259691"></a><code>torch.int32</code>/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p115061359097"><a name="zh-cn_topic_0000002021380113_p115061359097"></a><a name="zh-cn_topic_0000002021380113_p115061359097"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p250625911912"><a name="zh-cn_topic_0000002021380113_p250625911912"></a><a name="zh-cn_topic_0000002021380113_p250625911912"></a><code>torch.int32</code></p>
    </td>
    </tr>
    <tr id="en-us_topic_0000002021380113_row8523104511"><td class="cellrowborder" colspan="7" valign="top" headers="mcps1.2.8.1.1 mcps1.2.8.1.2 mcps1.2.8.1.3 mcps1.2.8.1.4 mcps1.2.8.1.5 mcps1.2.8.1.6 mcps1.2.8.1.7 "><p id="en-us_topic_0000002021380113_p1899351015458"><a name="en-us_topic_0000002021380113_p1899351015458"></a><a name="en-us_topic_0000002021380113_p1899351015458"></a>Note: None indicates that the input parameter or variable is <code>False</code>.</p>
    </td>
    </tr>
    </tbody>
    </table>
    <!-- end id21 -->

## Example

- Single-operator call
    - `torch.int8` type input scenario. The example code is as follows:

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
        # The npu_trans_quant_param function is called inside the API
        output = model(x1)
        ```

    <!-- npu="A3,910b" id18 -->
    - `torch.int32` type input scenario. The example code is as follows. Only the following products are supported:
        - <term>Atlas A2 products</term>
        - <term>Atlas A3 products</term>

        ```python
        import torch
        import torch_npu
        import logging
        import os
        from torch_npu.contrib.module import LinearQuant
        # Use the int32 type to carry int4 data. The actual int4 shape is x1: (1, 512), x2: (128, 512)
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
    <!-- end id18 -->

- Graph mode call. The example code is as follows:

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
        # Enable the high-bandwidth x2 data layout function
        tng.experimental.inference.use_internal_format_weight(model)
    model.bias.data = bias
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1)
    ```
