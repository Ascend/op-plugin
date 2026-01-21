# torch_npu.contrib.module.LinearQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 推理系列产品</term>     |    √     |

## 功能说明

LinearQuant是对torch_npu.npu_quant_matmul接口的封装类，完成A8W8、A4W4量化算子的矩阵乘计算。

## 函数原型

```
torch_npu.contrib.module.LinearQuant(in_features, out_features, *, bias=True, offset=False, pertoken_scale=False, device=None, dtype=None, output_dtype=None)
```

## 参数说明

**计算参数**

- **in_features**（`int`）：matmul计算中k轴的值。
- **out_features**（`int`）：matmul计算中n轴的值。
- **bias**（`bool`）：代表是否需要bias计算参数。如果设置成False，则bias不会加入量化matmul的计算。
- **offset**（`bool`）：代表是否需要offset计算参数。如果设置成False，则offset不会加入量化matmul的计算。
- **pertoken_scale**（`bool`）：可选参数，代表是否需要pertoken_scale计算参数。如果设置成False，则pertoken_scale不会加入量化matmul的计算。<term>Atlas 推理系列产品</term>当前不支持pertoken_scale。
- **device**：默认值为None。**预留参数，暂未使用**。
- **dtype**：默认值为None。**预留参数，暂未使用**。
- **output_dtype**（`ScalarType`）：表示输出Tensor的数据类型。默认值为None，代表输出Tensor数据类型为`int8`。
    - <term>Atlas 推理系列产品</term>：支持输入`int8`、`float16`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持输入`int8`、`float16`、`bfloat16`、`int32`。

**计算输入**

**x1**（`Tensor`）：数据格式支持$ND$，shape最少是2维，最多是6维。

- <term>Atlas 推理系列产品</term>：数据类型支持`int8`。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`和`int32`，其中`int32`表示使用本接口进行`int4`类型矩阵乘计算，`int32`类型承载的是`int4`数据，每个`int32`数据存放8个`int4`数据。

## 变量说明

- weight（`Tensor`）：与`x1`的数据类型须保持一致。数据格式支持$ND$，shape需要在2-6维范围。当数据类型为`int32`时，shape必须为2维。
    - <term>Atlas 推理系列产品</term>：数据类型支持`int8`，需要调用torchair.experimental.inference.use_internal_format_weight或torch_npu.npu_format_cast完成weight（batch, n, k）高性能数据排布功能。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`和`int32`（同`x1`，表示`int4`的数据计算），需要调用torch_npu.npu_format_cast完成weight（batch, n, k）高性能数据排布功能，但不推荐使用该module方式，推荐torch_npu.npu_quant_matmul。

- scale（`Tensor`）：量化计算的scale。数据格式支持$ND$，shape需要是1维(t,)，t=1或n，其中n与`weight`的n一致。如需传入`int64`数据类型的scale，需要提前调用torch_npu.npu_trans_quant_param接口来获取`int64`数据类型的scale。
    - <term>Atlas 推理系列产品</term>：数据类型支持`float32`、`int64`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`int64`、`bfloat16`。

- offset（`Tensor`）：量化计算的offset。可选参数。数据类型支持`float32`，数据格式支持$ND$，shape需要是1维(t,)，t=1或n，其中n与`weight`的n一致。
- pertoken_scale（`Tensor`）：可选参数，量化计算的pertoken。数据类型支持`float32`，数据格式支持$ND$，shape需要是1维(m,)，其中m与`x1`的m一致。<term>Atlas 推理系列产品</term>当前不支持pertoken_scale。
- bias（`Tensor`）：可选参数。矩阵乘中的bias。数据格式支持$ND$，shape支持1维(n,)或3维(batch, 1, n)，n与`weight`的n一致，同时batch值需要等于x1，weight broadcast后推导出的batch值。当输出为2、4、5、6维情况下，bias shape为1维；当输出为3维情况下，bias shape为1维或3维。
    - <term>Atlas 推理系列产品</term>：数据类型支持`int32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int32`、`bfloat16`、`float16`、`float32`。

- output_dtype（`ScalarType`）：可选参数。表示输出Tensor的数据类型。默认值为None，代表输出Tensor数据类型为`int8`。
    - <term>Atlas 推理系列产品</term>：支持输入`int8`、`float16`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持输入`int8`、`float16`、`bfloat16`、`int32`。

## 返回值说明

`Tensor`

代表量化matmul的计算结果：

- 如果output_dtype为`int8`或者None，输出的数据类型为`int8`。
- 如果output_dtype为`float16`，输出的数据类型为`float16`。
- 如果output_dtype为`bfloat16`，输出的数据类型为`bfloat16`。
- 如果output_dtype为`int32`，输出的数据类型为`int32`。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- `x1`、`weight`、`scale`不能是空。
- `x1`与`weight`最后一维的shape大小不能超过65535。
- **int4**类型计算的额外约束：

    当`x1`、`weight`的数据类型均为`int32`，每个`int32`类型的数据存放8个`int4`数据。输入shape需要将数据原本`int4`类型时的最后一维shape缩小8倍。`int4`数据的最后一维shape应为8的倍数，例如：进行(m, k)乘(k, n)的`int4`类型矩阵乘计算时，需要输入`int32`类型，shape为(m, k//8)、(k, n//8)的数据，其中k与n都应是8的倍数。`x1`只能接受shape为(m, k//8)且数据排布连续的数据，`weight`只能接受shape为(n, k//8)且数据排布连续的数据。

    > [!NOTE]  
    > 数据排布连续是指数组中所有相邻的数，包括换行时内存地址连续，使用Tensor.is_contiguous返回值为true则表明tensor数据排布连续。

- 输入参数或变量间支持的数据类型组合情况如下：

    **表1** <term>Atlas 推理系列产品</term>

    <a name="zh-cn_topic_0000002021380113_table75025595916"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000002021380113_row13503185919911"><th class="cellrowborder" valign="top" width="11.32113211321132%" id="mcps1.2.8.1.1"><p id="zh-cn_topic_0000002021380113_p35036591098"><a name="zh-cn_topic_0000002021380113_p35036591098"></a><a name="zh-cn_topic_0000002021380113_p35036591098"></a>x1（入参）</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.871187118711871%" id="mcps1.2.8.1.2"><p id="zh-cn_topic_0000002021380113_p450320598916"><a name="zh-cn_topic_0000002021380113_p450320598916"></a><a name="zh-cn_topic_0000002021380113_p450320598916"></a>weight（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.991399139913991%" id="mcps1.2.8.1.3"><p id="zh-cn_topic_0000002021380113_p125032592912"><a name="zh-cn_topic_0000002021380113_p125032592912"></a><a name="zh-cn_topic_0000002021380113_p125032592912"></a>scale（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.931293129312932%" id="mcps1.2.8.1.4"><p id="zh-cn_topic_0000002021380113_p750375912919"><a name="zh-cn_topic_0000002021380113_p750375912919"></a><a name="zh-cn_topic_0000002021380113_p750375912919"></a>offset（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="15.41154115411541%" id="mcps1.2.8.1.5"><p id="zh-cn_topic_0000002021380113_p16503175913919"><a name="zh-cn_topic_0000002021380113_p16503175913919"></a><a name="zh-cn_topic_0000002021380113_p16503175913919"></a>bias（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.001700170017003%" id="mcps1.2.8.1.6"><p id="zh-cn_topic_0000002021380113_p1150395911911"><a name="zh-cn_topic_0000002021380113_p1150395911911"></a><a name="zh-cn_topic_0000002021380113_p1150395911911"></a>pertoken_scale（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.47174717471747%" id="mcps1.2.8.1.7"><p id="zh-cn_topic_0000002021380113_p15503135912911"><a name="zh-cn_topic_0000002021380113_p15503135912911"></a><a name="zh-cn_topic_0000002021380113_p15503135912911"></a>output_dtype（入参或变量）</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000002021380113_row4503125913910"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p150320591294"><a name="zh-cn_topic_0000002021380113_p150320591294"></a><a name="zh-cn_topic_0000002021380113_p150320591294"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p1450395915920"><a name="zh-cn_topic_0000002021380113_p1450395915920"></a><a name="zh-cn_topic_0000002021380113_p1450395915920"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.991399139913991%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p115031259296"><a name="zh-cn_topic_0000002021380113_p115031259296"></a><a name="zh-cn_topic_0000002021380113_p115031259296"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.931293129312932%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p1050319597913"><a name="zh-cn_topic_0000002021380113_p1050319597913"></a><a name="zh-cn_topic_0000002021380113_p1050319597913"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.41154115411541%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p85035591799"><a name="zh-cn_topic_0000002021380113_p85035591799"></a><a name="zh-cn_topic_0000002021380113_p85035591799"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.001700170017003%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p750312591699"><a name="zh-cn_topic_0000002021380113_p750312591699"></a><a name="zh-cn_topic_0000002021380113_p750312591699"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.47174717471747%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p550365911917"><a name="zh-cn_topic_0000002021380113_p550365911917"></a><a name="zh-cn_topic_0000002021380113_p550365911917"></a>float16</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row750310595920"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p1750315599914"><a name="zh-cn_topic_0000002021380113_p1750315599914"></a><a name="zh-cn_topic_0000002021380113_p1750315599914"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p1850317591393"><a name="zh-cn_topic_0000002021380113_p1850317591393"></a><a name="zh-cn_topic_0000002021380113_p1850317591393"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.991399139913991%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p95034598912"><a name="zh-cn_topic_0000002021380113_p95034598912"></a><a name="zh-cn_topic_0000002021380113_p95034598912"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.931293129312932%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p14503359395"><a name="zh-cn_topic_0000002021380113_p14503359395"></a><a name="zh-cn_topic_0000002021380113_p14503359395"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.41154115411541%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p25033591296"><a name="zh-cn_topic_0000002021380113_p25033591296"></a><a name="zh-cn_topic_0000002021380113_p25033591296"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.001700170017003%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p115035592090"><a name="zh-cn_topic_0000002021380113_p115035592090"></a><a name="zh-cn_topic_0000002021380113_p115035592090"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.47174717471747%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p75033595916"><a name="zh-cn_topic_0000002021380113_p75033595916"></a><a name="zh-cn_topic_0000002021380113_p75033595916"></a>int8</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row183492120464"><td class="cellrowborder" colspan="7" valign="top" headers="mcps1.2.8.1.1 mcps1.2.8.1.2 mcps1.2.8.1.3 mcps1.2.8.1.4 mcps1.2.8.1.5 mcps1.2.8.1.6 mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p16486127124614"><a name="zh-cn_topic_0000002021380113_p16486127124614"></a><a name="zh-cn_topic_0000002021380113_p16486127124614"></a>注：None表示传入参数或变量为False的场景。</p>
    </td>
    </tr>
    </tbody>
    </table>

    **表2** <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

    <a name="zh-cn_topic_0000002021380113_table2504155910917"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000002021380113_row35048591395"><th class="cellrowborder" valign="top" width="11.32113211321132%" id="mcps1.2.8.1.1"><p id="zh-cn_topic_0000002021380113_p185044592912"><a name="zh-cn_topic_0000002021380113_p185044592912"></a><a name="zh-cn_topic_0000002021380113_p185044592912"></a>x1（入参）</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.871187118711871%" id="mcps1.2.8.1.2"><p id="zh-cn_topic_0000002021380113_p1750415591899"><a name="zh-cn_topic_0000002021380113_p1750415591899"></a><a name="zh-cn_topic_0000002021380113_p1750415591899"></a>weight（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="14.161416141614161%" id="mcps1.2.8.1.3"><p id="zh-cn_topic_0000002021380113_p125041959693"><a name="zh-cn_topic_0000002021380113_p125041959693"></a><a name="zh-cn_topic_0000002021380113_p125041959693"></a>scale（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.921292129212922%" id="mcps1.2.8.1.4"><p id="zh-cn_topic_0000002021380113_p145043591797"><a name="zh-cn_topic_0000002021380113_p145043591797"></a><a name="zh-cn_topic_0000002021380113_p145043591797"></a>offset（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="15.251525152515253%" id="mcps1.2.8.1.5"><p id="zh-cn_topic_0000002021380113_p1250455915910"><a name="zh-cn_topic_0000002021380113_p1250455915910"></a><a name="zh-cn_topic_0000002021380113_p1250455915910"></a>bias（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.461746174617463%" id="mcps1.2.8.1.6"><p id="zh-cn_topic_0000002021380113_p125045596917"><a name="zh-cn_topic_0000002021380113_p125045596917"></a><a name="zh-cn_topic_0000002021380113_p125045596917"></a>pertoken_scale（变量）</p>
    </th>
    <th class="cellrowborder" valign="top" width="17.01170117011701%" id="mcps1.2.8.1.7"><p id="zh-cn_topic_0000002021380113_p6504959094"><a name="zh-cn_topic_0000002021380113_p6504959094"></a><a name="zh-cn_topic_0000002021380113_p6504959094"></a>output_dtype（入参或变量）</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000002021380113_row1850445912918"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p105042591492"><a name="zh-cn_topic_0000002021380113_p105042591492"></a><a name="zh-cn_topic_0000002021380113_p105042591492"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p1550412593911"><a name="zh-cn_topic_0000002021380113_p1550412593911"></a><a name="zh-cn_topic_0000002021380113_p1550412593911"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p35044591897"><a name="zh-cn_topic_0000002021380113_p35044591897"></a><a name="zh-cn_topic_0000002021380113_p35044591897"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p2231164184310"><a name="zh-cn_topic_0000002021380113_p2231164184310"></a><a name="zh-cn_topic_0000002021380113_p2231164184310"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p175042059793"><a name="zh-cn_topic_0000002021380113_p175042059793"></a><a name="zh-cn_topic_0000002021380113_p175042059793"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p15627147184415"><a name="zh-cn_topic_0000002021380113_p15627147184415"></a><a name="zh-cn_topic_0000002021380113_p15627147184415"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p1950435916910"><a name="zh-cn_topic_0000002021380113_p1950435916910"></a><a name="zh-cn_topic_0000002021380113_p1950435916910"></a>float16</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row1650414599917"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p135041599910"><a name="zh-cn_topic_0000002021380113_p135041599910"></a><a name="zh-cn_topic_0000002021380113_p135041599910"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p950413591090"><a name="zh-cn_topic_0000002021380113_p950413591090"></a><a name="zh-cn_topic_0000002021380113_p950413591090"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p2504105916914"><a name="zh-cn_topic_0000002021380113_p2504105916914"></a><a name="zh-cn_topic_0000002021380113_p2504105916914"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p2050415914911"><a name="zh-cn_topic_0000002021380113_p2050415914911"></a><a name="zh-cn_topic_0000002021380113_p2050415914911"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p18504105920916"><a name="zh-cn_topic_0000002021380113_p18504105920916"></a><a name="zh-cn_topic_0000002021380113_p18504105920916"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p550512597915"><a name="zh-cn_topic_0000002021380113_p550512597915"></a><a name="zh-cn_topic_0000002021380113_p550512597915"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p1750585915914"><a name="zh-cn_topic_0000002021380113_p1750585915914"></a><a name="zh-cn_topic_0000002021380113_p1750585915914"></a>int8</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row175059594912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p550515591592"><a name="zh-cn_topic_0000002021380113_p550515591592"></a><a name="zh-cn_topic_0000002021380113_p550515591592"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p115052059396"><a name="zh-cn_topic_0000002021380113_p115052059396"></a><a name="zh-cn_topic_0000002021380113_p115052059396"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p050514591297"><a name="zh-cn_topic_0000002021380113_p050514591297"></a><a name="zh-cn_topic_0000002021380113_p050514591297"></a>float32/bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p1750595915910"><a name="zh-cn_topic_0000002021380113_p1750595915910"></a><a name="zh-cn_topic_0000002021380113_p1750595915910"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p650595920913"><a name="zh-cn_topic_0000002021380113_p650595920913"></a><a name="zh-cn_topic_0000002021380113_p650595920913"></a>int32/bfloat16/float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p1050518593911"><a name="zh-cn_topic_0000002021380113_p1050518593911"></a><a name="zh-cn_topic_0000002021380113_p1050518593911"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p19505959695"><a name="zh-cn_topic_0000002021380113_p19505959695"></a><a name="zh-cn_topic_0000002021380113_p19505959695"></a>bfloat16</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row950520592912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p2505175915915"><a name="zh-cn_topic_0000002021380113_p2505175915915"></a><a name="zh-cn_topic_0000002021380113_p2505175915915"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p15051859596"><a name="zh-cn_topic_0000002021380113_p15051859596"></a><a name="zh-cn_topic_0000002021380113_p15051859596"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p750515599915"><a name="zh-cn_topic_0000002021380113_p750515599915"></a><a name="zh-cn_topic_0000002021380113_p750515599915"></a>float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p85051595912"><a name="zh-cn_topic_0000002021380113_p85051595912"></a><a name="zh-cn_topic_0000002021380113_p85051595912"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p1150511596912"><a name="zh-cn_topic_0000002021380113_p1150511596912"></a><a name="zh-cn_topic_0000002021380113_p1150511596912"></a>int32/float16/float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p1505135911914"><a name="zh-cn_topic_0000002021380113_p1505135911914"></a><a name="zh-cn_topic_0000002021380113_p1505135911914"></a>float32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p850516591099"><a name="zh-cn_topic_0000002021380113_p850516591099"></a><a name="zh-cn_topic_0000002021380113_p850516591099"></a>float16</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row750514591599"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p105051659594"><a name="zh-cn_topic_0000002021380113_p105051659594"></a><a name="zh-cn_topic_0000002021380113_p105051659594"></a>int32</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p1550525916920"><a name="zh-cn_topic_0000002021380113_p1550525916920"></a><a name="zh-cn_topic_0000002021380113_p1550525916920"></a>int32</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p750512591997"><a name="zh-cn_topic_0000002021380113_p750512591997"></a><a name="zh-cn_topic_0000002021380113_p750512591997"></a>int64/float32</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p250517591991"><a name="zh-cn_topic_0000002021380113_p250517591991"></a><a name="zh-cn_topic_0000002021380113_p250517591991"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p19505195914919"><a name="zh-cn_topic_0000002021380113_p19505195914919"></a><a name="zh-cn_topic_0000002021380113_p19505195914919"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p550510599919"><a name="zh-cn_topic_0000002021380113_p550510599919"></a><a name="zh-cn_topic_0000002021380113_p550510599919"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p55058595918"><a name="zh-cn_topic_0000002021380113_p55058595918"></a><a name="zh-cn_topic_0000002021380113_p55058595918"></a>float16</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row550595915912"><td class="cellrowborder" valign="top" width="11.32113211321132%" headers="mcps1.2.8.1.1 "><p id="zh-cn_topic_0000002021380113_p6505175913912"><a name="zh-cn_topic_0000002021380113_p6505175913912"></a><a name="zh-cn_topic_0000002021380113_p6505175913912"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="11.871187118711871%" headers="mcps1.2.8.1.2 "><p id="zh-cn_topic_0000002021380113_p55051559192"><a name="zh-cn_topic_0000002021380113_p55051559192"></a><a name="zh-cn_topic_0000002021380113_p55051559192"></a>int8</p>
    </td>
    <td class="cellrowborder" valign="top" width="14.161416141614161%" headers="mcps1.2.8.1.3 "><p id="zh-cn_topic_0000002021380113_p75054591913"><a name="zh-cn_topic_0000002021380113_p75054591913"></a><a name="zh-cn_topic_0000002021380113_p75054591913"></a>float32/bfloat16</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.921292129212922%" headers="mcps1.2.8.1.4 "><p id="zh-cn_topic_0000002021380113_p45064591697"><a name="zh-cn_topic_0000002021380113_p45064591697"></a><a name="zh-cn_topic_0000002021380113_p45064591697"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="15.251525152515253%" headers="mcps1.2.8.1.5 "><p id="zh-cn_topic_0000002021380113_p7506259691"><a name="zh-cn_topic_0000002021380113_p7506259691"></a><a name="zh-cn_topic_0000002021380113_p7506259691"></a>int32/None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.461746174617463%" headers="mcps1.2.8.1.6 "><p id="zh-cn_topic_0000002021380113_p115061359097"><a name="zh-cn_topic_0000002021380113_p115061359097"></a><a name="zh-cn_topic_0000002021380113_p115061359097"></a>None</p>
    </td>
    <td class="cellrowborder" valign="top" width="17.01170117011701%" headers="mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p250625911912"><a name="zh-cn_topic_0000002021380113_p250625911912"></a><a name="zh-cn_topic_0000002021380113_p250625911912"></a>int32</p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002021380113_row8523104511"><td class="cellrowborder" colspan="7" valign="top" headers="mcps1.2.8.1.1 mcps1.2.8.1.2 mcps1.2.8.1.3 mcps1.2.8.1.4 mcps1.2.8.1.5 mcps1.2.8.1.6 mcps1.2.8.1.7 "><p id="zh-cn_topic_0000002021380113_p1899351015458"><a name="zh-cn_topic_0000002021380113_p1899351015458"></a><a name="zh-cn_topic_0000002021380113_p1899351015458"></a>注：None表示传入参数或变量为False的场景。</p>
    </td>
    </tr>
    </tbody>
    </table>

## 调用示例

- 单算子模式调用
    - int8类型输入场景，示例代码如下：

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
        # 接口内部调用npu_trans_quant_param功能
        output = model(x1)
        ```

    - int32类型输入场景，示例代码如下，仅支持如下产品：

        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> 
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

        ```python
        import torch
        import torch_npu
        import logging
        import os
        from torch_npu.contrib.module import LinearQuant
        # 用int32类型承载int4数据，实际int4 shape为x1：(1, 512) x2： (128, 512)
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

- 图模式调用，示例代码如下，仅支持如下产品：

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
        # 使能高带宽x2的数据排布功能
        tng.experimental.inference.use_internal_format_weight(model)
    model.bias.data = bias
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    output = model(x1)
    ```

