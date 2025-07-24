# torch_npu.npu_mm_all_reduce_base

## 功能说明

TP切分场景下，实现mm和all_reduce的融合，融合算子内部实现计算和通信流水并行。

>**说明：**<br>
>使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如BUS ERROR等。

## 函数原型

```
torch_npu.npu_mm_all_reduce_base(x1, x2, hcom, *, reduce_op='sum', bias=None, antiquant_scale=None, antiquant_offset=None, x3=None, dequant_scale=None, pertoken_scale=None, comm_quant_scale_1=None, comm_quant_scale_2=None, comm_turn=0, antiquant_group_size=0) -> Tensor
```

## 参数说明

- **x1** (`Tensor`)：数据类型支持`int8`、`float16`、`bfloat16`。数据格式支持$ND$，输入shape支持2维或者3维。
- **x2** (`Tensor`)：数据类型支持`float16`、`int8`、`bfloat16`，数据格式支持$NZ$（昇腾亲和排布格式）、$ND$。非量化场景，数据类型需要和`x1`保持一致，输入shape维度第0维和`x1`的最后一维保持一致。
- **hcom** (`str`)：通信域handle名，通过`get_hccl_comm_name`接口获取。
- <strong>*</strong>：代表其之前的变量是位置相关，按照顺序输入，必选；之后的变量是键值对赋值的，位置无关，可选（不输入会使用默认值）。
- **reduce_op** (`str`)：reduce操作类型，**当前版本仅支持**`sum`，默认值：`sum`。
- **bias** (`Tensor`)：可选输入，数据类型支持`int32`、`float16`、`bfloat16`，数据格式支持$ND$。`bias`当前仅支持一维，且维度大小与`output/x2`的最后一维大小相同。
- **antiquant_scale** (`Tensor`)：可选输入，伪量化场景对`x2`进行去量化的系数，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$。伪量化场景数据类型需要和`x1`保持一致。
    - per-tensor场景：shape为$[1]$。
    - per-channel场景：shape为$[1,n]$或者$[n]$，$n$为`x2`最后一维的大小。
    - per-group场景：shape为$[ceil(k, antiquant\_group\_size), n]$。其中$k$为`x2`第一维的大小，$n$为`x2`最后一维的大小，$antiquant\_group\_size$为伪量化场景对输入`x2`进行反量化计算的groupSize输入。

        >**说明：**<br>
        >$ceil(k, antiquant\_group\_size)$的计算逻辑为：$(k + antiquant\_group\_size - 1) / antiquant\_group\_size$，并对计算结果取整数部分。

- **antiquant_offset** (`Tensor`)：可选输入，伪量化场景对`x2`进行去量化的系数，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$。数据类型、shape需要和`antiquant_scale`保持一致。
- **x3** (`Tensor`)：可选输入，matmul计算后的偏移。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float16`、`bfloat16`，数据格式支持$ND$。数据类型、shape需要和输出`output`保持一致。

- **dequant_scale** (`Tensor`)：可选输入，matmul计算后的去量化系数。数据类型支持`int64`、`uint64`、`bfloat16`、`float32`；数据格式支持$ND$。
    - per-tensor场景：shape为$[1]$。
    - per-channel场景：shape为$[n]/[1,n]$，$n$为`x2`最后一维的大小。

- **pertoken_scale** (`Tensor`)：可选输入，matmul计算后的per-token去量化系数。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持`float32`。当`x1`为$[m,k]$时`pertoken_scale` shape为$[m]$；当`x1`为$[b, s, k]$时`pertoken_scale` shape为$[b*s]$。

- **comm_quant_scale_1** (`Tensor`)：可选输入，alltoall通信前后的量化、去量化系数。支持`float16`、`bfloat16`，支持$ND$格式。`x2`为$[k, n]$时shape为$[1, n]$或$[n]$，用户需保证每张卡上数据保持一致且正确。
- **comm_quant_scale_2** (`Tensor`)：可选输入，allgather通信前后的量化、去量化系数。支持`float16`、`bfloat16`，支持$ND$格式。`x2`为$[k, n]$时shape为$[1, n]$或$[n]$，用户需保证每张卡上数据保持一致且正确。
- **comm_turn** (`int`)：表示rank间通信切分粒度，默认值：`0`，表示默认的切分方式。当前版本仅支持输入`0`。
- **antiquant_group_size** (`int`)：表示伪量化per-group算法模式下，对输入`x2`进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在$k$轴方向的大小。当伪量化算法模式不为per-group时传入`0`；当伪量化算法模式为per-group时传入值的范围为`[32, min(k-1, INT_MAX)]`且值要求是32的倍数，其中$k$为`x2`第一维的大小。默认值`0`，为`0`则表示非per-group场景。

## 返回值
`Tensor`

数据类型非量化场景以及伪量化场景与`x1`保持一致，全量化场景输出数据类型为`float16`或`bfloat16`。shape第0维度和`x1`的0维保持一致，若`x1`为2维，shape第1维度和`x2`的1维保持一致，若`x1`为3维，shape第1维度和`x1`的1维保持一致，shape第2维度和`x2`的1维保持一致。

## 约束说明

- 该接口支持推理场景下使用。
- 增量场景不使能该融合算子，全量场景使能该融合算子。
- 该接口支持图模式（PyTorch 2.1版本）。
- 输入`x1`可为2维或者3维、`x2`必须是2维，分别为$(b, s, k)/(m, k)$, $(k, n)$，$k$轴满足mm算子入参要求，$k$轴相等。`bias`当前仅支持一维，且维度大小与`output`的最后一维大小相同。`x3`的shape与`output`的shape相同。
- `x1`不支持输入转置后的tensor，`x2`转置后输入，需要满足shape的第一维大小与`x1`的最后一维相同，满足matmul的计算条件。
- `antiquant_group_size`中$k$值的范围与matmul一致，为`[1,65535]`，`INT_MAX`大于$(k-1)$。
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    - 数据类型支持`bfloat16`。
    - `x1`、`x2`不支持为空tensor。
    - 支持1、2、4、8卡，并且仅支持hccs链路all mesh组网。
    - 非量化场景下，$m、k、n$的取值范围均为`[1, 2147483647]`。
    - `comm_quant_scale_1`，`comm_quant_scale_2`的shape应保持一致，dtype与输出的dtype保持一致，且只在全量化场景支持。

- 全量化场景：$m$取值范围均为`[1, 2147483647]`，`x1`、`x2`的最后一维范围为`[1, 65535]`，即$k$的取值范围为`[1, 65535]`、仅当`x2`(`shape=[n,k]`)为转置时$n$可以大于65535。
- 伪量化场景：$m$取值范围均为`[1, 2147483647]`，$k、n$的取值范围为`[1, 65535]`。
- Atlas A2 训练系列产品：一个模型中的通算融合算子（AllGatherMatmul、MatmulReduceScatter、MatmulAllReduce），仅支持相同通信域。
- 在长序列场景，随着$b/s$或者$m$的增大，可能出现内存不足或者计算超时。
- 不同场景下数据类型支持情况：

    **表1** 非量化场景

    <a name="zh-cn_topic_0000001721582972_table487481610422"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001721582972_row287481618427"><th class="cellrowborder" valign="top" width="13.450000000000001%" id="mcps1.2.10.1.1"><p id="zh-cn_topic_0000001721582972_p1117215415458"><a name="zh-cn_topic_0000001721582972_p1117215415458"></a><a name="zh-cn_topic_0000001721582972_p1117215415458"></a>产品型号</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.39%" id="mcps1.2.10.1.2"><p id="zh-cn_topic_0000001721582972_p087501664213"><a name="zh-cn_topic_0000001721582972_p087501664213"></a><a name="zh-cn_topic_0000001721582972_p087501664213"></a>x1</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.41%" id="mcps1.2.10.1.3"><p id="zh-cn_topic_0000001721582972_p687571612423"><a name="zh-cn_topic_0000001721582972_p687571612423"></a><a name="zh-cn_topic_0000001721582972_p687571612423"></a>x2</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.23%" id="mcps1.2.10.1.4"><p id="zh-cn_topic_0000001721582972_p6875101674217"><a name="zh-cn_topic_0000001721582972_p6875101674217"></a><a name="zh-cn_topic_0000001721582972_p6875101674217"></a>bias</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.79%" id="mcps1.2.10.1.5"><p id="zh-cn_topic_0000001721582972_p9875101612424"><a name="zh-cn_topic_0000001721582972_p9875101612424"></a><a name="zh-cn_topic_0000001721582972_p9875101612424"></a>x3</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.77%" id="mcps1.2.10.1.6"><p id="zh-cn_topic_0000001721582972_p13875171674217"><a name="zh-cn_topic_0000001721582972_p13875171674217"></a><a name="zh-cn_topic_0000001721582972_p13875171674217"></a>output（输出）</p>
    </th>
    <th class="cellrowborder" valign="top" width="11%" id="mcps1.2.10.1.7"><p id="zh-cn_topic_0000001721582972_p178759160422"><a name="zh-cn_topic_0000001721582972_p178759160422"></a><a name="zh-cn_topic_0000001721582972_p178759160422"></a>antiquant_scale</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.37%" id="mcps1.2.10.1.8"><p id="zh-cn_topic_0000001721582972_p188751168423"><a name="zh-cn_topic_0000001721582972_p188751168423"></a><a name="zh-cn_topic_0000001721582972_p188751168423"></a>antiquant_offset</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.59%" id="mcps1.2.10.1.9"><p id="zh-cn_topic_0000001721582972_p118751216104210"><a name="zh-cn_topic_0000001721582972_p118751216104210"></a><a name="zh-cn_topic_0000001721582972_p118751216104210"></a>dequant_scale</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001721582972_row1587511620421"><td class="cellrowborder" valign="top" width="13.450000000000001%" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000001721582972_p171731544459"><a name="zh-cn_topic_0000001721582972_p171731544459"></a><a name="zh-cn_topic_0000001721582972_p171731544459"></a><span id="zh-cn_topic_0000001721582972_ph7270121464516"><a name="zh-cn_topic_0000001721582972_ph7270121464516"></a><a name="zh-cn_topic_0000001721582972_ph7270121464516"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_3"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_3"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000001721582972_p18875171624214"><a name="zh-cn_topic_0000001721582972_p18875171624214"></a><a name="zh-cn_topic_0000001721582972_p18875171624214"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.41%" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000001721582972_p18751116164210"><a name="zh-cn_topic_0000001721582972_p18751116164210"></a><a name="zh-cn_topic_0000001721582972_p18751116164210"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000001721582972_p1187521614421"><a name="zh-cn_topic_0000001721582972_p1187521614421"></a><a name="zh-cn_topic_0000001721582972_p1187521614421"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.79%" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000001721582972_p118758167427"><a name="zh-cn_topic_0000001721582972_p118758167427"></a><a name="zh-cn_topic_0000001721582972_p118758167427"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.77%" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000001721582972_p1687531610424"><a name="zh-cn_topic_0000001721582972_p1687531610424"></a><a name="zh-cn_topic_0000001721582972_p1687531610424"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11%" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000001721582972_p18875616114215"><a name="zh-cn_topic_0000001721582972_p18875616114215"></a><a name="zh-cn_topic_0000001721582972_p18875616114215"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.37%" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000001721582972_p158711699440"><a name="zh-cn_topic_0000001721582972_p158711699440"></a><a name="zh-cn_topic_0000001721582972_p158711699440"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.59%" headers="mcps1.2.10.1.9 "><p id="zh-cn_topic_0000001721582972_p1641110107441"><a name="zh-cn_topic_0000001721582972_p1641110107441"></a><a name="zh-cn_topic_0000001721582972_p1641110107441"></a><code>None</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001721582972_row1887561616429"><td class="cellrowborder" valign="top" width="13.450000000000001%" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000001721582972_p1917317410451"><a name="zh-cn_topic_0000001721582972_p1917317410451"></a><a name="zh-cn_topic_0000001721582972_p1917317410451"></a><span id="zh-cn_topic_0000001721582972_ph650661584516"><a name="zh-cn_topic_0000001721582972_ph650661584516"></a><a name="zh-cn_topic_0000001721582972_ph650661584516"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_4"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_4"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.39%" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000001721582972_p158756160422"><a name="zh-cn_topic_0000001721582972_p158756160422"></a><a name="zh-cn_topic_0000001721582972_p158756160422"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.41%" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000001721582972_p4875111694213"><a name="zh-cn_topic_0000001721582972_p4875111694213"></a><a name="zh-cn_topic_0000001721582972_p4875111694213"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000001721582972_p1587541618424"><a name="zh-cn_topic_0000001721582972_p1587541618424"></a><a name="zh-cn_topic_0000001721582972_p1587541618424"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.79%" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000001721582972_p687621634212"><a name="zh-cn_topic_0000001721582972_p687621634212"></a><a name="zh-cn_topic_0000001721582972_p687621634212"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.77%" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000001721582972_p1087671674216"><a name="zh-cn_topic_0000001721582972_p1087671674216"></a><a name="zh-cn_topic_0000001721582972_p1087671674216"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11%" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000001721582972_p028911224418"><a name="zh-cn_topic_0000001721582972_p028911224418"></a><a name="zh-cn_topic_0000001721582972_p028911224418"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.37%" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000001721582972_p126912112445"><a name="zh-cn_topic_0000001721582972_p126912112445"></a><a name="zh-cn_topic_0000001721582972_p126912112445"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.59%" headers="mcps1.2.10.1.9 "><p id="zh-cn_topic_0000001721582972_p11153131116449"><a name="zh-cn_topic_0000001721582972_p11153131116449"></a><a name="zh-cn_topic_0000001721582972_p11153131116449"></a><code>None</code></p>
    </td>
    </tr>
    </tbody>
    </table>

    **表2** 伪量化场景

    <a name="zh-cn_topic_0000001721582972_table931191613475"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001721582972_row183111684715"><th class="cellrowborder" valign="top" width="13.36%" id="mcps1.2.10.1.1"><p id="zh-cn_topic_0000001721582972_p431161634715"><a name="zh-cn_topic_0000001721582972_p431161634715"></a><a name="zh-cn_topic_0000001721582972_p431161634715"></a>产品型号</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.280000000000001%" id="mcps1.2.10.1.2"><p id="zh-cn_topic_0000001721582972_p1431416104714"><a name="zh-cn_topic_0000001721582972_p1431416104714"></a><a name="zh-cn_topic_0000001721582972_p1431416104714"></a>x1</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.43%" id="mcps1.2.10.1.3"><p id="zh-cn_topic_0000001721582972_p231416124714"><a name="zh-cn_topic_0000001721582972_p231416124714"></a><a name="zh-cn_topic_0000001721582972_p231416124714"></a>x2</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.4%" id="mcps1.2.10.1.4"><p id="zh-cn_topic_0000001721582972_p831131694712"><a name="zh-cn_topic_0000001721582972_p831131694712"></a><a name="zh-cn_topic_0000001721582972_p831131694712"></a>bias</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.51%" id="mcps1.2.10.1.5"><p id="zh-cn_topic_0000001721582972_p143114164470"><a name="zh-cn_topic_0000001721582972_p143114164470"></a><a name="zh-cn_topic_0000001721582972_p143114164470"></a>x3</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.06%" id="mcps1.2.10.1.6"><p id="zh-cn_topic_0000001721582972_p1531111604713"><a name="zh-cn_topic_0000001721582972_p1531111604713"></a><a name="zh-cn_topic_0000001721582972_p1531111604713"></a>output（输出）</p>
    </th>
    <th class="cellrowborder" valign="top" width="11%" id="mcps1.2.10.1.7"><p id="zh-cn_topic_0000001721582972_p103117161470"><a name="zh-cn_topic_0000001721582972_p103117161470"></a><a name="zh-cn_topic_0000001721582972_p103117161470"></a>antiquant_scale</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.27%" id="mcps1.2.10.1.8"><p id="zh-cn_topic_0000001721582972_p123171624712"><a name="zh-cn_topic_0000001721582972_p123171624712"></a><a name="zh-cn_topic_0000001721582972_p123171624712"></a>antiquant_offset</p>
    </th>
    <th class="cellrowborder" valign="top" width="11.690000000000001%" id="mcps1.2.10.1.9"><p id="zh-cn_topic_0000001721582972_p6314166474"><a name="zh-cn_topic_0000001721582972_p6314166474"></a><a name="zh-cn_topic_0000001721582972_p6314166474"></a>dequant_scale</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001721582972_row5311716184711"><td class="cellrowborder" valign="top" width="13.36%" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000001721582972_p153113164472"><a name="zh-cn_topic_0000001721582972_p153113164472"></a><a name="zh-cn_topic_0000001721582972_p153113164472"></a><span id="zh-cn_topic_0000001721582972_ph113215165479"><a name="zh-cn_topic_0000001721582972_ph113215165479"></a><a name="zh-cn_topic_0000001721582972_ph113215165479"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_5"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_5"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.280000000000001%" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000001721582972_p83241610472"><a name="zh-cn_topic_0000001721582972_p83241610472"></a><a name="zh-cn_topic_0000001721582972_p83241610472"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.43%" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000001721582972_p232121615476"><a name="zh-cn_topic_0000001721582972_p232121615476"></a><a name="zh-cn_topic_0000001721582972_p232121615476"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000001721582972_p3329162478"><a name="zh-cn_topic_0000001721582972_p3329162478"></a><a name="zh-cn_topic_0000001721582972_p3329162478"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.51%" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000001721582972_p15328161479"><a name="zh-cn_topic_0000001721582972_p15328161479"></a><a name="zh-cn_topic_0000001721582972_p15328161479"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000001721582972_p113261611474"><a name="zh-cn_topic_0000001721582972_p113261611474"></a><a name="zh-cn_topic_0000001721582972_p113261611474"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11%" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000001721582972_p16738125012479"><a name="zh-cn_topic_0000001721582972_p16738125012479"></a><a name="zh-cn_topic_0000001721582972_p16738125012479"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000001721582972_p133251613477"><a name="zh-cn_topic_0000001721582972_p133251613477"></a><a name="zh-cn_topic_0000001721582972_p133251613477"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.10.1.9 "><p id="zh-cn_topic_0000001721582972_p183231615478"><a name="zh-cn_topic_0000001721582972_p183231615478"></a><a name="zh-cn_topic_0000001721582972_p183231615478"></a><code>None</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001721582972_row1232416104714"><td class="cellrowborder" valign="top" width="13.36%" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000001721582972_p12321516194716"><a name="zh-cn_topic_0000001721582972_p12321516194716"></a><a name="zh-cn_topic_0000001721582972_p12321516194716"></a><span id="zh-cn_topic_0000001721582972_ph632201611474"><a name="zh-cn_topic_0000001721582972_ph632201611474"></a><a name="zh-cn_topic_0000001721582972_ph632201611474"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_6"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_6"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.280000000000001%" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000001721582972_p03281614717"><a name="zh-cn_topic_0000001721582972_p03281614717"></a><a name="zh-cn_topic_0000001721582972_p03281614717"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.43%" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000001721582972_p114918524815"><a name="zh-cn_topic_0000001721582972_p114918524815"></a><a name="zh-cn_topic_0000001721582972_p114918524815"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.4%" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000001721582972_p163215165478"><a name="zh-cn_topic_0000001721582972_p163215165478"></a><a name="zh-cn_topic_0000001721582972_p163215165478"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.51%" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000001721582972_p2032131617475"><a name="zh-cn_topic_0000001721582972_p2032131617475"></a><a name="zh-cn_topic_0000001721582972_p2032131617475"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.06%" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000001721582972_p63216169474"><a name="zh-cn_topic_0000001721582972_p63216169474"></a><a name="zh-cn_topic_0000001721582972_p63216169474"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11%" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000001721582972_p1332141613478"><a name="zh-cn_topic_0000001721582972_p1332141613478"></a><a name="zh-cn_topic_0000001721582972_p1332141613478"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.27%" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000001721582972_p111021821144811"><a name="zh-cn_topic_0000001721582972_p111021821144811"></a><a name="zh-cn_topic_0000001721582972_p111021821144811"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="11.690000000000001%" headers="mcps1.2.10.1.9 "><p id="zh-cn_topic_0000001721582972_p6325161475"><a name="zh-cn_topic_0000001721582972_p6325161475"></a><a name="zh-cn_topic_0000001721582972_p6325161475"></a><code>None</code></p>
    </td>
    </tr>
    </tbody>
    </table>

    **表3** 全量化场景

    <a name="zh-cn_topic_0000001721582972_table18830115614816"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000001721582972_row148319564487"><th class="cellrowborder" valign="top" width="13.528647135286473%" id="mcps1.2.11.1.1"><p id="zh-cn_topic_0000001721582972_p1831185615482"><a name="zh-cn_topic_0000001721582972_p1831185615482"></a><a name="zh-cn_topic_0000001721582972_p1831185615482"></a>产品型号</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.439156084391561%" id="mcps1.2.11.1.2"><p id="zh-cn_topic_0000001721582972_p19831155618488"><a name="zh-cn_topic_0000001721582972_p19831155618488"></a><a name="zh-cn_topic_0000001721582972_p19831155618488"></a>x1</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.76912308769123%" id="mcps1.2.11.1.3"><p id="zh-cn_topic_0000001721582972_p148311256174817"><a name="zh-cn_topic_0000001721582972_p148311256174817"></a><a name="zh-cn_topic_0000001721582972_p148311256174817"></a>x2</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.529047095290471%" id="mcps1.2.11.1.4"><p id="zh-cn_topic_0000001721582972_p0831125610485"><a name="zh-cn_topic_0000001721582972_p0831125610485"></a><a name="zh-cn_topic_0000001721582972_p0831125610485"></a>bias</p>
    </th>
    <th class="cellrowborder" valign="top" width="8.899110088991101%" id="mcps1.2.11.1.5"><p id="zh-cn_topic_0000001721582972_p1583145634814"><a name="zh-cn_topic_0000001721582972_p1583145634814"></a><a name="zh-cn_topic_0000001721582972_p1583145634814"></a>x3</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.90900909909009%" id="mcps1.2.11.1.6"><p id="zh-cn_topic_0000001721582972_p13831135614486"><a name="zh-cn_topic_0000001721582972_p13831135614486"></a><a name="zh-cn_topic_0000001721582972_p13831135614486"></a>output（输出）</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.338966103389662%" id="mcps1.2.11.1.7"><p id="zh-cn_topic_0000001721582972_p583112564482"><a name="zh-cn_topic_0000001721582972_p583112564482"></a><a name="zh-cn_topic_0000001721582972_p583112564482"></a>antiquant_scale</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.64893510648935%" id="mcps1.2.11.1.8"><p id="zh-cn_topic_0000001721582972_p1831195624814"><a name="zh-cn_topic_0000001721582972_p1831195624814"></a><a name="zh-cn_topic_0000001721582972_p1831195624814"></a>antiquant_offset</p>
    </th>
    <th class="cellrowborder" valign="top" width="10.068993100689932%" id="mcps1.2.11.1.9"><p id="zh-cn_topic_0000001721582972_p11831185612485"><a name="zh-cn_topic_0000001721582972_p11831185612485"></a><a name="zh-cn_topic_0000001721582972_p11831185612485"></a>dequant_scale</p>
    </th>
    <th class="cellrowborder" valign="top" width="9.86901309869013%" id="mcps1.2.11.1.10"><p id="zh-cn_topic_0000001721582972_p43549175020"><a name="zh-cn_topic_0000001721582972_p43549175020"></a><a name="zh-cn_topic_0000001721582972_p43549175020"></a>pertoken_scale</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000001721582972_row1183135624817"><td class="cellrowborder" valign="top" width="13.528647135286473%" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000001721582972_p38312056194810"><a name="zh-cn_topic_0000001721582972_p38312056194810"></a><a name="zh-cn_topic_0000001721582972_p38312056194810"></a><span id="zh-cn_topic_0000001721582972_ph1683165618489"><a name="zh-cn_topic_0000001721582972_ph1683165618489"></a><a name="zh-cn_topic_0000001721582972_ph1683165618489"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_7"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_7"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.439156084391561%" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000001721582972_p17831115614818"><a name="zh-cn_topic_0000001721582972_p17831115614818"></a><a name="zh-cn_topic_0000001721582972_p17831115614818"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.76912308769123%" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000001721582972_p168317567489"><a name="zh-cn_topic_0000001721582972_p168317567489"></a><a name="zh-cn_topic_0000001721582972_p168317567489"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.529047095290471%" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000001721582972_p11831175634812"><a name="zh-cn_topic_0000001721582972_p11831175634812"></a><a name="zh-cn_topic_0000001721582972_p11831175634812"></a><code>int32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.899110088991101%" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000001721582972_p583185614483"><a name="zh-cn_topic_0000001721582972_p583185614483"></a><a name="zh-cn_topic_0000001721582972_p583185614483"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.90900909909009%" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000001721582972_p10831856134814"><a name="zh-cn_topic_0000001721582972_p10831856134814"></a><a name="zh-cn_topic_0000001721582972_p10831856134814"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.338966103389662%" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000001721582972_p1883118567482"><a name="zh-cn_topic_0000001721582972_p1883118567482"></a><a name="zh-cn_topic_0000001721582972_p1883118567482"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.64893510648935%" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000001721582972_p530094404910"><a name="zh-cn_topic_0000001721582972_p530094404910"></a><a name="zh-cn_topic_0000001721582972_p530094404910"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.068993100689932%" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000001721582972_p458310539498"><a name="zh-cn_topic_0000001721582972_p458310539498"></a><a name="zh-cn_topic_0000001721582972_p458310539498"></a><code>uint64</code>或<code>int64</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.86901309869013%" headers="mcps1.2.11.1.10 "><p id="zh-cn_topic_0000001721582972_p33541210508"><a name="zh-cn_topic_0000001721582972_p33541210508"></a><a name="zh-cn_topic_0000001721582972_p33541210508"></a><code>None</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001721582972_row1483115618483"><td class="cellrowborder" valign="top" width="13.528647135286473%" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000001721582972_p14831115612481"><a name="zh-cn_topic_0000001721582972_p14831115612481"></a><a name="zh-cn_topic_0000001721582972_p14831115612481"></a><span id="zh-cn_topic_0000001721582972_ph583195654817"><a name="zh-cn_topic_0000001721582972_ph583195654817"></a><a name="zh-cn_topic_0000001721582972_ph583195654817"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_8"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_8"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.439156084391561%" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000001721582972_p1299345055219"><a name="zh-cn_topic_0000001721582972_p1299345055219"></a><a name="zh-cn_topic_0000001721582972_p1299345055219"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.76912308769123%" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000001721582972_p1899315005215"><a name="zh-cn_topic_0000001721582972_p1899315005215"></a><a name="zh-cn_topic_0000001721582972_p1899315005215"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.529047095290471%" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000001721582972_p16993450125215"><a name="zh-cn_topic_0000001721582972_p16993450125215"></a><a name="zh-cn_topic_0000001721582972_p16993450125215"></a><code>int32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.899110088991101%" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000001721582972_p2075012525314"><a name="zh-cn_topic_0000001721582972_p2075012525314"></a><a name="zh-cn_topic_0000001721582972_p2075012525314"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.90900909909009%" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000001721582972_p2993165014524"><a name="zh-cn_topic_0000001721582972_p2993165014524"></a><a name="zh-cn_topic_0000001721582972_p2993165014524"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.338966103389662%" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000001721582972_p1399335011520"><a name="zh-cn_topic_0000001721582972_p1399335011520"></a><a name="zh-cn_topic_0000001721582972_p1399335011520"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.64893510648935%" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000001721582972_p09931750165215"><a name="zh-cn_topic_0000001721582972_p09931750165215"></a><a name="zh-cn_topic_0000001721582972_p09931750165215"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.068993100689932%" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000001721582972_p7993250135220"><a name="zh-cn_topic_0000001721582972_p7993250135220"></a><a name="zh-cn_topic_0000001721582972_p7993250135220"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.86901309869013%" headers="mcps1.2.11.1.10 "><p id="zh-cn_topic_0000001721582972_p1299335045210"><a name="zh-cn_topic_0000001721582972_p1299335045210"></a><a name="zh-cn_topic_0000001721582972_p1299335045210"></a><code>None</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001721582972_row10521123219533"><td class="cellrowborder" valign="top" width="13.528647135286473%" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000001721582972_p105521753165311"><a name="zh-cn_topic_0000001721582972_p105521753165311"></a><a name="zh-cn_topic_0000001721582972_p105521753165311"></a><span id="zh-cn_topic_0000001721582972_ph8552953145316"><a name="zh-cn_topic_0000001721582972_ph8552953145316"></a><a name="zh-cn_topic_0000001721582972_ph8552953145316"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_9"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_9"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.439156084391561%" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000001721582972_p355275325317"><a name="zh-cn_topic_0000001721582972_p355275325317"></a><a name="zh-cn_topic_0000001721582972_p355275325317"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.76912308769123%" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000001721582972_p1255265315316"><a name="zh-cn_topic_0000001721582972_p1255265315316"></a><a name="zh-cn_topic_0000001721582972_p1255265315316"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.529047095290471%" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000001721582972_p18552195312537"><a name="zh-cn_topic_0000001721582972_p18552195312537"></a><a name="zh-cn_topic_0000001721582972_p18552195312537"></a><code>int32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.899110088991101%" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000001721582972_p1955212532537"><a name="zh-cn_topic_0000001721582972_p1955212532537"></a><a name="zh-cn_topic_0000001721582972_p1955212532537"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.90900909909009%" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000001721582972_p6553145311533"><a name="zh-cn_topic_0000001721582972_p6553145311533"></a><a name="zh-cn_topic_0000001721582972_p6553145311533"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.338966103389662%" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000001721582972_p155531253115310"><a name="zh-cn_topic_0000001721582972_p155531253115310"></a><a name="zh-cn_topic_0000001721582972_p155531253115310"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.64893510648935%" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000001721582972_p155531553195310"><a name="zh-cn_topic_0000001721582972_p155531553195310"></a><a name="zh-cn_topic_0000001721582972_p155531553195310"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.068993100689932%" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000001721582972_p7191710155411"><a name="zh-cn_topic_0000001721582972_p7191710155411"></a><a name="zh-cn_topic_0000001721582972_p7191710155411"></a><code>float32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.86901309869013%" headers="mcps1.2.11.1.10 "><p id="zh-cn_topic_0000001721582972_p85531353165311"><a name="zh-cn_topic_0000001721582972_p85531353165311"></a><a name="zh-cn_topic_0000001721582972_p85531353165311"></a><code>float32</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000001721582972_row1636253975315"><td class="cellrowborder" valign="top" width="13.528647135286473%" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000001721582972_p1086716244545"><a name="zh-cn_topic_0000001721582972_p1086716244545"></a><a name="zh-cn_topic_0000001721582972_p1086716244545"></a><span id="zh-cn_topic_0000001721582972_ph14867162418546"><a name="zh-cn_topic_0000001721582972_ph14867162418546"></a><a name="zh-cn_topic_0000001721582972_ph14867162418546"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_10"></a><a name="zh-cn_topic_0000001721582972_zh-cn_topic_0000001312391781_term11962195213215_10"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.439156084391561%" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000001721582972_p1886752417545"><a name="zh-cn_topic_0000001721582972_p1886752417545"></a><a name="zh-cn_topic_0000001721582972_p1886752417545"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.76912308769123%" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000001721582972_p1086722417540"><a name="zh-cn_topic_0000001721582972_p1086722417540"></a><a name="zh-cn_topic_0000001721582972_p1086722417540"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.529047095290471%" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000001721582972_p15867142410548"><a name="zh-cn_topic_0000001721582972_p15867142410548"></a><a name="zh-cn_topic_0000001721582972_p15867142410548"></a><code>int32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="8.899110088991101%" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000001721582972_p198678245540"><a name="zh-cn_topic_0000001721582972_p198678245540"></a><a name="zh-cn_topic_0000001721582972_p198678245540"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.90900909909009%" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000001721582972_p1461012314556"><a name="zh-cn_topic_0000001721582972_p1461012314556"></a><a name="zh-cn_topic_0000001721582972_p1461012314556"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.338966103389662%" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000001721582972_p5867192412546"><a name="zh-cn_topic_0000001721582972_p5867192412546"></a><a name="zh-cn_topic_0000001721582972_p5867192412546"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.64893510648935%" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000001721582972_p10867224105410"><a name="zh-cn_topic_0000001721582972_p10867224105410"></a><a name="zh-cn_topic_0000001721582972_p10867224105410"></a><code>None</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="10.068993100689932%" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000001721582972_p18224124195515"><a name="zh-cn_topic_0000001721582972_p18224124195515"></a><a name="zh-cn_topic_0000001721582972_p18224124195515"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="9.86901309869013%" headers="mcps1.2.11.1.10 "><p id="zh-cn_topic_0000001721582972_p5867162410543"><a name="zh-cn_topic_0000001721582972_p5867162410543"></a><a name="zh-cn_topic_0000001721582972_p5867162410543"></a><code>float32</code></p>
    </td>
    </tr>
    </tbody>
    </table>

    >**说明：**<br>
    >全量化场景：若`dequant_scale`需要以`float32`类型传入，在调用`torch_npu.npu_mm_all_reduce_base`前，需通过`torch_npu.npu_trans_quant_param`接口对`dequant_scale`进行处理为`int64`类型（处理方法见对应的接口使用说明）。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp


    def run_mm_all_reduce_base(rank, world_size, master_ip, master_port, x1_shape, x2_shape, dtype):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=init_method)
        from torch.distributed.distributed_c10d import _get_default_group

        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)

        input_ = torch.randn(x1_shape, dtype=dtype).npu()
        weight = torch.randn(x2_shape, dtype=dtype).npu()
        output = torch_npu.npu_mm_all_reduce_base(input_, weight, hcom_info, reduce_op="sum")
        print("output: ", output)


    if __name__ == "__main__":
        worksize = 8
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [128, 512]
        x2_shape = [512, 64]
        dtype = torch.float16

        mp.spawn(
            run_mm_all_reduce_base,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape, dtype),
            nprocs=worksize,
        )
    
    # 执行上述代码的输出类似如下
    output:  tensor([[ 60.7500,  -0.8770, -32.7812,  ...,   6.9219,  45.1250,  -1.4062],
            [-32.4688,  -5.7734, -19.7500,  ...,   6.2227, -63.9688, -42.1250],
            [-10.0781,  70.0000, -40.5938,  ...,  16.0000,  28.5312,  34.9688],
            ...,
            [  6.4844,  33.2500, -12.0781,  ..., -57.5312, -37.0000, -14.3203],
            [ -9.2422, -41.1562,   4.7188,  ...,   6.2812, -12.9531, -64.6250],
            [-25.3750,  13.9141,   9.8281,  ..., -21.7188,  64.5625, -56.1562]],
        device='npu:1', dtype=torch.float16)
    output:  tensor([[ 60.7500,  -0.8770, -32.7812,  ...,   6.9219,  45.1250,  -1.4062],
            [-32.4688,  -5.7734, -19.7500,  ...,   6.2227, -63.9688, -42.1250],
            [-10.0781,  70.0000, -40.5938,  ...,  16.0000,  28.5312,  34.9688],
            ...,
            [  6.4844,  33.2500, -12.0781,  ..., -57.5312, -37.0000, -14.3203],
            [ -9.2422, -41.1562,   4.7188,  ...,   6.2812, -12.9531, -64.6250],
            [-25.3750,  13.9141,   9.8281,  ..., -21.7188,  64.5625, -56.1562]],
        device='npu:0', dtype=torch.float16)

    ```

- 图模式调用

     非量化、伪量化、全量化使能NZ调用示例如下：

    ```python
    import torch
    import torch_npu
    import torch.distributed as dist
    import torch.multiprocessing as mp
    import numpy as np


    class MM_ALLREDUCE_GRAPH_Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
            self,
            x1,
            x2,
            hcom,
            reduce_op,
            bias,
            antiquant_scale,
            antiquant_offset,
            x3,
            dequant_scale,
        ):
            output_npu = torch_npu.npu_mm_all_reduce_base(
                x1=x1,
                x2=x2,
                hcom=hcom,
                reduce_op=reduce_op,
                bias=bias,
                antiquant_scale=antiquant_scale,
                antiquant_offset=antiquant_offset,
                x3=x3,
                dequant_scale=dequant_scale,
            )
            return output_npu


    class MM_ALLREDUCE_A8W8_GRAPH_Model(MM_ALLREDUCE_GRAPH_Model):
        def __init__(self):
            super().__init__()

        def forward(
            self,
            x1,
            x2,
            hcom,
            reduce_op,
            bias,
            antiquant_scale,
            antiquant_offset,
            x3,
            dequant_scale,
        ):
            output_npu = torch_npu.npu_mm_all_reduce_base(
                x1=x1,
                x2=x2.t(),
                hcom=hcom,
                reduce_op=reduce_op,
                bias=bias,
                antiquant_scale=antiquant_scale,
                antiquant_offset=antiquant_offset,
                x3=x3,
                dequant_scale=dequant_scale,
            )
            return output_npu


    def define_model(model, graph_type):
        import torchair

        if graph_type == 1:  # 传统入图模式，静态shape+在线编译场景
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN入图模式，动态shape+二进制
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        else:
            print("Error type")
        return model


    def get_graph(
        input,
        weight,
        hcomm_info,
        dequant_scale,
        bias,
        antiquant_scale,
        antiquant_offset,
        x3,
    ):
        model = MM_ALLREDUCE_A8W8_GRAPH_Model()
        model = define_model(model, 2)  # 1:静态入图;2:动态入图;
        output = model(
            x1=input,
            x2=weight,
            hcom=hcomm_info,
            reduce_op="sum",
            bias=bias,
            antiquant_scale=antiquant_scale,
            antiquant_offset=antiquant_offset,
            x3=x3,
            dequant_scale=dequant_scale,
        )
        return output


    def run_mc2_a16w16(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.float16)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        output_a16w16 = get_graph(input, weight, hcom_info, None, None, None, None, None)
        return output_a16w16


    def run_mc2_a8w8(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.int8)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
        dequant_scale = (
            torch.randn(x2_shape[0], dtype=torch.float32)
            .uniform_(float(-10), float(10))
            .npu()
        )
        dequant_scale = torch_npu.npu_trans_quant_param(dequant_scale)
        output_a8w8 = get_graph(
            input, weight_nz, hcom_info, dequant_scale, None, None, None, None
        )
        return output_a8w8


    def run_mc2_a16w8(x1_shape, x2_shape, hcom_info):
        np_input = np.random.uniform(float(-3), float(3), size=x1_shape).astype(np.float16)
        np_weight = np.random.uniform(float(-3), float(3), size=x2_shape).astype(np.int8)
        input = torch.tensor(np_input).npu()
        weight = torch.tensor(np_weight).npu()
        weight_nz = torch_npu.npu_format_cast(weight.contiguous(), 29)
        antiquant_scale = (
            torch.randn(x2_shape[0], dtype=torch.float16)
            .uniform_(float(-1), float(1))
            .npu()
        )
        antiquant_offset = torch.ones(x2_shape[0], dtype=torch.float16).npu()
        output_a16w8 = get_graph(
            input, weight_nz, hcom_info, None, None, antiquant_scale, antiquant_offset, None
        )
        return output_a16w8


    def run_mm_all_reduce_base(
        rank, world_size, master_ip, master_port, x1_shape, x2_shape, op_type
    ):
        torch_npu.npu.set_device(rank)
        init_method = "tcp://" + master_ip + ":" + master_port
        dist.init_process_group(
            backend="hccl", rank=rank, world_size=world_size, init_method=init_method
        )
        from torch.distributed.distributed_c10d import _get_default_group

        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcom_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(
                rank
            )
        else:
            hcom_info = default_pg.get_hccl_comm_name(rank)
        output = None
        # 非量化调用
        if op_type == "a16w16":
            output = run_mc2_a16w16(x1_shape, x2_shape, hcom_info)
        # 伪量化调用
        if op_type == "a16w8":
            output = run_mc2_a16w8(x1_shape, x2_shape, hcom_info)
        # 全量化调用
        if op_type == "a8w8":
            output = run_mc2_a8w8(x1_shape, x2_shape, hcom_info)
        print("output:", output)


    if __name__ == "__main__":
        worksize = 2
        master_ip = "127.0.0.1"
        master_port = "50001"
        x1_shape = [1280, 5120]
        x2_shape = [640, 5120]
        op_type = "a16w8"  # Options: a16w16, a16w8, a8w8
        mp.spawn(
            run_mm_all_reduce_base,
            args=(worksize, master_ip, master_port, x1_shape, x2_shape, op_type),
            nprocs=worksize,
        )

    
    # 执行上述代码的输出类似如下
    output: tensor([[-3.6594e+01, -8.4219e+00, -7.3688e+01,  ..., -4.9531e+01,
            -4.4438e+01, -1.2300e+02],
            [ 1.3225e+02,  2.4175e+02, -1.6094e+01,  ...,  1.4062e+02,
            -1.5750e+01,  4.0375e+01],
            [ 1.1931e+02,  9.7000e+01, -1.4200e+02,  ..., -1.2912e+02,
            -3.6062e+01,  7.0750e+01],
            ...,
            [-1.1031e+02, -6.4750e+01, -1.6500e+01,  ...,  2.3675e+02,
            9.6750e+01, -1.2662e+02],
            [-1.2569e+02,  2.3288e+02,  6.6250e+01,  ...,  3.0812e+01,
            6.2500e-02, -2.0550e+02],
            [ 7.4062e+01, -6.0100e+02, -3.0750e+02,  ..., -2.1500e+02,
            -2.4450e+02,  3.2400e+02]], device='npu:1', dtype=torch.float16)
    output: tensor([[-3.6594e+01, -8.4219e+00, -7.3688e+01,  ..., -4.9531e+01,
            -4.4438e+01, -1.2300e+02],
            [ 1.3225e+02,  2.4175e+02, -1.6094e+01,  ...,  1.4062e+02,
            -1.5750e+01,  4.0375e+01],
            [ 1.1931e+02,  9.7000e+01, -1.4200e+02,  ..., -1.2912e+02,
            -3.6062e+01,  7.0750e+01],
            ...,
            [-1.1031e+02, -6.4750e+01, -1.6500e+01,  ...,  2.3675e+02,
            9.6750e+01, -1.2662e+02],
            [-1.2569e+02,  2.3288e+02,  6.6250e+01,  ...,  3.0812e+01,
            6.2500e-02, -2.0550e+02],
            [ 7.4062e+01, -6.0100e+02, -3.0750e+02,  ..., -2.1500e+02,
            -2.4450e+02,  3.2400e+02]], device='npu:0', dtype=torch.float16)
    ```

