# torch_npu.npu_grouped_matmul<a name="ZH-CN_TOPIC_0000002229788810"></a>

## 功能说明<a name="zh-cn_topic_0000002262888689_section1290611593405"></a>

-   算子功能：`npu_grouped_matmul`是一种对多个矩阵乘法（matmul）操作进行分组计算的高效方法。该API实现了对多个矩阵乘法操作的批量处理，通过将具有相同形状或相似形状的矩阵乘法操作组合在一起，减少内存访问开销和计算资源的浪费，从而提高计算效率。

-   计算公式：
    -   非量化场景（公式1）：

        ![](./figures/zh-cn_formulaimage_0000002262918025.png)

    -   perchannel、pertensor量化场景（公式2）：

        ![](./figures/zh-cn_formulaimage_0000002331514405.png)

        -   `x`为`int8`输入，`bias`为`int32`输入（公式2-1）：

            ![](./figures/zh-cn_formulaimage_0000002330304749.png)

        -   `x`为`int8`输入，`bias`为`bfloat16`、`float16`、`float32`输入，无offset（公式2-2）：

            ![](./figures/zh-cn_formulaimage_0000002296388208.png)

    -   pertoken、pertensor+pertensor、pertensor+perchannel量化场景（公式3）：

        ![](./figures/zh-cn_formulaimage_0000002228564948.png)

        -   `x`为`int8`输入，bias为`int32`输入（公式3-1）：

            ![](./figures/zh-cn_formulaimage_0000002330394453.png)

        -   `x`为`int8`输入，`bias`为`bfloat16`，`float16`，`float32`输入（公式3-2）：

            ![](./figures/zh-cn_formulaimage_0000002296235560.png)

    -   伪量化场景（公式4）：

        ![](./figures/zh-cn_formulaimage_0000002262918029.png)

## 函数原型<a name="zh-cn_topic_0000002262888689_section87878612417"></a>
```
npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=None, group_list_type=0, act_type=0, output_dtype=None, tuning_config=None) -> List[torch.Tensor]
```

## 参数说明<a name="zh-cn_topic_0000002262888689_section135561610204110"></a>

-   **x** (`List[Tensor]`)：输入矩阵列表，表示矩阵乘法中的左矩阵。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`float16`、`float32`、`bfloat16`和`int8`。
        -   <term>Atlas 推理系列产品</term>：`float16`。

    -   列表最大长度为128。
    -   当split\_item=0时，张量支持2至6维输入；其他情况下，张量仅支持2维输入。

-   **weight** (`List[Tensor]`)：权重矩阵列表，表示矩阵乘法中的右矩阵。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
            -   当`group_list`输入类型为`List[int]`时，支持`float16`、`float32`、`bfloat16`和`int8`。
            -   当`group_list`输入类型为`Tensor`时，支持`float16`、`float32`、`bfloat16`、`int4`和`int8`。

        -   <term>Atlas 推理系列产品</term>：`float16`。

    -   列表最大长度为128。
    -   每个张量支持2维或3维输入。

-   **bias** (`List[Tensor]`)：每个分组的矩阵乘法输出的独立偏置项。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`float16`、`float32`和`int32`。
        -   <term>Atlas 推理系列产品</term>：`float16`。

    -   列表长度与weight列表长度相同。
    -   每个张量仅支持1维输入。

-   **scale** (`List[Tensor]`)：用于缩放原数值以匹配量化后的范围值，代表量化参数中的缩放因子，对应公式（2）、公式（3）和公式（5）。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
            -   当`group_list`输入类型为`List[int]`时，支持`int64`。
            -   当`group_list`输入类型为`Tensor`时，支持`float32`、`bfloat16`和`int64`。

        -   <term>Atlas 推理系列产品</term>：仅支持传入`None`。

    -   列表长度与weight列表长度相同。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：每个张量仅支持1维输入。

-   **offset** (`List[Tensor]`)：用于调整量化后的数值偏移量，从而更准确地表示原始浮点数值，对应公式（2）。当前仅支持传入`None`。
-   **antiquant_scale** (`List[Tensor]`)：用于缩放原数值以匹配伪量化后的范围值，代表伪量化参数中的缩放因子，对应公式（4）。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`float16`、`bfloat16`。
        -   <term>Atlas 推理系列产品</term>：仅支持传入`None`。

    -   列表长度与weight列表长度相同。
    -   每个张量支持输入维度如下（其中$g$为matmul组数，$G$为per-group数，$G_i$为第i个tensor的per-group数）：
        -   伪量化per-channel场景，`weight`为单tensor时，shape限制为$[g, n]$；`weight`为多tensor时，shape限制为$[n_i]$。
        -   伪量化per-group场景，weight为单tensor时，shape限制为$[g, G, n]$; weight为多tensor时，shape限制为$[G_i, n_i]$。

-   **antiquant_offset** (`List[Tensor]`)：用于调整伪量化后的数值偏移量，从而更准确地表示原始浮点数值，对应公式（4）。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`float16`、`bfloat16`。
        -   <term>Atlas 推理系列产品</term>：仅支持传入`None`。

    -   列表长度与`weight`列表长度相同。
    -   每个张量输入维度和`antiquant_scale`输入维度一致。

-   **per_token_scale** (`List[Tensor]`)：用于缩放原数值以匹配量化后的范围值，代表per-token量化参数中由`x`量化引入的缩放因子，对应公式（3）和公式（5）。
    -   `group_list`输入类型为`List[int]`时，当前只支持传入`None`。
    -   `group_list`输入类型为`Tensor`时：
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`。
        -   列表长度与`x`列表长度相同。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：每个张量仅支持1维输入。

-   **group_list** (`List[int]`/`Tensor`)：用于指定分组的索引，表示x的第0维矩阵乘法的索引情况。数据类型支持`int64`。
    -   <term>Atlas 推理系列产品</term>：仅支持<code>**Tensor**</code>类型。仅支持1维输入，长度与<code>weight</code>列表长度相同。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持<code>**List[int]**</code>或<code>**Tensor**</code>类型。若为<code>**Tensor**</code>类型，仅支持1维输入，长度与<code>weight</code>列表长度相同。
    -   配置值要求如下：
        -   `group_list`输入类型为`List[int]`时，配置值必须为非负递增数列，且长度不能为1。
        -   `group_list`输入类型为`Tensor`时：
            -   当`group_list_type`为0时，`group_list`必须为非负、单调非递减数列。
            -   当`group_list_type`为1时，`group_list`必须为非负数列，且长度不能为1。
            -   当`group_list_type`为2时，`group_list` shape为$[E, 2]$，E表示Group大小，数据排布为$[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]$，其中groupSize为分组轴上每组大小，必须为非负数。

-   **activation_input** (`List[Tensor]`)：代表激活函数的反向输入，当前仅支持传入`None`。
-   **activation_quant_scale** (`List[Tensor]`)：预留参数，当前只支持传入`None`。
-   **activation_quant_offset** (`List[Tensor]`)：预留参数，当前只支持传入`None`。
-   **split_item** (`int`)：用于指定切分模式。数据类型支持`int32`。
    -   0、1：输出为多个张量，数量与`weight`相同。
    -   2、3：输出为单个张量。

-   **group_type** (`int`)：代表需要分组的轴。数据类型支持`int32`。
    -   `group_list`输入类型为`List[int]`时仅支持传入`None`。

    -   `group_list`输入类型为`Tensor`时，若矩阵乘为$C[m,n]=A[m,k]*B[k,n]$，`group_type`支持的枚举值为：-1代表不分组；0代表m轴分组；1代表n轴分组。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前支持取-1、0。
        -   <term>Atlas 推理系列产品</term>：当前只支持取0。

-   **group_list_type** (`int`)：代表`group_list`的表达形式。数据类型支持`int32`。
    -   `group_list`输入类型为`List[int]`时仅支持传入`None`。

    -   `group_list`输入类型为`Tensor`时可取值0、1或2：
        -   0：默认值，`group_list`中数值为分组轴大小的cumsum结果（累积和）。
        -   1：`group_list`中数值为分组轴上每组大小。
        -   2：`group_list` shape为$[E, 2]$，E表示Group大小，数据排布为$[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]$，其中groupSize为分组轴上每组大小。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅当`x`和`weight`参数输入类型为`INT8`，并且`group_type`取0（m轴分组）时，支持取2。
        -   <term>Atlas 推理系列产品</term>：不支持取2。
    
-   **act_type** (`int`)：代表激活函数类型。数据类型支持`int32`。
    -   `group_list`输入类型为`List[int]`时仅支持传入`None`。

    -   `group_list`输入类型为`Tensor`时，支持的枚举值包括：0代表不激活；1代表`RELU`激活；2代表`GELU_TANH`激活；3代表暂不支持；4代表`FAST_GELU`激活；5代表`SILU`激活。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为0-5。
        -   <term>Atlas 推理系列产品</term>：当前只支持传入0。

-   **output_dtype** (`torch.dtype`)：输出数据类型。支持的配置包括：
    -   `None`：默认值，表示输出数据类型与输入`x`的数据类型相同。
    -   与输出`y`数据类型一致的类型，具体参考[约束说明](#zh-cn_topic_0000002262888689_section618392112366)。

-   **tuning_config** (`List[int]`)：可选参数，数组中的第一个元素表示各个专家处理的token数的预期值，算子tiling时会按照数组中的第一个元素进行最优tiling，性能更优（使用场景参见[约束说明](#zh-cn_topic_0000002262888689_section618392112366)）；从第二个元素开始预留，用户无须填写，未来会进行扩展。如不使用该参数不传即可。
    -   <term>Atlas 推理系列产品</term>：当前暂不支持该参数。

## 返回值<a name="zh-cn_topic_0000002262888689_section1558311519405"></a>

`List[Tensor]`：

-   当`split_item`为0或1时，返回的张量数量与`weight`相同。
-   当`split_item`为2或3时，返回的张量数量为1。

## 约束说明<a name="zh-cn_topic_0000002262888689_section618392112366"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：内轴限制InnerLimit为65536。
-   `x`和`weight`中每一组tensor的最后一维大小都应小于InnerLimit。x<sub>i</sub>的最后一维指当x不转置时<code>x<sub>i</sub></code>的K轴或当`x`转置时<code>x<sub>i</sub></code>的$M$轴。<code>weight<sub>i</sub></code>的最后一维指当`weight`不转置时<code>weight<sub>i</sub></code>的$N$轴或当`weight`转置时<code>weight<sub>i</sub></code>的$K$轴。

-   `tuning_config`使用场景限制：

    仅在量化场景（输入`int8`，输出为`int32`/`bfloat16`/`float16`/`int8`，数据类型如下表），且为单tensor单专家的场景下使用。

    <a name="zh-cn_topic_0000002262888689_table20914818152219"></a>
    <table><thead align="left"><tr id="zh-cn_topic_0000002262888689_row4914191812218"><th class="cellrowborder" valign="top" width="20.79%" id="mcps1.1.5.1.1"><p id="zh-cn_topic_0000002262888689_p1791471812220"><a name="zh-cn_topic_0000002262888689_p1791471812220"></a><a name="zh-cn_topic_0000002262888689_p1791471812220"></a>x</p>
    </th>
    <th class="cellrowborder" valign="top" width="21.88%" id="mcps1.1.5.1.2"><p id="zh-cn_topic_0000002262888689_p3914181818223"><a name="zh-cn_topic_0000002262888689_p3914181818223"></a><a name="zh-cn_topic_0000002262888689_p3914181818223"></a>weight</p>
    </th>
    <th class="cellrowborder" valign="top" width="34.760000000000005%" id="mcps1.1.5.1.3"><p id="zh-cn_topic_0000002262888689_p18914181811229"><a name="zh-cn_topic_0000002262888689_p18914181811229"></a><a name="zh-cn_topic_0000002262888689_p18914181811229"></a>output_dtype</p>
    </th>
    <th class="cellrowborder" valign="top" width="22.57%" id="mcps1.1.5.1.4"><p id="zh-cn_topic_0000002262888689_p2091451810225"><a name="zh-cn_topic_0000002262888689_p2091451810225"></a><a name="zh-cn_topic_0000002262888689_p2091451810225"></a>y</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="zh-cn_topic_0000002262888689_row149151918132217"><td class="cellrowborder" valign="top" width="20.79%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p1391551818221"><a name="zh-cn_topic_0000002262888689_p1391551818221"></a><a name="zh-cn_topic_0000002262888689_p1391551818221"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="21.88%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p169152018152216"><a name="zh-cn_topic_0000002262888689_p169152018152216"></a><a name="zh-cn_topic_0000002262888689_p169152018152216"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="34.760000000000005%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p1991541892219"><a name="zh-cn_topic_0000002262888689_p1991541892219"></a><a name="zh-cn_topic_0000002262888689_p1991541892219"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="22.57%" headers="mcps1.1.5.1.4 "><p id="zh-cn_topic_0000002262888689_p169151918182218"><a name="zh-cn_topic_0000002262888689_p169151918182218"></a><a name="zh-cn_topic_0000002262888689_p169151918182218"></a><code>int8</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002262888689_row1391591820221"><td class="cellrowborder" valign="top" width="20.79%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p791511183228"><a name="zh-cn_topic_0000002262888689_p791511183228"></a><a name="zh-cn_topic_0000002262888689_p791511183228"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="21.88%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p991581812220"><a name="zh-cn_topic_0000002262888689_p991581812220"></a><a name="zh-cn_topic_0000002262888689_p991581812220"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="34.760000000000005%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p1191514181221"><a name="zh-cn_topic_0000002262888689_p1191514181221"></a><a name="zh-cn_topic_0000002262888689_p1191514181221"></a><code>bfloat16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="22.57%" headers="mcps1.1.5.1.4 "><p id="zh-cn_topic_0000002262888689_p14915181810226"><a name="zh-cn_topic_0000002262888689_p14915181810226"></a><a name="zh-cn_topic_0000002262888689_p14915181810226"></a><code>bfloat16</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002262888689_row14915141812213"><td class="cellrowborder" valign="top" width="20.79%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p1091561818225"><a name="zh-cn_topic_0000002262888689_p1091561818225"></a><a name="zh-cn_topic_0000002262888689_p1091561818225"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="21.88%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p091511892216"><a name="zh-cn_topic_0000002262888689_p091511892216"></a><a name="zh-cn_topic_0000002262888689_p091511892216"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="34.760000000000005%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p179151318182219"><a name="zh-cn_topic_0000002262888689_p179151318182219"></a><a name="zh-cn_topic_0000002262888689_p179151318182219"></a><code>float16</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="22.57%" headers="mcps1.1.5.1.4 "><p id="zh-cn_topic_0000002262888689_p1691591832218"><a name="zh-cn_topic_0000002262888689_p1691591832218"></a><a name="zh-cn_topic_0000002262888689_p1691591832218"></a><code>float16</code></p>
    </td>
    </tr>
    <tr id="zh-cn_topic_0000002262888689_row7915718142220"><td class="cellrowborder" valign="top" width="20.79%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p1291515186228"><a name="zh-cn_topic_0000002262888689_p1291515186228"></a><a name="zh-cn_topic_0000002262888689_p1291515186228"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="21.88%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p15915151814224"><a name="zh-cn_topic_0000002262888689_p15915151814224"></a><a name="zh-cn_topic_0000002262888689_p15915151814224"></a><code>int8</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="34.760000000000005%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p15916418162213"><a name="zh-cn_topic_0000002262888689_p15916418162213"></a><a name="zh-cn_topic_0000002262888689_p15916418162213"></a><code>int32</code></p>
    </td>
    <td class="cellrowborder" valign="top" width="22.57%" headers="mcps1.1.5.1.4 "><p id="zh-cn_topic_0000002262888689_p169161182228"><a name="zh-cn_topic_0000002262888689_p169161182228"></a><a name="zh-cn_topic_0000002262888689_p169161182228"></a><code>int32</code></p>
    </td>
    </tr>
    </tbody>
    </table>

-   各场景输入与输出数据类型使用约束：
    -   **`group_list`输入类型为`List[int]`时**，<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>数据类型使用约束。

        **表 1**  数据类型约束

        <a name="zh-cn_topic_0000002262888689_table14115265341"></a>
        <table><thead align="left"><tr id="zh-cn_topic_0000002262888689_row212026113415"><th class="cellrowborder" valign="top" width="8.04%" id="mcps1.2.10.1.1"><p id="zh-cn_topic_0000002262888689_p1428266348"><a name="zh-cn_topic_0000002262888689_p1428266348"></a><a name="zh-cn_topic_0000002262888689_p1428266348"></a>场景</p>
        </th>
        <th class="cellrowborder" valign="top" width="9.54%" id="mcps1.2.10.1.2"><p id="zh-cn_topic_0000002262888689_p15218269349"><a name="zh-cn_topic_0000002262888689_p15218269349"></a><a name="zh-cn_topic_0000002262888689_p15218269349"></a>x</p>
        </th>
        <th class="cellrowborder" valign="top" width="9.86%" id="mcps1.2.10.1.3"><p id="zh-cn_topic_0000002262888689_p162826103411"><a name="zh-cn_topic_0000002262888689_p162826103411"></a><a name="zh-cn_topic_0000002262888689_p162826103411"></a>weight</p>
        </th>
        <th class="cellrowborder" valign="top" width="12.4%" id="mcps1.2.10.1.4"><p id="zh-cn_topic_0000002262888689_p12213265347"><a name="zh-cn_topic_0000002262888689_p12213265347"></a><a name="zh-cn_topic_0000002262888689_p12213265347"></a>bias</p>
        </th>
        <th class="cellrowborder" valign="top" width="7.64%" id="mcps1.2.10.1.5"><p id="zh-cn_topic_0000002262888689_p921826143415"><a name="zh-cn_topic_0000002262888689_p921826143415"></a><a name="zh-cn_topic_0000002262888689_p921826143415"></a>scale</p>
        </th>
        <th class="cellrowborder" valign="top" width="13.28%" id="mcps1.2.10.1.6"><p id="zh-cn_topic_0000002262888689_p17210266349"><a name="zh-cn_topic_0000002262888689_p17210266349"></a><a name="zh-cn_topic_0000002262888689_p17210266349"></a>antiquant_scale</p>
        </th>
        <th class="cellrowborder" valign="top" width="13.13%" id="mcps1.2.10.1.7"><p id="zh-cn_topic_0000002262888689_p17282643414"><a name="zh-cn_topic_0000002262888689_p17282643414"></a><a name="zh-cn_topic_0000002262888689_p17282643414"></a>antiquant_offset</p>
        </th>
        <th class="cellrowborder" valign="top" width="13.43%" id="mcps1.2.10.1.8"><p id="zh-cn_topic_0000002262888689_p32182683415"><a name="zh-cn_topic_0000002262888689_p32182683415"></a><a name="zh-cn_topic_0000002262888689_p32182683415"></a>output_dtype</p>
        </th>
        <th class="cellrowborder" valign="top" width="12.68%" id="mcps1.2.10.1.9"><p id="zh-cn_topic_0000002262888689_p918920133617"><a name="zh-cn_topic_0000002262888689_p918920133617"></a><a name="zh-cn_topic_0000002262888689_p918920133617"></a>y</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="zh-cn_topic_0000002262888689_row17212260345"><td class="cellrowborder" rowspan="3" valign="top" width="8.04%" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000002262888689_p1821426143419"><a name="zh-cn_topic_0000002262888689_p1821426143419"></a><a name="zh-cn_topic_0000002262888689_p1821426143419"></a>非量化</p>
        <p id="zh-cn_topic_0000002262888689_p29466188235"><a name="zh-cn_topic_0000002262888689_p29466188235"></a><a name="zh-cn_topic_0000002262888689_p29466188235"></a></p>
        </td>
        <td class="cellrowborder" valign="top" width="9.54%" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000002262888689_p721826103416"><a name="zh-cn_topic_0000002262888689_p721826103416"></a><a name="zh-cn_topic_0000002262888689_p721826103416"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000002262888689_p11252617348"><a name="zh-cn_topic_0000002262888689_p11252617348"></a><a name="zh-cn_topic_0000002262888689_p11252617348"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="12.4%" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000002262888689_p2292693418"><a name="zh-cn_topic_0000002262888689_p2292693418"></a><a name="zh-cn_topic_0000002262888689_p2292693418"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="7.64%" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000002262888689_p168271384720"><a name="zh-cn_topic_0000002262888689_p168271384720"></a><a name="zh-cn_topic_0000002262888689_p168271384720"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" width="13.28%" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000002262888689_p684218334714"><a name="zh-cn_topic_0000002262888689_p684218334714"></a><a name="zh-cn_topic_0000002262888689_p684218334714"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" width="13.13%" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000002262888689_p3849113164710"><a name="zh-cn_topic_0000002262888689_p3849113164710"></a><a name="zh-cn_topic_0000002262888689_p3849113164710"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" width="13.43%" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000002262888689_p72152613419"><a name="zh-cn_topic_0000002262888689_p72152613419"></a><a name="zh-cn_topic_0000002262888689_p72152613419"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.10.1.9 "><p id="zh-cn_topic_0000002262888689_p1018913083618"><a name="zh-cn_topic_0000002262888689_p1018913083618"></a><a name="zh-cn_topic_0000002262888689_p1018913083618"></a><code>float16</code></p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000002262888689_row1423261346"><td class="cellrowborder" valign="top" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000002262888689_p152726133420"><a name="zh-cn_topic_0000002262888689_p152726133420"></a><a name="zh-cn_topic_0000002262888689_p152726133420"></a><code>bfloat16</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000002262888689_p320264344"><a name="zh-cn_topic_0000002262888689_p320264344"></a><a name="zh-cn_topic_0000002262888689_p320264344"></a><code>bfloat16</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000002262888689_p162126203413"><a name="zh-cn_topic_0000002262888689_p162126203413"></a><a name="zh-cn_topic_0000002262888689_p162126203413"></a><code>float32</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000002262888689_p2856133194710"><a name="zh-cn_topic_0000002262888689_p2856133194710"></a><a name="zh-cn_topic_0000002262888689_p2856133194710"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000002262888689_p287018317477"><a name="zh-cn_topic_0000002262888689_p287018317477"></a><a name="zh-cn_topic_0000002262888689_p287018317477"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000002262888689_p208765310478"><a name="zh-cn_topic_0000002262888689_p208765310478"></a><a name="zh-cn_topic_0000002262888689_p208765310478"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000002262888689_p10212614344"><a name="zh-cn_topic_0000002262888689_p10212614344"></a><a name="zh-cn_topic_0000002262888689_p10212614344"></a><code>bfloat16</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000002262888689_p1718910193612"><a name="zh-cn_topic_0000002262888689_p1718910193612"></a><a name="zh-cn_topic_0000002262888689_p1718910193612"></a><code>bfloat16</code></p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000002262888689_row7946171817236"><td class="cellrowborder" valign="top" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000002262888689_p69461118132315"><a name="zh-cn_topic_0000002262888689_p69461118132315"></a><a name="zh-cn_topic_0000002262888689_p69461118132315"></a><code>float32</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000002262888689_p1294618187236"><a name="zh-cn_topic_0000002262888689_p1294618187236"></a><a name="zh-cn_topic_0000002262888689_p1294618187236"></a><code>float32</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000002262888689_p19946018102313"><a name="zh-cn_topic_0000002262888689_p19946018102313"></a><a name="zh-cn_topic_0000002262888689_p19946018102313"></a><code>float32</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000002262888689_p294611816233"><a name="zh-cn_topic_0000002262888689_p294611816233"></a><a name="zh-cn_topic_0000002262888689_p294611816233"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000002262888689_p994618182232"><a name="zh-cn_topic_0000002262888689_p994618182232"></a><a name="zh-cn_topic_0000002262888689_p994618182232"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000002262888689_p169471918162316"><a name="zh-cn_topic_0000002262888689_p169471918162316"></a><a name="zh-cn_topic_0000002262888689_p169471918162316"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000002262888689_p15947918152319"><a name="zh-cn_topic_0000002262888689_p15947918152319"></a><a name="zh-cn_topic_0000002262888689_p15947918152319"></a><code>float32</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000002262888689_p17947718142320"><a name="zh-cn_topic_0000002262888689_p17947718142320"></a><a name="zh-cn_topic_0000002262888689_p17947718142320"></a><code>float32</code></p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000002262888689_row827266342"><td class="cellrowborder" valign="top" width="8.04%" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000002262888689_p182172673418"><a name="zh-cn_topic_0000002262888689_p182172673418"></a><a name="zh-cn_topic_0000002262888689_p182172673418"></a>per-channel<span id="zh-cn_topic_0000002262888689_ph1417104719313"><a name="zh-cn_topic_0000002262888689_ph1417104719313"></a><a name="zh-cn_topic_0000002262888689_ph1417104719313"></a>全</span>量化</p>
        </td>
        <td class="cellrowborder" valign="top" width="9.54%" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000002262888689_p3262615345"><a name="zh-cn_topic_0000002262888689_p3262615345"></a><a name="zh-cn_topic_0000002262888689_p3262615345"></a><code>int8</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000002262888689_p32142610340"><a name="zh-cn_topic_0000002262888689_p32142610340"></a><a name="zh-cn_topic_0000002262888689_p32142610340"></a><code>int8</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="12.4%" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000002262888689_p32326133415"><a name="zh-cn_topic_0000002262888689_p32326133415"></a><a name="zh-cn_topic_0000002262888689_p32326133415"></a><code>int32</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="7.64%" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000002262888689_p12202633415"><a name="zh-cn_topic_0000002262888689_p12202633415"></a><a name="zh-cn_topic_0000002262888689_p12202633415"></a><code>int64</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="13.28%" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000002262888689_p72845512472"><a name="zh-cn_topic_0000002262888689_p72845512472"></a><a name="zh-cn_topic_0000002262888689_p72845512472"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" width="13.13%" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000002262888689_p1329316513477"><a name="zh-cn_topic_0000002262888689_p1329316513477"></a><a name="zh-cn_topic_0000002262888689_p1329316513477"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" width="13.43%" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000002262888689_p142162663414"><a name="zh-cn_topic_0000002262888689_p142162663414"></a><a name="zh-cn_topic_0000002262888689_p142162663414"></a><code>int8</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.10.1.9 "><p id="zh-cn_topic_0000002262888689_p171891900362"><a name="zh-cn_topic_0000002262888689_p171891900362"></a><a name="zh-cn_topic_0000002262888689_p171891900362"></a><code>int8</code></p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000002262888689_row10272612346"><td class="cellrowborder" rowspan="2" valign="top" width="8.04%" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000002262888689_p113916152412"><a name="zh-cn_topic_0000002262888689_p113916152412"></a><a name="zh-cn_topic_0000002262888689_p113916152412"></a>伪量化</p>
        <p id="zh-cn_topic_0000002262888689_p173891815154113"><a name="zh-cn_topic_0000002262888689_p173891815154113"></a><a name="zh-cn_topic_0000002262888689_p173891815154113"></a></p>
        </td>
        <td class="cellrowborder" valign="top" width="9.54%" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000002262888689_p42771411437"><a name="zh-cn_topic_0000002262888689_p42771411437"></a><a name="zh-cn_topic_0000002262888689_p42771411437"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="9.86%" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000002262888689_p11390191514419"><a name="zh-cn_topic_0000002262888689_p11390191514419"></a><a name="zh-cn_topic_0000002262888689_p11390191514419"></a><code>int8</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="12.4%" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000002262888689_p239014158414"><a name="zh-cn_topic_0000002262888689_p239014158414"></a><a name="zh-cn_topic_0000002262888689_p239014158414"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="7.64%" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000002262888689_p1211287174718"><a name="zh-cn_topic_0000002262888689_p1211287174718"></a><a name="zh-cn_topic_0000002262888689_p1211287174718"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" width="13.28%" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000002262888689_p12390515164116"><a name="zh-cn_topic_0000002262888689_p12390515164116"></a><a name="zh-cn_topic_0000002262888689_p12390515164116"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="13.13%" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000002262888689_p13842153693919"><a name="zh-cn_topic_0000002262888689_p13842153693919"></a><a name="zh-cn_topic_0000002262888689_p13842153693919"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="13.43%" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000002262888689_p20365134033913"><a name="zh-cn_topic_0000002262888689_p20365134033913"></a><a name="zh-cn_topic_0000002262888689_p20365134033913"></a><code>float16</code></p>
        </td>
        <td class="cellrowborder" valign="top" width="12.68%" headers="mcps1.2.10.1.9 "><p id="zh-cn_topic_0000002262888689_p6189150173618"><a name="zh-cn_topic_0000002262888689_p6189150173618"></a><a name="zh-cn_topic_0000002262888689_p6189150173618"></a><code>float16</code></p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000002262888689_row182192616343"><td class="cellrowborder" valign="top" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000002262888689_p5225105534319"><a name="zh-cn_topic_0000002262888689_p5225105534319"></a><a name="zh-cn_topic_0000002262888689_p5225105534319"></a><code>bfloat16</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000002262888689_p1238916156419"><a name="zh-cn_topic_0000002262888689_p1238916156419"></a><a name="zh-cn_topic_0000002262888689_p1238916156419"></a><code>int8</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000002262888689_p938941519416"><a name="zh-cn_topic_0000002262888689_p938941519416"></a><a name="zh-cn_topic_0000002262888689_p938941519416"></a><code>float32</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000002262888689_p112518764719"><a name="zh-cn_topic_0000002262888689_p112518764719"></a><a name="zh-cn_topic_0000002262888689_p112518764719"></a>无需赋值</p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000002262888689_p1559816378432"><a name="zh-cn_topic_0000002262888689_p1559816378432"></a><a name="zh-cn_topic_0000002262888689_p1559816378432"></a><code>bfloat16</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000002262888689_p1084813664316"><a name="zh-cn_topic_0000002262888689_p1084813664316"></a><a name="zh-cn_topic_0000002262888689_p1084813664316"></a><code>bfloat16</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000002262888689_p936233634314"><a name="zh-cn_topic_0000002262888689_p936233634314"></a><a name="zh-cn_topic_0000002262888689_p936233634314"></a><code>bfloat16</code></p>
        </td>
        <td class="cellrowborder" valign="top" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000002262888689_p918914053619"><a name="zh-cn_topic_0000002262888689_p918914053619"></a><a name="zh-cn_topic_0000002262888689_p918914053619"></a><code>bfloat16</code></p>
        </td>
        </tr>
        </tbody>
        </table>

    -   **`group_list`输入类型为`Tensor`时**，数据类型使用约束。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

            **表 2**  数据类型约束

            <a name="zh-cn_topic_0000002262888689_table161954316327"></a>
            <table><thead align="left"><tr id="zh-cn_topic_0000002262888689_row119218313212"><th class="cellrowborder" valign="top" width="7.4700000000000015%" id="mcps1.2.11.1.1"><p id="zh-cn_topic_0000002262888689_p1519214312326"><a name="zh-cn_topic_0000002262888689_p1519214312326"></a><a name="zh-cn_topic_0000002262888689_p1519214312326"></a>场景</p>
            </th>
            <th class="cellrowborder" valign="top" width="8.110000000000001%" id="mcps1.2.11.1.2"><p id="zh-cn_topic_0000002262888689_p21922315325"><a name="zh-cn_topic_0000002262888689_p21922315325"></a><a name="zh-cn_topic_0000002262888689_p21922315325"></a>x</p>
            </th>
            <th class="cellrowborder" valign="top" width="8.530000000000001%" id="mcps1.2.11.1.3"><p id="zh-cn_topic_0000002262888689_p21926312326"><a name="zh-cn_topic_0000002262888689_p21926312326"></a><a name="zh-cn_topic_0000002262888689_p21926312326"></a>weight</p>
            </th>
            <th class="cellrowborder" valign="top" width="7.790000000000001%" id="mcps1.2.11.1.4"><p id="zh-cn_topic_0000002262888689_p4192931327"><a name="zh-cn_topic_0000002262888689_p4192931327"></a><a name="zh-cn_topic_0000002262888689_p4192931327"></a>bias</p>
            </th>
            <th class="cellrowborder" valign="top" width="7.450000000000001%" id="mcps1.2.11.1.5"><p id="zh-cn_topic_0000002262888689_p51921318322"><a name="zh-cn_topic_0000002262888689_p51921318322"></a><a name="zh-cn_topic_0000002262888689_p51921318322"></a>scale</p>
            </th>
            <th class="cellrowborder" valign="top" width="12.660000000000002%" id="mcps1.2.11.1.6"><p id="zh-cn_topic_0000002262888689_p101921737324"><a name="zh-cn_topic_0000002262888689_p101921737324"></a><a name="zh-cn_topic_0000002262888689_p101921737324"></a>antiquant_scale</p>
            </th>
            <th class="cellrowborder" valign="top" width="13.180000000000003%" id="mcps1.2.11.1.7"><p id="zh-cn_topic_0000002262888689_p19192730329"><a name="zh-cn_topic_0000002262888689_p19192730329"></a><a name="zh-cn_topic_0000002262888689_p19192730329"></a>antiquant_offset</p>
            </th>
            <th class="cellrowborder" valign="top" width="12.440000000000001%" id="mcps1.2.11.1.8"><p id="zh-cn_topic_0000002262888689_p219243133210"><a name="zh-cn_topic_0000002262888689_p219243133210"></a><a name="zh-cn_topic_0000002262888689_p219243133210"></a>per_token_scale</p>
            </th>
            <th class="cellrowborder" valign="top" width="13.570000000000002%" id="mcps1.2.11.1.9"><p id="zh-cn_topic_0000002262888689_p31924323211"><a name="zh-cn_topic_0000002262888689_p31924323211"></a><a name="zh-cn_topic_0000002262888689_p31924323211"></a>output_dtype</p>
            </th>
            <th class="cellrowborder" valign="top" width="8.8%" id="mcps1.2.11.1.10"><p id="zh-cn_topic_0000002262888689_p619273133212"><a name="zh-cn_topic_0000002262888689_p619273133212"></a><a name="zh-cn_topic_0000002262888689_p619273133212"></a>y</p>
            </th>
            </tr>
            </thead>
            <tbody><tr id="zh-cn_topic_0000002262888689_row14192143133219"><td class="cellrowborder" rowspan="3" valign="top" width="7.4700000000000015%" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p7192103183220"><a name="zh-cn_topic_0000002262888689_p7192103183220"></a><a name="zh-cn_topic_0000002262888689_p7192103183220"></a>非量化</p>
            </td>
            <td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p14192931323"><a name="zh-cn_topic_0000002262888689_p14192931323"></a><a name="zh-cn_topic_0000002262888689_p14192931323"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.530000000000001%" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p12192733329"><a name="zh-cn_topic_0000002262888689_p12192733329"></a><a name="zh-cn_topic_0000002262888689_p12192733329"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.790000000000001%" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p6192736324"><a name="zh-cn_topic_0000002262888689_p6192736324"></a><a name="zh-cn_topic_0000002262888689_p6192736324"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.450000000000001%" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p719219303212"><a name="zh-cn_topic_0000002262888689_p719219303212"></a><a name="zh-cn_topic_0000002262888689_p719219303212"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="12.660000000000002%" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p919219310326"><a name="zh-cn_topic_0000002262888689_p919219310326"></a><a name="zh-cn_topic_0000002262888689_p919219310326"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="13.180000000000003%" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p919213173219"><a name="zh-cn_topic_0000002262888689_p919213173219"></a><a name="zh-cn_topic_0000002262888689_p919213173219"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p619211313217"><a name="zh-cn_topic_0000002262888689_p619211313217"></a><a name="zh-cn_topic_0000002262888689_p619211313217"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="13.570000000000002%" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p131921432328"><a name="zh-cn_topic_0000002262888689_p131921432328"></a><a name="zh-cn_topic_0000002262888689_p131921432328"></a><code>None</code>/<code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.8%" headers="mcps1.2.11.1.10 "><p id="zh-cn_topic_0000002262888689_p10192173203214"><a name="zh-cn_topic_0000002262888689_p10192173203214"></a><a name="zh-cn_topic_0000002262888689_p10192173203214"></a><code>float16</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row1219318318323"><td class="cellrowborder" valign="top" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p1519214313328"><a name="zh-cn_topic_0000002262888689_p1519214313328"></a><a name="zh-cn_topic_0000002262888689_p1519214313328"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p19192132324"><a name="zh-cn_topic_0000002262888689_p19192132324"></a><a name="zh-cn_topic_0000002262888689_p19192132324"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p11921438322"><a name="zh-cn_topic_0000002262888689_p11921438322"></a><a name="zh-cn_topic_0000002262888689_p11921438322"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p31924313327"><a name="zh-cn_topic_0000002262888689_p31924313327"></a><a name="zh-cn_topic_0000002262888689_p31924313327"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p419218312320"><a name="zh-cn_topic_0000002262888689_p419218312320"></a><a name="zh-cn_topic_0000002262888689_p419218312320"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p1519383163218"><a name="zh-cn_topic_0000002262888689_p1519383163218"></a><a name="zh-cn_topic_0000002262888689_p1519383163218"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p71933383215"><a name="zh-cn_topic_0000002262888689_p71933383215"></a><a name="zh-cn_topic_0000002262888689_p71933383215"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p17193153113212"><a name="zh-cn_topic_0000002262888689_p17193153113212"></a><a name="zh-cn_topic_0000002262888689_p17193153113212"></a><code>None</code>/<code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p51931438329"><a name="zh-cn_topic_0000002262888689_p51931438329"></a><a name="zh-cn_topic_0000002262888689_p51931438329"></a><code>bfloat16</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row1619323183216"><td class="cellrowborder" valign="top" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p51931312329"><a name="zh-cn_topic_0000002262888689_p51931312329"></a><a name="zh-cn_topic_0000002262888689_p51931312329"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p9193113143214"><a name="zh-cn_topic_0000002262888689_p9193113143214"></a><a name="zh-cn_topic_0000002262888689_p9193113143214"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p1119323153210"><a name="zh-cn_topic_0000002262888689_p1119323153210"></a><a name="zh-cn_topic_0000002262888689_p1119323153210"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p1193113193216"><a name="zh-cn_topic_0000002262888689_p1193113193216"></a><a name="zh-cn_topic_0000002262888689_p1193113193216"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p819315312327"><a name="zh-cn_topic_0000002262888689_p819315312327"></a><a name="zh-cn_topic_0000002262888689_p819315312327"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p61935373214"><a name="zh-cn_topic_0000002262888689_p61935373214"></a><a name="zh-cn_topic_0000002262888689_p61935373214"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p1019311323216"><a name="zh-cn_topic_0000002262888689_p1019311323216"></a><a name="zh-cn_topic_0000002262888689_p1019311323216"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p121934323212"><a name="zh-cn_topic_0000002262888689_p121934323212"></a><a name="zh-cn_topic_0000002262888689_p121934323212"></a><code>None</code>/<code>float32</code>（仅<code>x/weight/y</code>均为单张量）</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p51939318329"><a name="zh-cn_topic_0000002262888689_p51939318329"></a><a name="zh-cn_topic_0000002262888689_p51939318329"></a><code>float32</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row419318315325"><td class="cellrowborder" rowspan="3" valign="top" width="7.4700000000000015%" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p919393193215"><a name="zh-cn_topic_0000002262888689_p919393193215"></a><a name="zh-cn_topic_0000002262888689_p919393193215"></a>per-channel<span id="zh-cn_topic_0000002262888689_ph1079417343311"><a name="zh-cn_topic_0000002262888689_ph1079417343311"></a><a name="zh-cn_topic_0000002262888689_ph1079417343311"></a>q全</span>量化</p>
            <p id="zh-cn_topic_0000002262888689_p219353193210"><a name="zh-cn_topic_0000002262888689_p219353193210"></a><a name="zh-cn_topic_0000002262888689_p219353193210"></a></p>
            <p id="zh-cn_topic_0000002262888689_p919313310329"><a name="zh-cn_topic_0000002262888689_p919313310329"></a><a name="zh-cn_topic_0000002262888689_p919313310329"></a></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p219316323211"><a name="zh-cn_topic_0000002262888689_p219316323211"></a><a name="zh-cn_topic_0000002262888689_p219316323211"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.530000000000001%" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p171931363215"><a name="zh-cn_topic_0000002262888689_p171931363215"></a><a name="zh-cn_topic_0000002262888689_p171931363215"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.790000000000001%" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p419319319325"><a name="zh-cn_topic_0000002262888689_p419319319325"></a><a name="zh-cn_topic_0000002262888689_p419319319325"></a><code>int32</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.450000000000001%" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p819312363212"><a name="zh-cn_topic_0000002262888689_p819312363212"></a><a name="zh-cn_topic_0000002262888689_p819312363212"></a><code>int64</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="12.660000000000002%" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p1519323193218"><a name="zh-cn_topic_0000002262888689_p1519323193218"></a><a name="zh-cn_topic_0000002262888689_p1519323193218"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="13.180000000000003%" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p1193032326"><a name="zh-cn_topic_0000002262888689_p1193032326"></a><a name="zh-cn_topic_0000002262888689_p1193032326"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p131932312329"><a name="zh-cn_topic_0000002262888689_p131932312329"></a><a name="zh-cn_topic_0000002262888689_p131932312329"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="13.570000000000002%" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p3193439328"><a name="zh-cn_topic_0000002262888689_p3193439328"></a><a name="zh-cn_topic_0000002262888689_p3193439328"></a><code>None</code>/<code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.8%" headers="mcps1.2.11.1.10 "><p id="zh-cn_topic_0000002262888689_p141935353217"><a name="zh-cn_topic_0000002262888689_p141935353217"></a><a name="zh-cn_topic_0000002262888689_p141935353217"></a><code>int8</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row10193103123211"><td class="cellrowborder" valign="top" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p219313313211"><a name="zh-cn_topic_0000002262888689_p219313313211"></a><a name="zh-cn_topic_0000002262888689_p219313313211"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p101931931329"><a name="zh-cn_topic_0000002262888689_p101931931329"></a><a name="zh-cn_topic_0000002262888689_p101931931329"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p11931439320"><a name="zh-cn_topic_0000002262888689_p11931439320"></a><a name="zh-cn_topic_0000002262888689_p11931439320"></a><code>int32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p20193143133213"><a name="zh-cn_topic_0000002262888689_p20193143133213"></a><a name="zh-cn_topic_0000002262888689_p20193143133213"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p121931735322"><a name="zh-cn_topic_0000002262888689_p121931735322"></a><a name="zh-cn_topic_0000002262888689_p121931735322"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p1119317313216"><a name="zh-cn_topic_0000002262888689_p1119317313216"></a><a name="zh-cn_topic_0000002262888689_p1119317313216"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p1419314313321"><a name="zh-cn_topic_0000002262888689_p1419314313321"></a><a name="zh-cn_topic_0000002262888689_p1419314313321"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p1319383163213"><a name="zh-cn_topic_0000002262888689_p1319383163213"></a><a name="zh-cn_topic_0000002262888689_p1319383163213"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p1019310319322"><a name="zh-cn_topic_0000002262888689_p1019310319322"></a><a name="zh-cn_topic_0000002262888689_p1019310319322"></a><code>bfloat16</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row9194331321"><td class="cellrowborder" valign="top" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p4193532326"><a name="zh-cn_topic_0000002262888689_p4193532326"></a><a name="zh-cn_topic_0000002262888689_p4193532326"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p181939310321"><a name="zh-cn_topic_0000002262888689_p181939310321"></a><a name="zh-cn_topic_0000002262888689_p181939310321"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p10193336327"><a name="zh-cn_topic_0000002262888689_p10193336327"></a><a name="zh-cn_topic_0000002262888689_p10193336327"></a><code>int32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p20193123163211"><a name="zh-cn_topic_0000002262888689_p20193123163211"></a><a name="zh-cn_topic_0000002262888689_p20193123163211"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p181931311325"><a name="zh-cn_topic_0000002262888689_p181931311325"></a><a name="zh-cn_topic_0000002262888689_p181931311325"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p519343113216"><a name="zh-cn_topic_0000002262888689_p519343113216"></a><a name="zh-cn_topic_0000002262888689_p519343113216"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p3193839328"><a name="zh-cn_topic_0000002262888689_p3193839328"></a><a name="zh-cn_topic_0000002262888689_p3193839328"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p1519373183214"><a name="zh-cn_topic_0000002262888689_p1519373183214"></a><a name="zh-cn_topic_0000002262888689_p1519373183214"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p19194173163216"><a name="zh-cn_topic_0000002262888689_p19194173163216"></a><a name="zh-cn_topic_0000002262888689_p19194173163216"></a><code>float16</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row1819417323212"><td class="cellrowborder" rowspan="2" valign="top" width="7.4700000000000015%" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p1319410311327"><a name="zh-cn_topic_0000002262888689_p1319410311327"></a><a name="zh-cn_topic_0000002262888689_p1319410311327"></a>per-token<span id="zh-cn_topic_0000002262888689_ph15710104163117"><a name="zh-cn_topic_0000002262888689_ph15710104163117"></a><a name="zh-cn_topic_0000002262888689_ph15710104163117"></a>全</span>量化</p>
            <p id="zh-cn_topic_0000002262888689_p131942312325"><a name="zh-cn_topic_0000002262888689_p131942312325"></a><a name="zh-cn_topic_0000002262888689_p131942312325"></a></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p81941931323"><a name="zh-cn_topic_0000002262888689_p81941931323"></a><a name="zh-cn_topic_0000002262888689_p81941931323"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.530000000000001%" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p11194173113216"><a name="zh-cn_topic_0000002262888689_p11194173113216"></a><a name="zh-cn_topic_0000002262888689_p11194173113216"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.790000000000001%" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p1119416333215"><a name="zh-cn_topic_0000002262888689_p1119416333215"></a><a name="zh-cn_topic_0000002262888689_p1119416333215"></a><code>int32</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.450000000000001%" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p8194239321"><a name="zh-cn_topic_0000002262888689_p8194239321"></a><a name="zh-cn_topic_0000002262888689_p8194239321"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="12.660000000000002%" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p16194738320"><a name="zh-cn_topic_0000002262888689_p16194738320"></a><a name="zh-cn_topic_0000002262888689_p16194738320"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="13.180000000000003%" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p119420311329"><a name="zh-cn_topic_0000002262888689_p119420311329"></a><a name="zh-cn_topic_0000002262888689_p119420311329"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p1819415333215"><a name="zh-cn_topic_0000002262888689_p1819415333215"></a><a name="zh-cn_topic_0000002262888689_p1819415333215"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="13.570000000000002%" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p019413113213"><a name="zh-cn_topic_0000002262888689_p019413113213"></a><a name="zh-cn_topic_0000002262888689_p019413113213"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.8%" headers="mcps1.2.11.1.10 "><p id="zh-cn_topic_0000002262888689_p819418343219"><a name="zh-cn_topic_0000002262888689_p819418343219"></a><a name="zh-cn_topic_0000002262888689_p819418343219"></a><code>bfloat16</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row91948311324"><td class="cellrowborder" valign="top" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p1719473193217"><a name="zh-cn_topic_0000002262888689_p1719473193217"></a><a name="zh-cn_topic_0000002262888689_p1719473193217"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p1194143193211"><a name="zh-cn_topic_0000002262888689_p1194143193211"></a><a name="zh-cn_topic_0000002262888689_p1194143193211"></a><code>int8</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p1619413312322"><a name="zh-cn_topic_0000002262888689_p1619413312322"></a><a name="zh-cn_topic_0000002262888689_p1619413312322"></a><code>int32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p1819453153217"><a name="zh-cn_topic_0000002262888689_p1819453153217"></a><a name="zh-cn_topic_0000002262888689_p1819453153217"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p1194123183214"><a name="zh-cn_topic_0000002262888689_p1194123183214"></a><a name="zh-cn_topic_0000002262888689_p1194123183214"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p119419315324"><a name="zh-cn_topic_0000002262888689_p119419315324"></a><a name="zh-cn_topic_0000002262888689_p119419315324"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p119414303218"><a name="zh-cn_topic_0000002262888689_p119414303218"></a><a name="zh-cn_topic_0000002262888689_p119414303218"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p141945323219"><a name="zh-cn_topic_0000002262888689_p141945323219"></a><a name="zh-cn_topic_0000002262888689_p141945323219"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p1519417303217"><a name="zh-cn_topic_0000002262888689_p1519417303217"></a><a name="zh-cn_topic_0000002262888689_p1519417303217"></a><code>float16</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row819453143216"><td class="cellrowborder" rowspan="2" valign="top" width="7.4700000000000015%" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p3194138325"><a name="zh-cn_topic_0000002262888689_p3194138325"></a><a name="zh-cn_topic_0000002262888689_p3194138325"></a>伪量化</p>
            <p id="zh-cn_topic_0000002262888689_p17194103183219"><a name="zh-cn_topic_0000002262888689_p17194103183219"></a><a name="zh-cn_topic_0000002262888689_p17194103183219"></a></p>
            <p id="zh-cn_topic_0000002262888689_p1719420373214"><a name="zh-cn_topic_0000002262888689_p1719420373214"></a><a name="zh-cn_topic_0000002262888689_p1719420373214"></a></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.110000000000001%" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p1819419318324"><a name="zh-cn_topic_0000002262888689_p1819419318324"></a><a name="zh-cn_topic_0000002262888689_p1819419318324"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.530000000000001%" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p13194431327"><a name="zh-cn_topic_0000002262888689_p13194431327"></a><a name="zh-cn_topic_0000002262888689_p13194431327"></a><code>int8</code>/<code>int4</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.790000000000001%" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p1619419319328"><a name="zh-cn_topic_0000002262888689_p1619419319328"></a><a name="zh-cn_topic_0000002262888689_p1619419319328"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.450000000000001%" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p1919453153211"><a name="zh-cn_topic_0000002262888689_p1919453153211"></a><a name="zh-cn_topic_0000002262888689_p1919453153211"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="12.660000000000002%" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p619433143219"><a name="zh-cn_topic_0000002262888689_p619433143219"></a><a name="zh-cn_topic_0000002262888689_p619433143219"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="13.180000000000003%" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p11948310324"><a name="zh-cn_topic_0000002262888689_p11948310324"></a><a name="zh-cn_topic_0000002262888689_p11948310324"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="12.440000000000001%" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p919453193216"><a name="zh-cn_topic_0000002262888689_p919453193216"></a><a name="zh-cn_topic_0000002262888689_p919453193216"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="13.570000000000002%" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p519411312327"><a name="zh-cn_topic_0000002262888689_p519411312327"></a><a name="zh-cn_topic_0000002262888689_p519411312327"></a><code>None</code>/<code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="8.8%" headers="mcps1.2.11.1.10 "><p id="zh-cn_topic_0000002262888689_p2019417313210"><a name="zh-cn_topic_0000002262888689_p2019417313210"></a><a name="zh-cn_topic_0000002262888689_p2019417313210"></a><code>float16</code></p>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row119512319328"><td class="cellrowborder" valign="top" headers="mcps1.2.11.1.1 "><p id="zh-cn_topic_0000002262888689_p1519423153211"><a name="zh-cn_topic_0000002262888689_p1519423153211"></a><a name="zh-cn_topic_0000002262888689_p1519423153211"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.2 "><p id="zh-cn_topic_0000002262888689_p51945353217"><a name="zh-cn_topic_0000002262888689_p51945353217"></a><a name="zh-cn_topic_0000002262888689_p51945353217"></a><code>int8</code>/<code>int4</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.3 "><p id="zh-cn_topic_0000002262888689_p1519415313214"><a name="zh-cn_topic_0000002262888689_p1519415313214"></a><a name="zh-cn_topic_0000002262888689_p1519415313214"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.4 "><p id="zh-cn_topic_0000002262888689_p121947313211"><a name="zh-cn_topic_0000002262888689_p121947313211"></a><a name="zh-cn_topic_0000002262888689_p121947313211"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.5 "><p id="zh-cn_topic_0000002262888689_p1219419319323"><a name="zh-cn_topic_0000002262888689_p1219419319323"></a><a name="zh-cn_topic_0000002262888689_p1219419319323"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.6 "><p id="zh-cn_topic_0000002262888689_p1019412383220"><a name="zh-cn_topic_0000002262888689_p1019412383220"></a><a name="zh-cn_topic_0000002262888689_p1019412383220"></a><code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.7 "><p id="zh-cn_topic_0000002262888689_p1519412383210"><a name="zh-cn_topic_0000002262888689_p1519412383210"></a><a name="zh-cn_topic_0000002262888689_p1519412383210"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.8 "><p id="zh-cn_topic_0000002262888689_p13195434323"><a name="zh-cn_topic_0000002262888689_p13195434323"></a><a name="zh-cn_topic_0000002262888689_p13195434323"></a><code>None</code>/<code>bfloat16</code></p>
            </td>
            <td class="cellrowborder" valign="top" headers="mcps1.2.11.1.9 "><p id="zh-cn_topic_0000002262888689_p191951130324"><a name="zh-cn_topic_0000002262888689_p191951130324"></a><a name="zh-cn_topic_0000002262888689_p191951130324"></a><code>bfloat16</code></p>
            </td>
            </tr>
            </tbody>
            </table>

            >**说明：**<br> 
            >-   伪量化场景，若`weight`的类型为`int8`，仅支持per-channel模式；若`weight`的类型为`int4`，支持per-channel和per-group两种模式。若为per-group，per-group数$G$或$G_i$必须要能整除对应的$k_i$。若`weight`为多tensor，定义per-group长度$s_i= k_i/G_i$，要求所有$s_i(i=1,2,...g)$都相等。
            >-   伪量化场景，若`weight`的类型为`int4`，则`weight`中每一组tensor的最后一维大小都应是偶数。<code>weight<sub>i</sub></code>的最后一维指`weight`不转置时<code>weight<sub>i</sub></code>的N轴或当weight转置时weight<sub>i</sub>的$K$轴。并且在per-group场景下，当`weight`转置时，要求per-group长度$s_i$是偶数。tensor转置：指若tensor shape为$[M,K]$时，则stride为$[1,M]$,数据排布为$[K,M]$的场景，即非连续tensor。
            >-   当前PyTorch不支持`int4`类型数据，需要使用时可以通过[torch\_npu.npu\_quantize](torch_npu-npu_quantize.md)接口使用`int32`数据表示`int4`。

        -   <term>Atlas 推理系列产品</term>：

            **表 3**  数据类型约束

            <a name="zh-cn_topic_0000002262888689_table819793193212"></a>
            <table><thead align="left"><tr id="zh-cn_topic_0000002262888689_row6197193133217"><th class="cellrowborder" valign="top" width="9.63096309630963%" id="mcps1.2.10.1.1"><p id="zh-cn_topic_0000002262888689_p111971536329"><a name="zh-cn_topic_0000002262888689_p111971536329"></a><a name="zh-cn_topic_0000002262888689_p111971536329"></a>x</p>
            </th>
            <th class="cellrowborder" valign="top" width="9.450945094509452%" id="mcps1.2.10.1.2"><p id="zh-cn_topic_0000002262888689_p51971032321"><a name="zh-cn_topic_0000002262888689_p51971032321"></a><a name="zh-cn_topic_0000002262888689_p51971032321"></a>weight</p>
            </th>
            <th class="cellrowborder" valign="top" width="9.92099209920992%" id="mcps1.2.10.1.3"><p id="zh-cn_topic_0000002262888689_p419717319323"><a name="zh-cn_topic_0000002262888689_p419717319323"></a><a name="zh-cn_topic_0000002262888689_p419717319323"></a>bias</p>
            </th>
            <th class="cellrowborder" valign="top" width="7.950795079507951%" id="mcps1.2.10.1.4"><p id="zh-cn_topic_0000002262888689_p12197123103210"><a name="zh-cn_topic_0000002262888689_p12197123103210"></a><a name="zh-cn_topic_0000002262888689_p12197123103210"></a>scale</p>
            </th>
            <th class="cellrowborder" valign="top" width="14.24142414241424%" id="mcps1.2.10.1.5"><p id="zh-cn_topic_0000002262888689_p2019720310324"><a name="zh-cn_topic_0000002262888689_p2019720310324"></a><a name="zh-cn_topic_0000002262888689_p2019720310324"></a>antiquant_scale</p>
            </th>
            <th class="cellrowborder" valign="top" width="13.221322132213222%" id="mcps1.2.10.1.6"><p id="zh-cn_topic_0000002262888689_p619713315329"><a name="zh-cn_topic_0000002262888689_p619713315329"></a><a name="zh-cn_topic_0000002262888689_p619713315329"></a>antiquant_offset</p>
            </th>
            <th class="cellrowborder" valign="top" width="13.57135713571357%" id="mcps1.2.10.1.7"><p id="zh-cn_topic_0000002262888689_p719719316326"><a name="zh-cn_topic_0000002262888689_p719719316326"></a><a name="zh-cn_topic_0000002262888689_p719719316326"></a>per_token_scale</p>
            </th>
            <th class="cellrowborder" valign="top" width="11.221122112211221%" id="mcps1.2.10.1.8"><p id="zh-cn_topic_0000002262888689_p319763103219"><a name="zh-cn_topic_0000002262888689_p319763103219"></a><a name="zh-cn_topic_0000002262888689_p319763103219"></a>output_dtype</p>
            </th>
            <th class="cellrowborder" valign="top" width="10.791079107910791%" id="mcps1.2.10.1.9"><p id="zh-cn_topic_0000002262888689_p71974343210"><a name="zh-cn_topic_0000002262888689_p71974343210"></a><a name="zh-cn_topic_0000002262888689_p71974343210"></a>y</p>
            </th>
            </tr>
            </thead>
            <tbody><tr id="zh-cn_topic_0000002262888689_row11976311325"><td class="cellrowborder" valign="top" width="9.63096309630963%" headers="mcps1.2.10.1.1 "><p id="zh-cn_topic_0000002262888689_p1019710373214"><a name="zh-cn_topic_0000002262888689_p1019710373214"></a><a name="zh-cn_topic_0000002262888689_p1019710373214"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="9.450945094509452%" headers="mcps1.2.10.1.2 "><p id="zh-cn_topic_0000002262888689_p201977313212"><a name="zh-cn_topic_0000002262888689_p201977313212"></a><a name="zh-cn_topic_0000002262888689_p201977313212"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="9.92099209920992%" headers="mcps1.2.10.1.3 "><p id="zh-cn_topic_0000002262888689_p181977393218"><a name="zh-cn_topic_0000002262888689_p181977393218"></a><a name="zh-cn_topic_0000002262888689_p181977393218"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="7.950795079507951%" headers="mcps1.2.10.1.4 "><p id="zh-cn_topic_0000002262888689_p19197173123211"><a name="zh-cn_topic_0000002262888689_p19197173123211"></a><a name="zh-cn_topic_0000002262888689_p19197173123211"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="14.24142414241424%" headers="mcps1.2.10.1.5 "><p id="zh-cn_topic_0000002262888689_p1319710333218"><a name="zh-cn_topic_0000002262888689_p1319710333218"></a><a name="zh-cn_topic_0000002262888689_p1319710333218"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="13.221322132213222%" headers="mcps1.2.10.1.6 "><p id="zh-cn_topic_0000002262888689_p191975314327"><a name="zh-cn_topic_0000002262888689_p191975314327"></a><a name="zh-cn_topic_0000002262888689_p191975314327"></a>无需赋值</p>
            </td>
            <td class="cellrowborder" valign="top" width="13.57135713571357%" headers="mcps1.2.10.1.7 "><p id="zh-cn_topic_0000002262888689_p12197183103210"><a name="zh-cn_topic_0000002262888689_p12197183103210"></a><a name="zh-cn_topic_0000002262888689_p12197183103210"></a><code>float32</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="11.221122112211221%" headers="mcps1.2.10.1.8 "><p id="zh-cn_topic_0000002262888689_p21971831326"><a name="zh-cn_topic_0000002262888689_p21971831326"></a><a name="zh-cn_topic_0000002262888689_p21971831326"></a><code>float16</code></p>
            </td>
            <td class="cellrowborder" valign="top" width="10.791079107910791%" headers="mcps1.2.10.1.9 "><p id="zh-cn_topic_0000002262888689_p10197153193216"><a name="zh-cn_topic_0000002262888689_p10197153193216"></a><a name="zh-cn_topic_0000002262888689_p10197153193216"></a><code>float16</code></p>
            </td>
            </tr>
            </tbody>
            </table>

-   根据输入`x`、输入`weight`与输出`y`的Tensor数量不同，支持以下几种场景。场景中的“单”表示单个张量，“多”表示多个张量。场景顺序为`x`、`weight`、`y`，例如“单多单”表示`x`为单张量，`weight`为多张量，`y`为单张量。
    -   **`group_list`输入类型为`List[int]`时**，<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>各场景的限制。

        <a name="zh-cn_topic_0000002262888689_table159001938125715"></a>
        <table><thead align="left"><tr id="zh-cn_topic_0000002262888689_row128990386575"><th class="cellrowborder" valign="top" width="10.51105110511051%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002262888689_p1089963865717"><a name="zh-cn_topic_0000002262888689_p1089963865717"></a><a name="zh-cn_topic_0000002262888689_p1089963865717"></a>支持场景</p>
        </th>
        <th class="cellrowborder" valign="top" width="24.71247124712471%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002262888689_p13899338185716"><a name="zh-cn_topic_0000002262888689_p13899338185716"></a><a name="zh-cn_topic_0000002262888689_p13899338185716"></a>场景说明</p>
        </th>
        <th class="cellrowborder" valign="top" width="64.77647764776478%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002262888689_p88991238105712"><a name="zh-cn_topic_0000002262888689_p88991238105712"></a><a name="zh-cn_topic_0000002262888689_p88991238105712"></a>场景限制</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="zh-cn_topic_0000002262888689_row14899738115719"><td class="cellrowborder" valign="top" width="10.51105110511051%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002262888689_p1889983825713"><a name="zh-cn_topic_0000002262888689_p1889983825713"></a><a name="zh-cn_topic_0000002262888689_p1889983825713"></a>多多多</p>
        </td>
        <td class="cellrowborder" valign="top" width="24.71247124712471%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002262888689_p127892451588"><a name="zh-cn_topic_0000002262888689_p127892451588"></a><a name="zh-cn_topic_0000002262888689_p127892451588"></a><code>x</code>和<code>weight</code>为多张量，<code>y</code>为多张量。每组数据的张量是独立的。</p>
        </td>
        <td class="cellrowborder" valign="top" width="64.77647764776478%" headers="mcps1.1.4.1.3 "><a name="zh-cn_topic_0000002262888689_ol17899133811579"></a><a name="zh-cn_topic_0000002262888689_ol17899133811579"></a><ol id="zh-cn_topic_0000002262888689_ol17899133811579"><li>仅支持<code>split_item</code>为0或1。</li><li><code>x</code>中tensor要求维度一致且支持2-6维，<code>weight</code>中tensor需为2维，<code>y</code>中tensor维度和x保持一致。</li><li><code>x</code>中tensor大于2维，<code>group_list</code>必须传空。</li><li><code>x</code>中tensor为2维且传入<code>group_list</code>，<code>group_list</code>的差值需与<code>x</code>中tensor的第一维一一对应。</li></ol>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000002262888689_row1389933815577"><td class="cellrowborder" valign="top" width="10.51105110511051%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002262888689_p1899938135717"><a name="zh-cn_topic_0000002262888689_p1899938135717"></a><a name="zh-cn_topic_0000002262888689_p1899938135717"></a>单多单</p>
        </td>
        <td class="cellrowborder" valign="top" width="24.71247124712471%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002262888689_p560545515811"><a name="zh-cn_topic_0000002262888689_p560545515811"></a><a name="zh-cn_topic_0000002262888689_p560545515811"></a><code>x</code>为单张量，<code>weight</code>为多张量，<code>y</code>为单张量。</p>
        </td>
        <td class="cellrowborder" valign="top" width="64.77647764776478%" headers="mcps1.1.4.1.3 "><a name="zh-cn_topic_0000002262888689_ol48991388572"></a><a name="zh-cn_topic_0000002262888689_ol48991388572"></a><ol id="zh-cn_topic_0000002262888689_ol48991388572"><li>仅支持<code>split_item</code>为2或3。</li><li>必须传<code>group_list</code>，且最后一个值与x中tensor的第一维相等。</li><li><code>x</code>、<code>weight</code>、<code>y</code>中tensor需为2维。</li><li><code>weight</code>中每个tensor的N轴必须相等。</li></ol>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000002262888689_row2900183875718"><td class="cellrowborder" valign="top" width="10.51105110511051%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002262888689_p198994383577"><a name="zh-cn_topic_0000002262888689_p198994383577"></a><a name="zh-cn_topic_0000002262888689_p198994383577"></a>单多多</p>
        </td>
        <td class="cellrowborder" valign="top" width="24.71247124712471%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002262888689_p148991338115716"><a name="zh-cn_topic_0000002262888689_p148991338115716"></a><a name="zh-cn_topic_0000002262888689_p148991338115716"></a><code>x</code>为单张量，<code>weight</code>为多张量，<code>y</code>为多张量。</p>
        </td>
        <td class="cellrowborder" valign="top" width="64.77647764776478%" headers="mcps1.1.4.1.3 "><a name="zh-cn_topic_0000002262888689_ol49001138205714"></a><a name="zh-cn_topic_0000002262888689_ol49001138205714"></a><ol id="zh-cn_topic_0000002262888689_ol49001138205714"><li>仅支持<code>split_item</code>为0或1。</li><li>必须传<code>group_list</code>，<code>group_list</code>的差值需与<code>y</code>中tensor的第一维一一对应。</li><li><code>x</code>、<code>weight</code>、<code>y</code>中tensor需为2维。</li></ol>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000002262888689_row9900103825714"><td class="cellrowborder" valign="top" width="10.51105110511051%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002262888689_p13900193835715"><a name="zh-cn_topic_0000002262888689_p13900193835715"></a><a name="zh-cn_topic_0000002262888689_p13900193835715"></a>多多单</p>
        </td>
        <td class="cellrowborder" valign="top" width="24.71247124712471%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002262888689_p18599172041210"><a name="zh-cn_topic_0000002262888689_p18599172041210"></a><a name="zh-cn_topic_0000002262888689_p18599172041210"></a><code>x</code>和<code>weight</code>为多张量，<code>y</code>为单张量。每组矩阵乘法的结果连续存放在同一个张量中。</p>
        </td>
        <td class="cellrowborder" valign="top" width="64.77647764776478%" headers="mcps1.1.4.1.3 "><a name="zh-cn_topic_0000002262888689_ol12900138155714"></a><a name="zh-cn_topic_0000002262888689_ol12900138155714"></a><ol id="zh-cn_topic_0000002262888689_ol12900138155714"><li>仅支持<code>split_item</code>为2或3。</li><li><code>x</code>、<code>weight</code>、<code>y</code>中tensor需为2维。</li><li><code>weight</code>中每个tensor的N轴必须相等。</li><li>若传入<code>group_list</code>，<code>group_list</code>的差值需与<code>x</code>中tensor的第一维一一对应。</li></ol>
        </td>
        </tr>
        </tbody>
        </table>

    -   **`group_list`输入类型为`Tensor`时**，各场景的限制。
        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

            >**说明：**<br> 
            >-   量化、伪量化仅支持`group_type`为-1和0场景。
            >-   仅pertoken量化场景支持激活函数计算。

            <a name="zh-cn_topic_0000002262888689_table154441543199"></a>
            <table><thead align="left"><tr id="zh-cn_topic_0000002262888689_row34435413192"><th class="cellrowborder" valign="top" width="10.23%" id="mcps1.1.5.1.1"><p id="zh-cn_topic_0000002262888689_p164433412191"><a name="zh-cn_topic_0000002262888689_p164433412191"></a><a name="zh-cn_topic_0000002262888689_p164433412191"></a>group_type</p>
            </th>
            <th class="cellrowborder" valign="top" width="8.780000000000001%" id="mcps1.1.5.1.2"><p id="zh-cn_topic_0000002262888689_p4443545199"><a name="zh-cn_topic_0000002262888689_p4443545199"></a><a name="zh-cn_topic_0000002262888689_p4443545199"></a>支持场景</p>
            </th>
            <th class="cellrowborder" valign="top" width="18.34%" id="mcps1.1.5.1.3"><p id="zh-cn_topic_0000002262888689_p184431411911"><a name="zh-cn_topic_0000002262888689_p184431411911"></a><a name="zh-cn_topic_0000002262888689_p184431411911"></a>场景说明</p>
            </th>
            <th class="cellrowborder" valign="top" width="62.64999999999999%" id="mcps1.1.5.1.4"><p id="zh-cn_topic_0000002262888689_p5443744195"><a name="zh-cn_topic_0000002262888689_p5443744195"></a><a name="zh-cn_topic_0000002262888689_p5443744195"></a>场景限制</p>
            </th>
            </tr>
            </thead>
            <tbody><tr id="zh-cn_topic_0000002262888689_row13443144121914"><td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p13443104111917"><a name="zh-cn_topic_0000002262888689_p13443104111917"></a><a name="zh-cn_topic_0000002262888689_p13443104111917"></a>-1</p>
            </td>
            <td class="cellrowborder" valign="top" width="8.780000000000001%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p1944344141910"><a name="zh-cn_topic_0000002262888689_p1944344141910"></a><a name="zh-cn_topic_0000002262888689_p1944344141910"></a>多多多</p>
            </td>
            <td class="cellrowborder" valign="top" width="18.34%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p19935447499"><a name="zh-cn_topic_0000002262888689_p19935447499"></a><a name="zh-cn_topic_0000002262888689_p19935447499"></a><code>x</code>和<code>weight</code>为多张量，<code>y</code>为多张量。每组数据的张量是独立的。</p>
            </td>
            <td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000002262888689_ol44436419196"></a><a name="zh-cn_topic_0000002262888689_ol44436419196"></a><ol id="zh-cn_topic_0000002262888689_ol44436419196"><li>仅支持<code>split_item</code>为0或1。</li><li><code>x</code>中tensor要求维度一致且支持2-6维，<code>weight</code>中tensor需为2维，<code>y</code>中tensor维度和x保持一致<span id="zh-cn_topic_0000002262888689_ph164382454812"><a name="zh-cn_topic_0000002262888689_ph164382454812"></a><a name="zh-cn_topic_0000002262888689_ph164382454812"></a>。</span></li><li><code>group_list</code>必须传空。</li><li>支持<code>weight</code>转置，但<code>weight</code>中每个tensor是否转置需保持统一。</li><li><code>x</code>不支持转置。</li></ol>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row744364101913"><td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p344314418195"><a name="zh-cn_topic_0000002262888689_p344314418195"></a><a name="zh-cn_topic_0000002262888689_p344314418195"></a>0</p>
            </td>
            <td class="cellrowborder" valign="top" width="8.780000000000001%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p244318413194"><a name="zh-cn_topic_0000002262888689_p244318413194"></a><a name="zh-cn_topic_0000002262888689_p244318413194"></a>单单单</p>
            </td>
            <td class="cellrowborder" valign="top" width="18.34%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p6443446198"><a name="zh-cn_topic_0000002262888689_p6443446198"></a><a name="zh-cn_topic_0000002262888689_p6443446198"></a><code>x</code>、<code>weight</code>与<code>y</code>均为单张量。</p>
            </td>
            <td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000002262888689_ol154431742190"></a><a name="zh-cn_topic_0000002262888689_ol154431742190"></a><ol id="zh-cn_topic_0000002262888689_ol154431742190"><li>仅支持<code>split_item</code>为2或3。</li><li><code>weight</code>中tensor需为3维，<code>x</code>、<code>y</code>中tensor需为2维。</li><li>必须传<code>group_list</code>，且当<code>group_list_type</code>为0时，最后一个值与<code>x</code>中tensor的第一维相等，当<code>group_list_type</code>为1时，数值的总和与<code>x</code>中tensor的第一维相等，当<code>group_list_type</code>为2时，第二列数值的总和与<code>x</code>中tensor的第一维相等。</li><li><code>group_list</code>第1维最大支持1024，即最多支持1024个group。</li><li>支持<code>weight</code>转置。</li><li><code>x</code>不支持转置。</li></ol>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row1844484161919"><td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p144432047193"><a name="zh-cn_topic_0000002262888689_p144432047193"></a><a name="zh-cn_topic_0000002262888689_p144432047193"></a>0</p>
            </td>
            <td class="cellrowborder" valign="top" width="8.780000000000001%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p1044324111920"><a name="zh-cn_topic_0000002262888689_p1044324111920"></a><a name="zh-cn_topic_0000002262888689_p1044324111920"></a>单多单</p>
            </td>
            <td class="cellrowborder" valign="top" width="18.34%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p62831854134915"><a name="zh-cn_topic_0000002262888689_p62831854134915"></a><a name="zh-cn_topic_0000002262888689_p62831854134915"></a><code>x</code>为单张量，<code>weight</code>为多张量，<code>y</code>为单张量。</p>
            </td>
            <td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000002262888689_ol444415471920"></a><a name="zh-cn_topic_0000002262888689_ol444415471920"></a><ol id="zh-cn_topic_0000002262888689_ol444415471920"><li>仅支持<code>split_item</code>为2或3。</li><li>必须传<code>group_list</code>，且当<code>group_list_type</code>为0时，最后一个值与x中tensor的第一维相等，当<code>group_list_type</code>为1时，数值的总和与<code>x</code>中tensor的第一维相等且长度最大为128，当<code>group_list_type</code>为2时，第二列数值的总和与<code>x</code>中tensor的第一维相等且长度最大为128。</li><li><code>x</code>、<code>weight</code>、<code>y</code>中tensor需为2维。</li><li><code>weight</code>中每个tensor的N轴必须相等。</li><li>支持<code>weight</code>转置，但<code>weight</code>中每个tensor是否转置需保持统一。</li><li><code>x</code>不支持转置。</li></ol>
            </td>
            </tr>
            <tr id="zh-cn_topic_0000002262888689_row1144464101918"><td class="cellrowborder" valign="top" width="10.23%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p1344434131910"><a name="zh-cn_topic_0000002262888689_p1344434131910"></a><a name="zh-cn_topic_0000002262888689_p1344434131910"></a>0</p>
            </td>
            <td class="cellrowborder" valign="top" width="8.780000000000001%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p16444164181920"><a name="zh-cn_topic_0000002262888689_p16444164181920"></a><a name="zh-cn_topic_0000002262888689_p16444164181920"></a>多多单</p>
            </td>
            <td class="cellrowborder" valign="top" width="18.34%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p164775185016"><a name="zh-cn_topic_0000002262888689_p164775185016"></a><a name="zh-cn_topic_0000002262888689_p164775185016"></a><code>x</code>和<code>weight</code>为多张量，<code>y</code>为单张量。每组矩阵乘法的结果连续存放在同一个张量中。</p>
            </td>
            <td class="cellrowborder" valign="top" width="62.64999999999999%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000002262888689_ol24442412194"></a><a name="zh-cn_topic_0000002262888689_ol24442412194"></a><ol id="zh-cn_topic_0000002262888689_ol24442412194"><li>仅支持<code>split_item</code>为2或3。</li><li><code>x</code>、<code>weight</code>、<code>y</code>中tensor需为2维。</li><li><code>weight</code>中每个tensor的N轴必须相等。</li><li>若传入<code>group_list</code>，当<code>group_list_type</code>为0时，<code>group_list</code>的差值需与<code>x</code>中tensor的第一维一一对应，当<code>group_list_type</code>为1时，<code>group_list</code>的数值需与<code>x</code>中tensor的第一维一一对应且长度最大为128，当<code>group_list_type</code>为2时，<code>group_list</code>第二列的数值需与<code>x</code>中tensor的第一维一一对应且长度最大为128。</li><li>支持<code>weight</code>转置，但<code>weight</code>中每个tensor是否转置需保持统一。</li><li><code>x</code>不支持转置。</li></ol>
            </td>
            </tbody>
            </table>

        -   <term>Atlas 推理系列产品</term>：

            输入输出只支持`float16`的数据类型，输出`y`的n轴大小需要是16的倍数。

            <a name="zh-cn_topic_0000002262888689_table2044519410191"></a>
            <table><thead align="left"><tr id="zh-cn_topic_0000002262888689_row10445541194"><th class="cellrowborder" valign="top" width="10.48%" id="mcps1.1.5.1.1"><p id="zh-cn_topic_0000002262888689_p44448491914"><a name="zh-cn_topic_0000002262888689_p44448491914"></a><a name="zh-cn_topic_0000002262888689_p44448491914"></a>group_type</p>
            </th>
            <th class="cellrowborder" valign="top" width="8.82%" id="mcps1.1.5.1.2"><p id="zh-cn_topic_0000002262888689_p1744512414193"><a name="zh-cn_topic_0000002262888689_p1744512414193"></a><a name="zh-cn_topic_0000002262888689_p1744512414193"></a>支持场景</p>
            </th>
            <th class="cellrowborder" valign="top" width="19.43%" id="mcps1.1.5.1.3"><p id="zh-cn_topic_0000002262888689_p5923112217503"><a name="zh-cn_topic_0000002262888689_p5923112217503"></a><a name="zh-cn_topic_0000002262888689_p5923112217503"></a>场景说明</p>
            </th>
            <th class="cellrowborder" valign="top" width="61.27%" id="mcps1.1.5.1.4"><p id="zh-cn_topic_0000002262888689_p84456421912"><a name="zh-cn_topic_0000002262888689_p84456421912"></a><a name="zh-cn_topic_0000002262888689_p84456421912"></a>场景限制</p>
            </th>
            </tr>
            </thead>
            <tbody><tr id="zh-cn_topic_0000002262888689_row54455471913"><td class="cellrowborder" valign="top" width="10.48%" headers="mcps1.1.5.1.1 "><p id="zh-cn_topic_0000002262888689_p844517431912"><a name="zh-cn_topic_0000002262888689_p844517431912"></a><a name="zh-cn_topic_0000002262888689_p844517431912"></a>0</p>
            </td>
            <td class="cellrowborder" valign="top" width="8.82%" headers="mcps1.1.5.1.2 "><p id="zh-cn_topic_0000002262888689_p444517471917"><a name="zh-cn_topic_0000002262888689_p444517471917"></a><a name="zh-cn_topic_0000002262888689_p444517471917"></a>单单单</p>
            </td>
            <td class="cellrowborder" valign="top" width="19.43%" headers="mcps1.1.5.1.3 "><p id="zh-cn_topic_0000002262888689_p2181155195014"><a name="zh-cn_topic_0000002262888689_p2181155195014"></a><a name="zh-cn_topic_0000002262888689_p2181155195014"></a><code>x</code>、<code>weight</code>与<code>y</code>均为单张量</p>
            </td>
            <td class="cellrowborder" valign="top" width="61.27%" headers="mcps1.1.5.1.4 "><a name="zh-cn_topic_0000002262888689_ol9445448197"></a><a name="zh-cn_topic_0000002262888689_ol9445448197"></a><ol id="zh-cn_topic_0000002262888689_ol9445448197"><li>仅支持<code>split_item</code>为2或3。</li><li><code>weight</code>中tensor需为3维，<code>x</code>、<code>y</code>中tensor需为2维。</li><li>必须传<code>group_list</code>，且当<code>group_list_type</code>为0时，最后一个值与x中tensor的第一维相等，当<code>group_list_type</code>为1时，数值的总和与x中tensor的第一维相等。</li><li><code>group_list</code>第1维最大支持1024，即最多支持1024个group。</li><li>支持<code>weight</code>转置，不支持<code>x</code>转置。</li></ol>
            </td>
            </tr>
            </tbody>
            </table>

## 支持的型号<a name="zh-cn_topic_0000002262888689_section1185202694112"></a>

-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
-   <term>Atlas 推理系列产品</term>

## 调用示例<a name="zh-cn_topic_0000002262888689_section1566973054111"></a>

-   单算子模式调用

    通用调用示例

    ```python
    import torch
    import torch_npu
    
    x1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
    x2 = torch.randn(1024, 256, device='npu', dtype=torch.float16)
    x3 = torch.randn(512, 1024, device='npu', dtype=torch.float16)
    x = [x1, x2, x3]
    
    weight1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
    weight2 = torch.randn(256, 1024, device='npu', dtype=torch.float16)
    weight3 = torch.randn(1024, 128, device='npu', dtype=torch.float16)
    weight = [weight1, weight2, weight3]
    
    bias1 = torch.randn(256, device='npu', dtype=torch.float16)
    bias2 = torch.randn(1024, device='npu', dtype=torch.float16)
    bias3 = torch.randn(128, device='npu', dtype=torch.float16)
    bias = [bias1, bias2, bias3]
    
    group_list = None
    split_item = 0
    npu_out = torch_npu.npu_grouped_matmul(x, weight, bias=bias, group_list=group_list, split_item=split_item, group_type=-1)
    ```

-   图模式调用
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

        ```python
        import torch
        import torch.nn as nn
        import torch_npu
        import torchair as tng
        from torchair.configs.compiler_config import CompilerConfig
        
        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)
        
        class GMMModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, weight):
                return torch_npu.npu_grouped_matmul(x, weight, group_type=-1)
        
        def main():
            x1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
            x2 = torch.randn(1024, 256, device='npu', dtype=torch.float16)
            x3 = torch.randn(512, 1024, device='npu', dtype=torch.float16)
            x = [x1, x2, x3]
            
            weight1 = torch.randn(256, 256, device='npu', dtype=torch.float16)
            weight2 = torch.randn(256, 1024, device='npu', dtype=torch.float16)
            weight3 = torch.randn(1024, 128, device='npu', dtype=torch.float16)
            weight = [weight1, weight2, weight3]
            
            model = GMMModel().npu()
            model = torch.compile(model, backend=npu_backend, dynamic=False)
            custom_output = model(x, weight)
        
        if __name__ == '__main__':
            main()
        ```

