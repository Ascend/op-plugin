# torch_npu.npu_grouped_matmul<a name="ZH-CN_TOPIC_0000002229788810"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |
|<term>Atlas 推理系列产品</term>    | √  |

## 功能说明<a name="zh-cn_topic_0000002262888689_section1290611593405"></a>

-   API功能：`npu_grouped_matmul`是一种对多个矩阵乘法（matmul）操作进行分组计算的高效方法。该API实现了对多个矩阵乘法操作的批量处理，通过将具有相同形状或相似形状的矩阵乘法操作组合在一起，减少内存访问开销和计算资源的浪费，从而提高计算效率。

-   计算公式：

     公式中$@$符号表示矩阵乘法，$\times$符号表示矩阵Hadamard乘积：

    -   非量化场景（公式1）：

        $y_i = x_i @ weight_i + bias_i$

    -   perchannel、pertensor量化场景（公式2）：

        $y_i = (x_i @ weight_i) \times scale_i + offset_i$

        -   `x`为`int8`输入，`bias`为`int32`输入（公式2-1）：

            $y_i = (x_i @ weight_i + bias_i) \times scale_i + offset_i$

        -   `x`为`int8`输入，`bias`为`bfloat16`、`float16`、`float32`输入，无offset（公式2-2）：

            $y_i = (x_i @ weight_i) \times scale_i + bias_i$

    -   pertoken、pertensor+pertensor、pertensor+perchannel量化场景（公式3）：

        $y_i = (x_i @ weight_i + bias_i) \times scale_i \times pertokenscale_i$

        -   `x`为`int8`输入，bias为`int32`输入（公式3-1）：

            $y_i = (x_i @ weight_i + bias_i) \times scale_i \times pertokenscale_i$

        -   `x`为`int8`输入，`bias`为`bfloat16`，`float16`，`float32`输入（公式3-2）：

            $y_i = (x_i @ weight_i) \times scale_i \times pertokenscale_i + bias_i$
        -   `x`为`int4`输入, `weight`的数据类型为`int4`，数据排布格式为`NZ`的输入（公式3-3）:

            $y_i=x_i@ (weight_i \times scale_i) \times pertokenscale_i$
        

    -   伪量化场景（公式4）：

        $y_i = x_i @ ((weight_i + antiquant\_offset_i) \times antiquant\_scale_i) + bias_i$

## 函数原型<a name="zh-cn_topic_0000002262888689_section87878612417"></a>
```
npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=None, group_list_type=0, act_type=0, output_dtype=None, tuning_config=None) -> List[Tensor]
```

## 参数说明<a name="zh-cn_topic_0000002262888689_section135561610204110"></a>

- **x** (`List[Tensor]`)：必选参数。输入矩阵列表，表示矩阵乘法中的左矩阵。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`float16`、`float32`、`bfloat16`、`int8`和`int4`。
        -   <term>Atlas 推理系列产品</term>：`float16`。

    -   列表最大长度为128。
    -   当split\_item=0时，张量支持2至6维输入；其他情况下，张量仅支持2维输入。

- **weight** (`List[Tensor]`)：必选参数。权重矩阵列表，表示矩阵乘法中的右矩阵。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
            -   当`group_list`输入类型为`List[int]`时，支持`float16`、`float32`、`bfloat16`和`int8`。
            -   当`group_list`输入类型为`Tensor`时，支持`float16`、`float32`、`bfloat16`、`int4`和`int8`。

        -   <term>Atlas 推理系列产品</term>：`float16`。

    -   列表最大长度为128。
    -   每个张量支持2维或3维输入。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **bias** (`List[Tensor]`)：可选参数。每个分组的矩阵乘法输出的独立偏置项。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`float16`、`float32`和`int32`。
        -   <term>Atlas 推理系列产品</term>：`float16`。

    -   列表长度与weight列表长度相同。
    -   每个张量仅支持1维输入。

- **scale** (`List[Tensor]`)：可选参数。用于缩放原数值以匹配量化后的范围值，代表量化参数中的缩放因子，对应公式（2）、公式（3）。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
            -   当`group_list`输入类型为`List[int]`时，支持`int64`。
            -   当`group_list`输入类型为`Tensor`时，支持`float32`、`bfloat16`和`int64`。

        -   <term>Atlas 推理系列产品</term>：仅支持传入`None`。

    -   列表长度与weight列表长度相同。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：每个张量仅支持1维输入。

- **offset** (`List[Tensor]`)：可选参数。用于调整量化后的数值偏移量，从而更准确地表示原始浮点数值，对应公式（2）。当前仅支持传入`None`。
- **antiquant_scale** (`List[Tensor]`)：可选参数。用于缩放原数值以匹配伪量化后的范围值，代表伪量化参数中的缩放因子，对应公式（4）。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`float16`、`bfloat16`。
        -   <term>Atlas 推理系列产品</term>：仅支持传入`None`。

    -   列表长度与weight列表长度相同。
    -   每个张量支持输入维度如下（其中$g$为matmul组数，$G$为pergroup数，$G_i$为第i个tensor的pergroup数）：
        -   伪量化perchannel场景，`weight`为单tensor时，shape限制为$[g, n]$；`weight`为多tensor时，shape限制为$[n_i]$。
        -   伪量化pergroup场景，weight为单tensor时，shape限制为$[g, G, n]$; weight为多tensor时，shape限制为$[G_i, n_i]$。

- **antiquant_offset** (`List[Tensor]`)：可选参数。用于调整伪量化后的数值偏移量，从而更准确地表示原始浮点数值，对应公式（4）。
    -   支持的数据类型如下：
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`float16`、`bfloat16`。
        -   <term>Atlas 推理系列产品</term>：仅支持传入`None`。

    -   列表长度与`weight`列表长度相同。
    -   每个张量输入维度和`antiquant_scale`输入维度一致。

- **per_token_scale** (`List[Tensor]`)：可选参数。用于缩放原数值以匹配量化后的范围值，代表pertoken量化参数中由`x`量化引入的缩放因子，对应公式（3）和公式（5）。
    -   `group_list`输入类型为`List[int]`时，当前只支持传入`None`。
    -   `group_list`输入类型为`Tensor`时：
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`。
        -   列表长度与`x`列表长度相同。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：每个张量仅支持1维输入。

- **group_list** (`List[int]`/`Tensor`)：可选参数。用于指定分组的索引，表示x的第0维矩阵乘法的索引情况。数据类型支持`int64`。
    -   <term>Atlas 推理系列产品</term>：仅支持<code>**Tensor**</code>类型。仅支持1维输入，长度与<code>weight</code>列表长度相同。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持<code>**List[int]**</code>或<code>**Tensor**</code>类型。若为<code>**Tensor**</code>类型，仅支持1维输入，长度与<code>weight</code>列表长度相同。
    -   配置值要求如下：
        -   `group_list`输入类型为`List[int]`时，配置值必须为非负递增数列，且长度不能为1。
        -   `group_list`输入类型为`Tensor`时：
            -   当`group_list_type`为0时，`group_list`必须为非负、单调非递减数列。
            -   当`group_list_type`为1时，`group_list`必须为非负数列，且长度不能为1。
            -   当`group_list_type`为2时，`group_list` shape为$[E, 2]$，E表示Group大小，数据排布为$[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]$，其中groupSize为分组轴上每组大小，必须为非负数。

- **activation_input** (`List[Tensor]`)：可选参数。代表激活函数的反向输入，当前仅支持传入`None`。
- **activation_quant_scale** (`List[Tensor]`)：可选参数。预留参数，当前只支持传入`None`。
- **activation_quant_offset** (`List[Tensor]`)：可选参数。预留参数，当前只支持传入`None`。
- **split_item** (`int`)：可选参数。用于指定切分模式。数据类型支持`int32`。
    -   0、1：输出为多个张量，数量与`weight`相同。
    -   2、3：输出为单个张量。

- **group_type** (`int`)：可选参数。代表需要分组的轴。数据类型支持`int32`。
    -   `group_list`输入类型为`List[int]`时仅支持传入`None`。

    -   `group_list`输入类型为`Tensor`时，若矩阵乘为$C[m,n]=A[m,k]*B[k,n]$，`group_type`支持的枚举值为：-1代表不分组；0代表m轴分组；1代表n轴分组。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前支持取-1、0。
        -   <term>Atlas 推理系列产品</term>：当前只支持取0。

- **group_list_type** (`int`)：可选参数。代表`group_list`的表达形式。数据类型支持`int32`。
    -   `group_list`输入类型为`List[int]`时仅支持传入`None`。

    -   `group_list`输入类型为`Tensor`时可取值0、1或2：
        -   0：默认值，`group_list`中数值为分组轴大小的cumsum结果（累积和）。
        -   1：`group_list`中数值为分组轴上每组大小。
        -   2：`group_list` shape为$[E, 2]$，E表示Group大小，数据排布为$[[groupIdx0, groupSize0], [groupIdx1, groupSize1]...]$，其中groupSize为分组轴上每组大小。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅当`x`和`weight`参数输入类型为`INT8`，并且`group_type`取0（m轴分组）时，支持取2。
        -   <term>Atlas 推理系列产品</term>：不支持取2。
    
- **act_type** (`int`)：可选参数。代表激活函数类型。数据类型支持`int32`。
    -   `group_list`输入类型为`List[int]`时仅支持传入`None`。

    -   `group_list`输入类型为`Tensor`时，支持的枚举值包括：0代表不激活；1代表`RELU`激活；2代表`GELU_TANH`激活；3代表暂不支持；4代表`FAST_GELU`激活；5代表`SILU`激活。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为0-5。
        -   <term>Atlas 推理系列产品</term>：当前只支持传入0。

- **output_dtype** (`torch.dtype`)：可选参数。输出数据类型。支持的配置包括：
    -   `None`：默认值，表示输出数据类型与输入`x`的数据类型相同。
    -   与输出`y`数据类型一致的类型，具体参考[约束说明](#zh-cn_topic_0000002262888689_section618392112366)。

- **tuning_config** (`List[int]`)：可选参数，数组中的第一个元素表示各个专家处理的token数的预期值，算子tiling时会按照数组中的第一个元素进行最优tiling，性能更优（使用场景参见[约束说明](#zh-cn_topic_0000002262888689_section618392112366)）；从第二个元素开始预留，用户无须填写，未来会进行扩展。如不使用该参数不传即可。
    -   <term>Atlas 推理系列产品</term>：当前暂不支持该参数。

## 返回值说明<a name="zh-cn_topic_0000002262888689_section1558311519405"></a>

`List[Tensor]`：

-   当`split_item`为0或1时，返回的张量数量与`weight`相同。
-   当`split_item`为2或3时，返回的张量数量为1。

## 约束说明<a name="zh-cn_topic_0000002262888689_section618392112366"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：内轴限制InnerLimit为65536。
-   `x`和`weight`中每一组tensor的最后一维大小都应小于InnerLimit。x<sub>i</sub>的最后一维指当x不转置时<code>x<sub>i</sub></code>的K轴或当`x`转置时<code>x<sub>i</sub></code>的$M$轴。<code>weight<sub>i</sub></code>的最后一维指当`weight`不转置时<code>weight<sub>i</sub></code>的$N$轴或当`weight`转置时<code>weight<sub>i</sub></code>的$K$轴。

-   `tuning_config`使用场景限制：

    仅在量化场景（输入`int8`，输出为`int32`/`bfloat16`/`float16`/`int8`，数据类型如下表），且为单tensor单专家的场景下使用。
    |x|	weight|output_dtype|y|
    |---------|--------|--------|--------|
    |`int8`|`int8`|`int8`|`int8`|
    |`int8`|`int8`|`bfloat16`|`bfloat16`|
    |`int8`|`int8`|`float16`|`float16`|
    |`int8`|`int8`|`int32`|`int32`|

-   各场景输入与输出数据类型使用约束：
    -   **`group_list`输入类型为`List[int]`时**，<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>数据类型使用约束。

        **表 1**  数据类型约束
        |场景|x|weight|bias|scale|antiquant_scale|antiquant_offset|output_dtype|y|
        |---------|--------|--------|--------|--------|--------|--------|--------|--------|
        |非量化|`float16`|`float16`|`float16`|无需赋值|无需赋值|无需赋值|`float16`|`float16`|
        |非量化|`bfloat16`|`bfloat16`|`float32`|无需赋值|无需赋值|无需赋值|`bfloat16`|`bfloat16`|
        |非量化|`float32`|`float32`|`float32`|无需赋值|无需赋值|无需赋值|`float32`|`float32`|
        |perchannel全量化|`int8`|`int8`|`int32`|`int64`|无需赋值|无需赋值|`int8`|`int8`|
        |伪量化|`float16`|`int8`|`float16`|无需赋值|`float16`|`float16`|`float16`|`float16`|
        |伪量化|`bfloat16`|`int8`|`float32`|无需赋值|`bfloat16`|`bfloat16`|`bfloat16`|`bfloat16`|

    -   **`group_list`输入类型为`Tensor`时**，数据类型使用约束。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

            **表 2**  数据类型约束
            |场景|x|weight|bias|scale|antiquant_scale|antiquant_offset|per_token_scale|output_dtype|y|
            |---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
            |非量化|`float16`|`float16`|`float16`|无需赋值|无需赋值|无需赋值|无需赋值|None/`float16`|`float16`|
            |非量化|`bfloat16`|`bfloat16`|`float32`|无需赋值|无需赋值|无需赋值|无需赋值|None/`bfloat16`|`bfloat16`|
            |非量化|`float32`|`float32`|`float32`|无需赋值|无需赋值|无需赋值|无需赋值|None/`float32`（仅`x`/`weight`/`y`均为单张量）|`float32`|
            |perchannel全量化|`int8`|`int8`|`int32`|`int64`|无需赋值|无需赋值|无需赋值|None/`int8`|`int8`|
            |perchannel全量化|`int8`|`int8`|`int32`|`bfloat16`|无需赋值|无需赋值|无需赋值|`bfloat16`|`bfloat16`|
            |perchannel全量化|`int8`|`int8`|`int32`|`float32`|无需赋值|无需赋值|无需赋值|`float16`|`float16`|
            |pertoken全量化|`int8`|`int8`|`int32`|`bfloat16`|无需赋值|无需赋值|`float32`|`bfloat16`|`bfloat16`|
            |pertoken全量化|`int8`|`int8`|`int32`|`float32`|无需赋值|无需赋值|`float32`|`float16`|`float16`|
            |pertoken全量化|`int4`|`int4`|无需赋值|`uint64`|无需赋值|无需赋值|None/`float32`|`float16`|`float16`|
            |pertoken全量化|`int4`|`int4`|无需赋值|`uint64`|无需赋值|无需赋值|None/`float32`|`bfloat16`|`bfloat16`|
            |伪量化|`float16`|`int8`/`int4`|`float16`|无需赋值|`float16`|`float16`|无需赋值|None/`float16`|`float16`|
            |伪量化|`bfloat16`|`int8`/`int4`|`float32`|无需赋值|`bfloat16`|`bfloat16`|无需赋值|None/`bfloat16`|`bfloat16`|

            > [!NOTE]   
            > -   伪量化场景，若`weight`的类型为`int8`，仅支持perchannel模式；若`weight`的类型为`int4`，支持perchannel和pergroup两种模式。若为pergroup，pergroup数$G$或$G_i$必须要能整除对应的$k_i$。若`weight`为多tensor，定义pergroup长度$s_i= k_i/G_i$，要求所有$s_i(i=1,2,...g)$都相等。
            > -   伪量化场景，若`weight`的类型为`int4`，则`weight`中每一组tensor的最后一维大小都应是偶数。<code>weight<sub>i</sub></code>的最后一维指`weight`不转置时<code>weight<sub>i</sub></code>的N轴或当weight转置时weight<sub>i</sub>的$K$轴。并且在pergroup场景下，当`weight`转置时，要求pergroup长度$s_i$是偶数。tensor转置：指若tensor shape为$[M,K]$时，则stride为$[1,M]$,数据排布为$[K,M]$的场景，即非连续tensor。
            > -   当前PyTorch不支持`int4`类型数据，需要使用时可以通过[torch\_npu.npu\_quantize](torch_npu-npu_quantize.md)接口使用`int32`数据表示`int4`。

        -   <term>Atlas 推理系列产品</term>：

            **表 3**  数据类型约束
            |x|weight|bias|scale|antiquant_scale|antiquant_offset|per_token_scale|output_dtype|y|
            |--------|--------|--------|--------|--------|--------|--------|--------|--------|
            |`float16`|`float16`|`float16`|无需赋值|无需赋值|无需赋值|`float32`|`float16`|`float16`|
            
-   根据输入`x`、输入`weight`与输出`y`的Tensor数量不同，支持以下几种场景。场景中的“单”表示单个张量，“多”表示多个张量。场景顺序为`x`、`weight`、`y`，例如“单多单”表示`x`为单张量，`weight`为多张量，`y`为单张量。
    -   **`group_list`输入类型为`List[int]`时**，<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>各场景的限制。

        |支持场景|场景说明|场景限制|
        |--------|--------|--------|
        |多多多|`x`和`weight`为多张量，`y`为多张量。每组数据的张量是独立的。|1.仅支持`split_item`为0或1。<br>2.`x`中tensor要求维度一致且支持2-6维，`weight`中tensor需为2维，`y`中tensor维度和x保持一致。<br>3.`x`中tensor大于2维，`group_list`必须传空。<br>4.`x`中tensor为2维且传入`group_list`，`group_list`的差值需与`x`中tensor的第一维一一对应。|
        |单多单|`x`为单张量，`weight`为多张量，`y`为单张量。|1.仅支持`split_item`为2或3。<br>2.必须传`group_list`，且最后一个值与`x`中tensor的第一维相等。<br>3.`x`、`weight`、`y`中tensor需为2维。<br>4.`weight`中每个tensor的N轴必须相等。|
        |单多多|`x`为单张量，`weight`为多张量，`y`为多张量。|1.仅支持`split_item`为0或1。<br>2.必须传`group_list`，`group_list`的差值需与`y`中tensor的第一维一一对应。<br>3.`x`、`weight`、`y`中tensor需为2维。|
        |多多单|`x`和`weight`为多张量，`y`为单张量。每组矩阵乘法的结果连续存放在同一个张量中。|1.仅支持`split_item`为2或3。<br>2.`x`、`weight`、`y`中tensor需为2维。<br>3.`weight`中每个tensor的N轴必须相等。<br>4.若传入`group_list`，`group_list`的差值需与`x`中tensor的第一维一一对应。|
        
    -   **`group_list`输入类型为`Tensor`时**，各场景的限制。
        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

            > [!NOTE]   
            > -   量化、伪量化仅支持`group_type`为-1和0场景。
            > -   仅pertoken量化场景支持激活函数计算。

            |group_type|支持场景|场景说明|场景限制|
            |--------|--------|--------|--------|
            |-1|多多多|`x`和`weight`为多张量，`y`为多张量。每组数据的张量是独立的。|1.仅支持`split_item`为0或1。<br>2.`x`中tensor要求维度一致且支持2-6维，`weight`中tensor需为2维，`y`中tensor维度和`x`保持一致。<br>3.`group_list`必须传空。<br>4.支持`weight`转置，但`weight`中每个tensor是否转置需保持统一。<br>5.`x`不支持转置。|
            |0|单单单|`x`、`weight`与`y`均为单张量。|1.仅支持`split_item`为2或3。<br>2.`weight`中tensor需为3维，`x`、`y`中tensor需为2维。<br>3.必须传`group_list`，且当`group_list_type`为0时，最后一个值与`x`中tensor的第一维相等，当`group_list_type`为1时，数值的总和与`x`中tensor的第一维相等，当`group_list_type`为2时，第二列数值的总和与`x`中tensor的第一维相等。<br>4.`group_list`第1维最大支持1024，即最多支持1024个group。<br>5.支持`weight`转置。<br>6.`x`不支持转置。|
            |0|单多单|`x`为单张量，`weight`为多张量，`y`为单张量。|1.仅支持`split_item`为2或3。<br>2.必须传`group_list`，且当`group_list_type`为0时，最后一个值与`x`中tensor的第一维相等，当`group_list_type`为1时，数值的总和与`x`中tensor的第一维相等且长度最大为128，当`group_list_type`为2时，第二列数值的总和与`x`中tensor的第一维相等且长度最大为128。<br>3.`x`、`weight`、`y`中tensor需为2维。<br>4.`weight`中每个tensor的N轴必须相等。<br>5.支持`weight`转置，但`weight`中每个tensor是否转置需保持统一。<br>6.`x`不支持转置。|
            |0|多多单|`x`和`weight`为多张量，`y`为单张量。每组矩阵乘法的结果连续存放在同一个张量中。|1.仅支持`split_item`为2或3。<br>2.`x`、`weight`、`y`中tensor需为2维。<br>3.`weight`中每个tensor的N轴必须相等。<br>4.若传入`group_list`，当`group_list_type`为0时，`group_list`的差值需与`x`中tensor的第一维一一对应，当`group_list_type`为1时，`group_list`的数值需与`x`中tensor的第一维一一对应且长度最大为128，当`group_list_type`为2时，`group_list`第二列的数值需与`x`中tensor的第一维一一对应且长度最大为128。<br>5.支持`weight`转置，但`weight`中每个tensor是否转置需保持统一。<br>6.`x`不支持转置。|


        -   <term>Atlas 推理系列产品</term>：

            输入输出只支持`float16`的数据类型，输出`y`的n轴大小需要是16的倍数。
            |group_type|支持场景|场景说明|场景限制|
            |--------|--------|--------|--------|
            |0|单单单|`x`、`weight`与`y`均为单张量。|1.仅支持`split_item`为2或3。<br>2.`weight`中tensor需为3维，`x`、`y`中tensor需为2维。<br>3.必须传`group_list`，且当`group_list_type`为0时，最后一个值与`x`中tensor的第一维相等，当`group_list_type`为1时，数值的总和与`x`中tensor的第一维相等。<br>4.`group_list`第1维最大支持1024，即最多支持1024个group。<br>5.支持`weight`转置，不支持`x`转置。|

## 调用示例<a name="zh-cn_topic_0000002262888689_section1566973054111"></a>

-   单算子模式调用

    -   通用调用示例

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

    -   x为int4输入, weight的数据类型为int4数据排布格式为NZ，调用示例如下：

        ```python
        import numpy as np
        import torch
        import torch_npu

        E, K, N = 1, 16, 64
        x = torch.randint(10, (15, 16), dtype=torch.int8).npu()
        weight = torch.randint(10, (1, 16, 64), dtype=torch.int8).npu()

        x_quant = torch_npu.npu_quantize(x.to(torch.float32), torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)
        weight_nz = torch_npu.npu_format_cast(weight.to(torch.float32), 29)
        weight_quant = torch_npu.npu_quantize(weight_nz, torch.tensor([1.]).npu(), None, torch.quint4x2, -1, False)

        scale = torch.rand((E, 1, N), dtype=torch.float32).npu()

        k_group = scale.shape[1]
        scale_np = scale.cpu().numpy()
        scale_uint32 = scale_np.astype(np.float32)
        scale_uint32.dtype = np.uint32
        scale_uint64 = np.zeros((E, k_group, N * 2), dtype=np.uint32)
        scale_uint64[...,::2] = scale_uint32
        scale_uint64.dtype = np.int64
        scale = torch.from_numpy(scale_uint64).npu()

        group_list = torch.Tensor([14]).to(torch.int64).npu()
        per_token_scale = torch.rand((15), dtype=torch.float32).npu()

        output = torch_npu.npu_grouped_matmul([x_quant], [weight_quant], scale=[scale], per_token_scale=[per_token_scale],
                                                group_list=group_list, group_list_type=0, group_type=0,
                                                split_item=3, output_dtype=torch.float16)
        ```

-   图模式调用
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>/<term>Atlas 推理系列产品</term>/<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

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

