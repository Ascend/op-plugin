# torch\_npu.npu\_grouped\_matmul

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id4 -->

## 功能说明

- **API功能**：npu\_grouped\_matmul是一种对多个矩阵乘法（matmul）操作进行分组计算的高效方法。该API实现了对多个矩阵乘法操作的批量处理，通过将具有相同形状或相似形状的矩阵乘法操作组合在一起，减少内存访问开销和计算资源的浪费，从而提高计算效率。典型应用于MoE（混合专家，Mixture of Experts）模型中按专家分组执行矩阵乘：`weight`通常为多tensor（每个专家一份），`group_list`指定各专家处理的token范围。

    > [!NOTE]
    > 本算子当前支持非量化、全量化、伪量化场景，其中全量化和非量化场景适用于训练或推理场景，伪量化场景仅适用于推理场景。

  <details>
  <summary><b>名词解释</b></summary>

  - **量化粒度**：描述单个矩阵的量化参数分布方式。
    - **T**（pertensor）：每个Tensor共用一个量化参数。
    - **C**（perchannel）：量化对象为右矩阵，每个通道独立参数。
    - **K**（pertoken）：量化对象为左矩阵，每个token独立参数。
    - **G**（pergroup）：在reduce轴（即K轴）上分组，每组独立参数。
    - **B**（perblock）：在所有轴上分块，每块独立参数。
  - **量化模式**：左矩阵、右矩阵的量化粒度组合，写作「左粒度-右粒度」。
    - **T-T**：pertensor-pertensor。
    - **T-C**：pertensor-perchannel。
    - **K-T**：pertoken-pertensor。
    - **K-C**：pertoken-perchannel。
    - **G-B**：pergroup-perblock。
    - **K-G**：pertoken-pergroup。
    - **K-CG**：pertoken-perchannel-pergroup，即pertoken量化`x`、pergroup量化`weight`、perchannel反量化输出。
  - **MX量化**：Microscaling Formats，OCP《Microscaling Formats (MX) 规范》定义的低精度数据表示方式，属于pergroup量化模式的特例，量化参数类型为`torch.float8_e8m0fnu`且group size为32。
  - **场景类型**：描述参与量化的矩阵及其量化参数来源。
    - **非量化**：左右矩阵均不量化。
    - **全量化**：左右矩阵均量化。
    - **伪量化**：仅对`weight`量化。
    - **静态量化**：量化参数预先确定。
    - **动态量化**：量化参数由调用方根据输入数据在线计算后，通过`scale`（对应`weight`）和`per_token_scale`（对应`x`）传入，算子本身不生成量化参数。

  </details>

- **计算公式**：

    注意公式中@符号表示矩阵乘法，×符号表示矩阵Hadamard乘积。

  - 非量化场景（公式1），约束见[非量化场景约束](#非量化场景约束)：

    $$
    y_i = x_i \text{@} \mathit{weight}_i + \mathit{bias}_i
    $$

  - 静态量化（T-T、T-C）场景（公式2），约束见[静态量化场景约束](#静态量化t-tt-c场景约束)：

    $$
    y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i + \mathit{offset}_i
    $$

    - x为`torch.int8`输入，bias为`torch.int32`输入（公式2-1）：

        $$
        y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i + \mathit{offset}_i
        $$

    - x为`torch.int8`输入，bias为`torch.bfloat16`、`torch.float16`、`torch.float32`输入，无offset（公式2-2）：

        $$
        y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i + \mathit{bias}_i
        $$

  - 动态量化（K-T、K-C、T-T、T-C）场景（公式3），约束见[动态量化场景约束](#动态量化k-tk-ct-tt-c场景约束)：

    $$
    y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i
    $$

    - x为`torch.int8`输入，bias为`torch.int32`输入（公式3-1）：

        $$
        y_i = (x_i \text{@} \mathit{weight}_i + \mathit{bias}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i
        $$

    - x为`torch.int8`输入，bias为`torch.bfloat16`、`torch.float16`、`torch.float32`输入（公式3-2）：

        $$
        y_i = (x_i \text{@} \mathit{weight}_i) \times \mathit{scale}_i \times \mathit{pertokenscale}_i + \mathit{bias}_i
        $$

  - 伪量化（perchannel、pergroup）场景（公式4），约束见[伪量化场景约束](#伪量化perchannelpergroup场景约束)：

    $$
    y_i = x_i \text{@} \left((\mathit{weight}_i + \mathit{antiquant\_offset}_i) \times \mathit{antiquant\_scale}_i\right) + \mathit{bias}_i
    $$

  <!-- npu="950" id5 -->
  - 动态量化（G-B、MX）场景（公式5）：仅适用于<term>Ascend 950PR&950DT系列产品</term>，约束见[动态量化GB-MX场景约束](#动态量化g-bmx场景约束)

    $$
    \mathit{y}_i[m,n] = \sum_{j=0}^{\mathit{k}_{\mathit{loops}}-1}\left(\left(\sum_{k=0}^{\mathit{gsk}-1}\left(\mathit{x}_{\mathit{slice},i} \times \mathit{weight}_{\mathit{slice},i}\right)\right)\times\left(\mathit{pertokenscale}_i\left[\frac{m}{\mathit{gsm}},j\right] \times \mathit{scale}_i\left[j,\frac{n}{\mathit{gsn}}\right]\right)\right) + \mathit{bias}_i[n]
    $$

    其中gsm、gsn和gsk分别代表M、N、K轴的量化的block size，K<sub>i</sub>为每个分组的K的大小，x\_slice<sub>i</sub>代表x<sub>i</sub>第m行长度为gsk的向量，weight\_slice<sub>i</sub>代表weight<sub>i</sub>第n列长度为gsk的向量，K轴均从j\*gsk起始切片，j的取值范围\[0, k\_loops\)，k\_loops=ceil\(K<sub>i</sub>/gsk\)，支持最后的切片长度不足gsk。
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - 伪量化（MX）场景（公式6）：仅适用于<term>Ascend 950PR&950DT系列产品</term>，约束见[伪量化MX场景约束](#伪量化mx场景约束)
    - x为`torch.float16`、`torch.bfloat16`输入，weight为`torch.float32`（`torch.float4_e2m1fn_x2`）输入（公式6-1）：

        $$
        y_i = x_i \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i) + \mathit{bias}_i
        $$

    - x为`torch.float8_e4m3fn`输入，weight为`torch.float32`（`torch.float4_e2m1fn_x2`）输入（公式6-2）：

        $$
        y_i = (x_i \times \mathit{pertokenscale}_i) \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i) + \mathit{bias}_i
        $$
  <!-- end id6 -->
  <!-- npu="950" id7 -->
  - 伪量化（K-CG）场景（公式7）：仅适用于<term>Ascend 950PR&950DT系列产品</term>，约束见[伪量化K-CG场景约束](#伪量化k-cg场景约束)

    $$
    y_i = \left(x_i \text{@} (\mathit{weight}_i \times \mathit{antiquant\_scale}_i)\right) \times \mathit{scale}_i \times \mathit{pertokenscale}_i + \mathit{bias}_i
    $$

    其中antiquant\_scale<sub>i</sub>为weight矩阵pergroup量化参数，scale<sub>i</sub>为weight矩阵perchannel量化参数，pertokenscale<sub>i</sub>为pertoken量化参数。
  <!-- end id7 -->

### 场景速查表

<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  | 场景 | 公式 | 主要量化参数 | 约束说明 | 调用示例 |
  | --- | --- | --- | --- | --- |
  | 非量化 | 公式1 | 无 | [非量化场景约束](#非量化场景约束) | [非量化场景示例](#非量化场景示例)<br>[图模式非量化示例](#图模式非量化示例) |
  | 静态量化（T-T、T-C） | 公式2 | `scale` | [静态量化（T-T、T-C）场景约束](#静态量化t-tt-c场景约束) | [全量化weightnz示例](#全量化weightnz示例)<br>[图模式全量化示例](#图模式全量化示例)、[图模式全量化weightnz示例](#图模式全量化weightnz示例) |
  | 动态量化（K-T、K-C、T-T、T-C） | 公式3 | `scale`、`per_token_scale` | [动态量化（K-T、K-C、T-T、T-C）场景约束](#动态量化k-tk-ct-tt-c场景约束) | [K-T动态量化示例](#k-t动态量化示例) |
  | 伪量化（perchannel、pergroup） | 公式4 | `antiquant_scale`、`antiquant_offset` | [伪量化（perchannel、pergroup）场景约束](#伪量化perchannelpergroup场景约束) | [pergroup伪量化示例](#pergroup伪量化示例) |
  | 动态量化（G-B、MX） | 公式5 | `scale`、`per_token_scale` | [动态量化（G-B、MX）场景约束](#动态量化g-bmx场景约束) | [MX量化示例](#MX量化示例)<br>[图模式G-B动态量化示例](#图模式g-b动态量化示例)、[图模式MxA4W4全量化示例](#图模式MX-a4w4全量化示例) |
  | 伪量化（MX） | 公式6 | `antiquant_scale`、`per_token_scale` | [伪量化（MX）场景约束](#伪量化mx场景约束) | [MX伪量化示例](#MX伪量化示例)<br>[图模式MX伪量化示例](#图模式MX伪量化示例) |
  | 伪量化（K-CG） | 公式7 | `antiquant_scale`、`scale`、`per_token_scale` | [伪量化（K-CG）场景约束](#伪量化k-cg场景约束) | - |
<!-- end id8 -->

<!-- npu="A3,910b" id9 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

  | 场景 | 公式 | 主要量化参数 | 约束说明 | 调用示例 |
  | --- | --- | --- | --- | --- |
  | 非量化 | 公式1 | 无 | [非量化场景约束](#非量化场景约束) | [非量化场景示例](#非量化场景示例)<br>[图模式非量化示例](#图模式非量化示例) |
  | 静态量化（T-T、T-C） | 公式2 | `scale` | [静态量化（T-T、T-C）场景约束](#静态量化t-tt-c场景约束) | - |
  | 动态量化（K-T、K-C、T-T、T-C） | 公式3 | `scale`、`per_token_scale` | [动态量化（K-T、K-C、T-T、T-C）场景约束](#动态量化k-tk-ct-tt-c场景约束) | - |
  | 伪量化（perchannel、pergroup） | 公式4 | `antiquant_scale`、`antiquant_offset` | [伪量化（perchannel、pergroup）场景约束](#伪量化perchannelpergroup场景约束) | - |
<!-- end id9 -->

<!-- npu="310p" id10 -->
- <term>Atlas推理系列产品</term>：

  | 场景 | 公式 | 主要量化参数 | 约束说明 | 调用示例 |
  | --- | --- | --- | --- | --- |
  | 非量化 | 公式1 | 无 | [非量化场景约束](#非量化场景约束) | [非量化场景示例](#非量化场景示例)<br>[图模式非量化示例](#图模式非量化示例) |
<!-- end id10 -->

## 函数原型

```python
torch_npu.npu_grouped_matmul(x, weight, *, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=-1, group_list_type=0, act_type=0, output_dtype=None, tuning_config=None, x_dtype=None, weight_dtype=None, scale_dtype=None, per_token_scale_dtype=None)
```

## 参数说明

- **`x`**（`List[Tensor]`）：**必选参数**，输入矩阵列表，表示矩阵乘法中的左矩阵。
  - 数据格式支持$ND$，支持的数据类型如下：

    <!-- npu="A3,910b" id11 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int8`。
    <!-- end id11 -->
    <!-- npu="310p" id12 -->
    - <term>Atlas推理系列产品</term>：`torch.float16`。
    <!-- end id12 -->
    <!-- npu="950" id13 -->
    - <term>Ascend 950PR&950DT系列产品</term>：`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.float16`、`torch.bfloat16`、`torch.float32`、`torch.int8`、`torch_npu.hifloat8`、`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch_npu.int4`。其中`torch_npu.int4`实际上是用`torch.int8`或者`torch.int32`承载，保持[-8,7]值域；`torch_npu.hifloat8`/float4系列需配置可选参数`x_dtype`为对应类型，此时`x`本身的`dtype`不再生效，但仍需保证`x`本身的`dtype`为8bit的数据类型，以保证shape正确；其中float4在M轴分组时内轴为`K`，K轴分组时内轴为`M`，float4内轴需为偶数，以保证8bits可以转换为2个float4；`torch_npu.float4_e1m2fn_x2`数据类型仅在`weight`为$FRACTAL\_NZ$时支持。
    <!-- end id13 -->

  - 列表最大长度支持如下：
    <!-- npu="A3,910b,310p" id109 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：列表最大长度为128。
    <!-- end id109 -->

    <!-- npu="950" id14 -->
    - <term>Ascend 950PR&950DT系列产品</term>：非量化场景支持列表最大长度为1024，伪量化场景支持列表最大长度为128，全量化场景仅支持单tensor。
    <!-- end id14 -->

  - 当`split_item`=0、1时，多tensor组合场景（多多多、单多单、单多多、多多单，详见[支持场景约束](#支持场景约束)）张量支持2至6维输入；其他情况下，张量仅支持2维输入。各量化场景下的数据类型组合及shape约束详见[约束说明](#约束说明)。

- **`weight`**（`List[Tensor]`）：**必选参数**，权重矩阵列表，表示矩阵乘法中的右矩阵。
  - 数据格式支持$ND$/$FRACTAL\_NZ$，支持的数据类型如下：

    <!-- npu="A3,910b" id15 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：
      - 当`group_list`输入类型为`List[int]`时，支持`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch.int8`。
      - 当`group_list`输入类型为`Tensor`时，支持`torch.float16`、`torch.float32`、`torch.bfloat16`、`torch_npu.int4`、`torch.int8`。
    <!-- end id15 -->
    <!-- npu="310p" id16 -->
    - <term>Atlas推理系列产品</term>：`torch.float16`。
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - <term>Ascend 950PR&950DT系列产品</term>：支持`torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.int8`、`torch_npu.hifloat8`、`torch.bfloat16`、`torch.float16`、`torch.float32`、`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`、`torch.int32`、`torch_npu.int4`。其中`torch_npu.int4`实际上是用`torch.int8`或者`torch.int32`承载，保持[-8,7]值域；`torch_npu.hifloat8`/float4系列需配置可选参数`weight_dtype`为对应类型，此时`weight`本身的`dtype`不再生效，但仍需保证`weight`本身的`dtype`为8bit的数据类型，以保证shape正确；`torch_npu.float4_e1m2fn_x2`数据类型仅在`weight`为$FRACTAL\_NZ$时支持。各场景下`weight`的数据格式（$ND$/$FRACTAL\_NZ$）、承载方式（如`torch.float32`/`torch.int32`承载float4/int4、`torch.uint8`承载MxA8W4）及转置要求详见[约束说明](#约束说明)；$FRACTAL\_NZ$格式可通过`torch_npu.npu_format_cast`接口实现$ND$转$FRACTAL\_NZ$格式。
    <!-- end id17 -->

  - 列表最大长度支持如下：
    <!-- npu="A3,910b,310p" id110 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：列表最大长度为128。
    <!-- end id110 -->

    <!-- npu="950" id18 -->
    - <term>Ascend 950PR&950DT系列产品</term>：非量化场景支持列表最大长度为1024，伪量化场景支持列表最大长度为128，全量化场景仅支持单tensor；全量化MX量化且`weight`为$FRACTAL\_NZ$格式时，支持多`weight`多tensor输入，支持tensorlist最大长度为128。
    <!-- end id18 -->

  - 每个张量支持2维或3维输入。

- <strong>*</strong>：代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **`bias`**（`List[Tensor]`）：**可选参数**，每个分组的矩阵乘法输出的独立偏置项。
  - 数据格式支持$ND$，支持的数据类型如下：

    <!-- npu="A3,910b" id19 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.float32`和`torch.int32`。
    <!-- end id19 -->
    <!-- npu="310p" id20 -->
    - <term>Atlas推理系列产品</term>：`torch.float16`。
    <!-- end id20 -->
    <!-- npu="950" id21 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型与`x`数据类型及量化场景相关，各场景下的具体组合详见[约束说明](#约束说明)；不满足组合要求时需传None。
    <!-- end id21 -->

  - 列表长度与`weight`列表长度相同。
  - 每个张量支持1维或2维输入，各场景下shape要求详见[约束说明](#约束说明)。

- **`scale`**（`List[Tensor]`）：**可选参数**，用于缩放原数值以匹配量化后的范围值，代表量化参数中的缩放因子。
  - 数据格式和数据类型支持如下：

    <!-- npu="A3,910b" id22 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅支持$ND$格式。
      - 当`group_list`输入类型为`List[int]`时，支持`torch.int64`。
      - 当`group_list`输入类型为`Tensor`时，支持`torch.float32`、`torch.bfloat16`、`torch.int64`。
    <!-- end id22 -->
    <!-- npu="310p" id23 -->
    - <term>Atlas推理系列产品</term>：仅支持传入None。
    <!-- end id23 -->
    <!-- npu="950" id24 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
      - $ND$格式：当`group_list`输入类型为`torch.Tensor`时，支持`torch.int64`、`torch.bfloat16`、`torch.float32`、`torch.float8_e8m0fnu`（`x`为`torch.float16`/`torch.bfloat16`时的支持类型详见[约束说明](#约束说明)）。其中`torch.float8_e8m0fnu`需配置`scale_dtype`为`torch_npu.float8_e8m0fnu`，此时`scale`本身的`dtype`不再生效，但仍需保证`scale`本身的`dtype`为8bit的数据类型，以保证shape正确；各量化场景下支持的具体类型详见[约束说明](#约束说明)。
      - $FRACTAL\_NZ$格式：仅支持`torch.float8_e8m0fnu`，且需配置`scale_dtype`为`torch_npu.float8_e8m0fnu`。
    <!-- end id24 -->

  - 列表长度与`weight`列表长度相同。

  <!-- npu="A3,910b" id25 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：每个张量仅支持1维输入。
  <!-- end id25 -->
  <!-- npu="950" id26 -->
  - <term>Ascend 950PR&950DT系列产品</term>：每个张量的维度与量化场景、分组轴相关，详见[静态量化场景约束](#静态量化t-tt-c场景约束)、[动态量化场景约束](#动态量化k-tk-ct-tt-c场景约束)、[伪量化场景约束](#伪量化perchannelpergroup场景约束)、[动态量化GB-MX场景约束](#动态量化g-bmx场景约束)。
  <!-- end id26 -->

- **`offset`**（`List[Tensor]`）：**可选参数**，用于调整量化后的数值偏移量，从而更准确地表示原始浮点数值。数据格式支持$ND$。

  <!-- npu="A3,910b,310p" id111 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：仅支持传入None。
  <!-- end id111 -->
  <!-- npu="950" id27 -->
  - <term>Ascend 950PR&950DT系列产品</term>：仅S8S4 K-C伪量化场景支持`torch.float32`，详见[伪量化场景约束](#伪量化perchannelpergroup场景约束)。
  <!-- end id27 -->

- **`antiquant_scale`**（`List[Tensor]`）：**可选参数**，用于缩放原数值以匹配伪量化后的范围值，代表伪量化参数中的缩放因子。
  - 数据格式支持$ND$，支持的数据类型如下：

    <!-- npu="A3,910b" id28 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.bfloat16`。
    <!-- end id28 -->
    <!-- npu="310p" id29 -->
    - <term>Atlas推理系列产品</term>：仅支持传入None。
    <!-- end id29 -->
    <!-- npu="950" id30 -->
    - <term>Ascend 950PR&950DT系列产品</term>：`torch.float16`、`torch.bfloat16`、`torch.float8_e8m0fnu`，其中`torch.float8_e8m0fnu`需使用`torch.uint8`承载。
    <!-- end id30 -->

  - 列表长度与`weight`列表长度相同。
  - 每个张量的shape随伪量化模式（perchannel、pergroup、MX、K-CG）不同，详见[伪量化场景约束](#伪量化perchannelpergroup场景约束)、[伪量化MX场景约束](#伪量化mx场景约束)、[伪量化K-CG场景约束](#伪量化k-cg场景约束)。

- **`antiquant_offset`**（`List[Tensor]`）：**可选参数**，用于调整伪量化后的数值偏移量，从而更准确地表示原始浮点数值。
  - 数据格式支持$ND$，支持的数据类型如下：

    <!-- npu="A3,910b" id31 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`torch.float16`、`torch.bfloat16`。
    <!-- end id31 -->
    <!-- npu="310p" id32 -->
    - <term>Atlas推理系列产品</term>：仅支持传入None。
    <!-- end id32 -->
    <!-- npu="950" id33 -->
    - <term>Ascend 950PR&950DT系列产品</term>：`torch.float16`、`torch.bfloat16`。
    <!-- end id33 -->

  - 列表长度与`weight`列表长度相同。
  - 每个张量输入维度和`antiquant_scale`输入维度一致。

- **`per_token_scale`**（`List[Tensor]`）：**可选参数**，用于缩放原数值以匹配量化后的范围值，代表量化参数中由`x`量化引入的缩放因子，数据格式支持$ND$。
  - `group_list`输入类型为`List[int]`时，当前只支持传入None。
  - `group_list`输入类型为`Tensor`时：

    <!-- npu="A3,910b" id34 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：数据类型支持`torch.float32`。
    <!-- end id34 -->
    <!-- npu="950" id35 -->
    - <term>Ascend 950PR&950DT系列产品</term>：数据类型支持`torch.float32`、`torch.float8_e8m0fnu`，其中`torch.float8_e8m0fnu`需配置`per_token_scale_dtype`为`torch_npu.float8_e8m0fnu`，此时`per_token_scale`本身的`dtype`不再生效，但仍需保证`per_token_scale`本身的`dtype`为8bit的数据类型，以保证shape正确。
    <!-- end id35 -->
    <!-- npu="310p" id36 -->
    - <term>Atlas推理系列产品</term>：仅支持传入None。
    <!-- end id36 -->

  - 列表长度与`x`列表长度相同。

    <!-- npu="A3,910b" id37 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：每个张量仅支持1维输入。
    <!-- end id37 -->
    <!-- npu="950" id38 -->
    - <term>Ascend 950PR&950DT系列产品</term>：每个张量的维度与量化场景相关（MX量化、MX伪量化、K-CG伪量化），详见[动态量化GB-MX场景约束](#动态量化g-bmx场景约束)、[伪量化MX场景约束](#伪量化mx场景约束)、[伪量化K-CG场景约束](#伪量化k-cg场景约束)。
    <!-- end id38 -->

- **`group_list`**（`List[int]`/`torch.Tensor`）：**可选参数**，用于指定分组的索引，表示`x`的第0维矩阵乘法的索引情况。数据格式支持$ND$，数据类型支持`torch.int64`。

  <!-- npu="310p" id39 -->
  - <term>Atlas推理系列产品</term>：仅支持`torch.Tensor`类型。仅支持1维输入，长度与`weight`列表长度相同。
  <!-- end id39 -->
  <!-- npu="A3,910b" id40 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：支持`List[int]`或`torch.Tensor`类型。若为`torch.Tensor`类型，仅支持1维输入，长度与`weight`列表长度相同。
  <!-- end id40 -->
  <!-- npu="950" id41 -->
  - <term>Ascend 950PR&950DT系列产品</term>：仅支持`torch.Tensor`类型。列表长度对应分组数：`group_type`为0且`weight`为单tensor时，长度与`weight`的最高维E一致。按`group_list_type`区分表示：
    - `group_list_type`=0（累积和）：1维，数值为各组在分组轴上的结束位置（例`[2, 2, 5]`表示3组）；末值对应分组轴大小，`group_type`为0时对应`x`的M轴、`group_type`为2时对应K轴，且末值不大于该轴大小。
    - `group_list_type`=1（组大小）：1维，数值为分组轴上每组大小（例`[2, 0, 3]`）；所有组大小之和（总和）不大于分组轴大小。
    - `group_list_type`=2（索引+组大小）：2维`[E, 2]`，每行为`[groupIdx, groupSize]`（例`[[0, 2], [2, 3], [1, 0]]`）；第二列数值之和不大于分组轴大小。
  <!-- end id41 -->
  - 配置值要求如下：
    - `group_list`输入类型为`List[int]`时，配置值必须为非负递增数列，且长度不能为0。
    - `group_list`输入类型为`Tensor`时：
      - 当`group_list_type`为0时，`group_list`必须为非负、单调非递减数列。
      - 当`group_list_type`为1时，`group_list`必须为非负数列，且长度不能为0。
      - 当`group_list_type`为2时，`group_list`的shape为\[E, 2\]，E表示Group大小，数据排布为\[\[groupIdx0, groupSize0\], \[groupIdx1, groupSize1\], ...\]，其中groupSize为分组轴上每组大小，必须为非负数。所有groupSize非0的分组按groupIdx有序排列在前，所有groupSize为0的分组按groupIdx有序排列在后，确保非零组前置、零值组后置，且组内有序。

- **`activation_input`**（`List[Tensor]`）：**可选参数**，代表激活函数的反向输入，当前仅支持传入None。
- **`activation_quant_scale`**（`List[Tensor]`）：**可选参数**，预留参数，当前只支持传入None。
- **`activation_quant_offset`**（`List[Tensor]`）：**可选参数**，预留参数，当前只支持传入None。
- **`split_item`**（`int`）：**可选参数**，用于指定切分模式。数据类型支持`int64`。
  - 0、1：输出为多个张量，数量与weight相同。
  - 2、3：输出为单个张量。

  <!-- npu="A3,910b,310p" id112 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：支持取0、1、2、3，具体取值受场景约束，详见[支持场景约束](#支持场景约束)。
  <!-- end id112 -->
  <!-- npu="950" id42 -->
  - <term>Ascend 950PR&950DT系列产品</term>：非量化和伪量化支持取0/1/2/3，全量化当前支持取2和3。
  <!-- end id42 -->

- **`group_type`**（`int`）：**可选参数**，代表需要分组的轴。数据类型支持`int64`。
  - `group_list`输入类型为`List[int]`时仅支持传入None。
  - `group_list`输入类型为`Tensor`时，若矩阵乘为C\[m,n\]=A\[m,k\]xB\[k,n\]，`group_type`支持的枚举值为：-1代表不分组；0代表m轴分组；1代表n轴分组，2代表k轴分组。

    <!-- npu="A3,910b" id43 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当前支持取-1、0、2。
    <!-- end id43 -->
    <!-- npu="310p" id44 -->
    - <term>Atlas推理系列产品</term>：当前只支持取0。
    <!-- end id44 -->
    <!-- npu="950" id45 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
      - 非量化场景支持取-1、0、2，weightNz场景时不支持取2。
      - 全量化场景支持取0、2。
      - 伪量化场景支持取-1、0。
    <!-- end id45 -->

- **`group_list_type`**（`int`）：**可选参数**，代表`group_list`的表达形式。数据类型支持`int64`。
  - `group_list`输入类型为`List[int]`时仅支持传入None。
  - `group_list`输入类型为`Tensor`时可取值0、1或2：
    - 0：默认值，`group_list`中数值为分组轴大小的cumsum（累积和）结果，例`[2, 2, 5]`表示3组，各组结束位置为2、2、5。
    - 1：`group_list`中数值为分组轴上每组大小，例`[2, 0, 3]`表示3组，每组大小分别为2、0、3。
    - 2：`group_list` shape为\[E, 2\]，E表示Group大小，数据排布为\[\[groupIdx0, groupSize0\], \[groupIdx1, groupSize1\], ...\]，其中groupSize为分组轴上每组大小，例`[[0, 2], [2, 3], [1, 0]]`表示3个分组的索引与大小。

    <!-- npu="A3,910b" id46 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：仅当`x`和`weight`参数输入类型为`torch.int8`，并且`group_type`取0（m轴分组）时，支持取2。
    <!-- end id46 -->
    <!-- npu="310p" id47 -->
    - <term>Atlas推理系列产品</term>：不支持取2。
    <!-- end id47 -->
    <!-- npu="950" id48 -->
    - <term>Ascend 950PR&950DT系列产品</term>：
      - 仅全量化场景下且`group_type`取0（m轴分组）时支持取2，其余场景不支持。
      - K-CG伪量化场景下，`group_list_type`仅支持取1。
    <!-- end id48 -->

- **`act_type`**（`int`）：**可选参数**，代表激活函数类型。数据类型支持`int64`。
  - `group_list`输入类型为`List[int]`时仅支持传入None。
  - `group_list`输入类型为`Tensor`时，支持的枚举值包括：0代表NONE，不启用激活；1代表RELU激活；2代表GELU\_TANH激活；3代表GELU\_ERF\_FUNC激活，目前暂不支持；4代表FAST\_GELU激活；5代表SILU激活。

    <!-- npu="A3,910b" id49 -->
    - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：取值范围为0-5。
    <!-- end id49 -->
    <!-- npu="310p" id50 -->
    - <term>Atlas推理系列产品</term>：当前只支持传入0。
    <!-- end id50 -->
    <!-- npu="950" id51 -->
    - <term>Ascend 950PR&950DT系列产品</term>：当前支持传入0、1、2、4、5，各量化场景下的具体支持范围详见[约束说明](#约束说明)。
    <!-- end id51 -->

- **`output_dtype`**（`int`）：**可选参数**，输出数据类型。支持的配置包括：
  - None：默认值，表示输出数据类型与输入`x`的数据类型相同。
  - 需与输出`y`的数据类型一致，具体取值见[约束说明](#约束说明)中各场景数据类型表的`y`列。

- **`tuning_config`**（`List[int]`）：**可选参数**，调优参数。如不使用该参数不传即可。
  <!-- npu="A3,910b" id113 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：当前暂不支持该参数。
  <!-- end id113 -->
  <!-- npu="310p" id52 -->
  - <term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id52 -->
  <!-- npu="950" id53 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当前仅S8S4 K-C和S8S4 K-CG伪量化场景、以及满足[通用约束](#通用约束)中量化场景数据类型组合（输入`x`为`torch.int8`，且为单tensor单专家场景）时支持。
    - 第一个元素表示各个专家处理的token数的预期值。支持0和正整数，取值范围为\[0, min(M, UINT32_MAX)\]，其中M为输入`x`的行数。0表示不指定预期值；正整数表示各个专家处理的token数的预期值，且不能大于M。当前仅校验并保存该值，暂不参与实际调优计算。
    - 第二个元素仅支持0或1。设置为0或不传入该元素时，不开启`weight`特殊格式，`weight`按照常规\[E, K, N\]逻辑布局解析；设置为1时，开启`weight`特殊格式，仅支持`offset`不为空的perchannel场景，此时`weight`需按照\[E, N, K\]排布后转换为$FRACTAL\_NZ$格式，且`weight`列表长度为1。例如，传入`[0, 1]`表示不指定预期token数，并开启`weight`特殊格式。
  <!-- end id53 -->

- **`x_dtype`**（`int`）：**可选参数**，输入`x`的真实数据类型。默认值None，表示输入`x`真实的数据类型与输入`x`的`dtype`相同。

  <!-- npu="A3,910b,310p" id54 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id54 -->
  <!-- npu="950" id55 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持`torch_npu.hifloat8`、`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`。
  <!-- end id55 -->

- **`weight_dtype`**（`int`）：**可选参数**，输入`weight`的真实数据类型。默认值None，表示输入`weight`真实的数据类型与输入`weight`的`dtype`相同。伪量化场景使用`torch.float32`表示`torch.float4_e2m1fn_x2`时无需传入。

  <!-- npu="A3,910b,310p" id56 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id56 -->
  <!-- npu="950" id57 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持`torch_npu.hifloat8`、`torch.float4_e2m1fn_x2`、`torch_npu.float4_e1m2fn_x2`。
  <!-- end id57 -->

- **`scale_dtype`**（`int`）：**可选参数**，输入`scale`的真实数据类型。默认值None，表示输入`scale`真实的数据类型与输入`scale`的`dtype`相同。

  <!-- npu="A3,910b,310p" id58 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id58 -->
  <!-- npu="950" id59 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当前仅支持`torch.float8_e8m0fnu`。
  <!-- end id59 -->

- **`per_token_scale_dtype`**（`int`）：**可选参数**，输入`per_token_scale`的真实数据类型。默认值None，表示`per_token_scale`真实的数据类型与输入`per_token_scale`的`dtype`相同。

  <!-- npu="A3,910b,310p" id60 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>、<term>Atlas推理系列产品</term>：当前暂不支持该参数。
  <!-- end id60 -->
  <!-- npu="950" id61 -->
  - <term>Ascend 950PR&950DT系列产品</term>：当前仅支持`torch.float8_e8m0fnu`。
  <!-- end id61 -->

## 返回值说明

**`y`**（`List[Tensor]`）：输出Tensor，数据格式支持$ND$。

- 当`split_item`为0或1时，返回的张量数量与`weight`相同。
- 当`split_item`为2或3时，返回的张量数量为1。

## 约束说明

> [!NOTE]
>
> 场景约束的表格若涉及数据类型组合，未列出的可选量化参数在对应场景下则无需传入，如[非量化场景约束](#非量化场景约束)涉及`x`、`weight`、`y`、`bias`和`output_dtype`，那么可选参数`scale`、`offset`、`antiquant_scale`、`antiquant_offset`、`per_token_scale`无需传入。

### 通用约束

- 该接口在全量化和非量化场景适用于训练或推理场景，伪量化场景仅适用于推理场景。
- 该接口支持单算子模式和TorchAir图模式。
- 输入`x`、`weight`与输出`y`的Tensor数量组合称为场景：单表示单个张量，多表示多个张量，场景顺序为x、weight、y，例如“单多单”表示`x`为单张量、`weight`为多张量、`y`为单张量。各产品支持的场景及限制见[支持场景约束](#支持场景约束)。
- `x`和`weight`中每一组tensor的最后一维大小都应小于InnerLimit。`x`<sub>i</sub>的最后一维指当`x`不转置时`x`<sub>i</sub>的K轴或当`x`转置时`x`<sub>i</sub>的M轴；`weight`<sub>i</sub>的最后一维指当`weight`不转置时`weight`<sub>i</sub>的N轴或当`weight`转置时`weight`<sub>i</sub>的K轴。
- `x`和`weight`中每一组tensor的每一维大小，在32字节对齐后都应小于int32的最大值2147483647。

  <!-- npu="A3,910b" id62 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：内轴限制InnerLimit为65536。
  <!-- end id62 -->

- `split_item`为0或1时返回的张量数量与`weight`相同；为2或3时返回1个张量。
- `output_dtype`配置为与输出`y`数据类型一致的类型；未配置时输出数据类型与输入`x`的数据类型相同。
- `group_list`配置值要求：输入类型为`List[int]`时必须为非负递增数列且长度不能为0；输入类型为`Tensor`时，`group_list_type`为0时须非负、单调非递减，为1时须非负且长度不能为0，为2时shape为\[E, 2\]且groupSize非负、非零组前置与零值组后置有序。

    <!-- npu="950" id63 -->
  - <term>Ascend 950PR&950DT系列产品</term>：WeightNZ场景说明如下：
    - 仅伪量化与全量化（输入`x`/`weight`数据类型为`torch.int8`、`x`/`weight`数据类型为`torch.float8_e4m3fn`/`torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`且`scale_dtype`与`per_token_scale_dtype`为`torch.float8_e8m0fnu`）weight支持$FRACTAL\_NZ$数据格式。
    - 全量化场景下，`weight`输入为$FRACTAL\_NZ$数据格式时，K轴和N轴均不能为1。MXFP4场景，K必须大于2，weight不转置时N必须大于2。MXFP4场景支持静态图模式，不支持动态图模式。MXFP4静态图模式场景E必须大于1且K必须大于64。x和weight均为`torch_npu.int4`时，不支持图模式运行，K必须保持8对齐。
  <!-- end id63 -->

  <!-- npu="950" id64 -->
  - <term>Ascend 950PR&950DT系列产品</term>：`tuning_config`使用场景限制：
    - S8S4 K-C和S8S4 K-CG伪量化场景。
    - 量化场景：输入为`torch.int8`，输出为`torch.int32`/`torch.bfloat16`/`torch.float16`/`torch.int8`，且为单tensor单专家场景，数据类型如下表。

      | x | weight | output_dtype | y |
      | --- | --- | --- | --- |
      | `torch.int8` | `torch.int8` | `torch.int8` | `torch.int8` |
      | `torch.int8` | `torch.int8` | `torch.bfloat16` | `torch.bfloat16` |
      | `torch.int8` | `torch.int8` | `torch.float16` | `torch.float16` |
      | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int32` |
  <!-- end id64 -->

### 支持场景约束

> [!NOTE]
>
> - 量化、伪量化仅支持`group_type`为-1和0场景。
> - 仅pertoken量化场景支持激活函数计算，各场景下`act_type`的取值范围详见对应场景约束。
> - x、weight、y 输入为 TensorList。"单" = 单元素TensorList，"多" = 多元素TensorList。

<!-- npu="A3,910b" id65 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`group_list`输入类型为`List[int]`时支持的场景如下：

  | x | weight | y | split_item | group_list | 其余场景限制 |
  | --- | --- | --- | --- | --- | --- |
  | 多 | 多 | 多 | 0/1 | `x`中tensor大于2维时必须传空；`x`中tensor为2维且传入时，差值需与`x`中tensor的第一维一一对应 | `x`中tensor维度需一致且支持2-6维，`weight`需2维，`y`维度与`x`一致 |
  | 单 | 多 | 单 | 2/3 | 必须传，最后一个值与`x`中tensor的第一维相等 | 1）`x`、`weight`、`y`需2维；<br>  2）`weight`中各tensor的N轴必须相等 |
  | 单 | 多 | 多 | 0/1 | 必须传，差值需与`y`中tensor的第一维一一对应 | `x`、`weight`、`y`需2维 |
  | 多 | 多 | 单 | 2/3 | 可选，若传则差值需与`x`中tensor的第一维一一对应 | 1）`x`、`weight`、`y`需2维；<br>  2）`weight`中各tensor的N轴必须相等 |

  该产品下各量化场景约束见：[非量化](#非量化场景约束)、[静态量化（T-T、T-C）](#静态量化t-tt-c场景约束)、[动态量化（K-T、K-C、T-T、T-C）](#动态量化k-tk-ct-tt-c场景约束)、[伪量化（perchannel、pergroup）](#伪量化perchannelpergroup场景约束)。
<!-- end id65 -->

<!-- npu="A3,910b" id66 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：`group_list`输入类型为`Tensor`时支持的场景如下：

  | group_type | x | weight | y | split_item | group_list | 转置 | 其余场景限制 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | -1 | 多 | 多 | 多 | 0/1 | 必须传空 | x不转置；weight可转置（各tensor需统一） | `x`中tensor维度需一致且支持2-6维，`weight`需2维，`y`维度与`x`一致 |
  | 0 | 单 | 单 | 单 | 2/3 | 必须传；`group_list_type`为0/1/2时的取值要求见[`group_list`](#参数说明)；最大1024组 | x不转置；weight可转置 | `weight`需3维，`x`、`y`需2维 |
  | 0 | 单 | 多 | 单 | 2/3 | 必须传；`group_list_type`为0/1/2时的取值要求见[`group_list`](#参数说明)；最大128组 | x不转置；weight可转置（各tensor需统一） | 1）`x`、`weight`、`y`需2维；<br>  2）`weight`中各tensor的N轴必须相等 |
  | 0 | 多 | 多 | 单 | 2/3 | 可选，若传需满足`group_list_type`为0/1/2的取值要求；最大128组 | x不转置；weight可转置（各tensor需统一） | 1）`x`、`weight`、`y`需2维；<br>  2）`weight`中各tensor的N轴必须相等 |
  | 2 | 单 | 单 | 单 | 2/3 | 必须传；`group_list_type`为0/1/2时的取值要求见[`group_list`](#参数说明)；最大1024组 | x必须转置；weight不转置 | `x`、`weight`需2维，`y`需3维 |

  该产品下各量化场景约束见：[非量化](#非量化场景约束)、[静态量化（T-T、T-C）](#静态量化t-tt-c场景约束)、[动态量化（K-T、K-C、T-T、T-C）](#动态量化k-tk-ct-tt-c场景约束)、[伪量化（perchannel、pergroup）](#伪量化perchannelpergroup场景约束)。
<!-- end id66 -->

<!-- npu="310p" id67 -->
- <term>Atlas推理系列产品</term>：`group_list`输入类型为`Tensor`时支持的场景如下：

  输入输出只支持`torch.float16`的数据类型，输出y的n轴大小需要是16的倍数。

  | group_type | x | weight | y | split_item | group_list | 转置 | 其余场景限制 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | 0 | 单 | 单 | 单 | 2/3 | 必须传；`group_list_type`为0/1时的取值要求见[`group_list`](#参数说明)；最大1024组 | x不转置；weight可转置 | `weight`需3维，`x`、`y`需2维 |

  该产品下各量化场景约束见：[非量化](#非量化场景约束)。
<!-- end id67 -->

<!-- npu="950" id68 -->
- <term>Ascend 950PR&950DT系列产品</term>：`group_list`输入类型为`Tensor`时支持的场景如下：

  > [!NOTE]
  >
  > - 全量化仅支持`group_type`为0和2的单单单场景，仅在输入x和weight为`torch_npu.hifloat8`、`torch.float8_e4m3fn`、`torch.float8_e5m2`或`torch.float4_e2m1fn_x2`的全量化支持训练场景即`group_type`为2。
  > - 伪量化场景支持`group_type`为-1的多多多场景，以及`group_type`为0的单单单和单多单场景。其中，单多单表示`x`为单张量、`weight`为多张量、`y`为单张量。
  >   - perchannel伪量化场景，且输入`x`为`torch.float16`/`torch.bfloat16`、`weight`为`torch.int8`/`torch.int32`（`torch_npu.int4`）时。
  >   - pergroup伪量化场景，且输入`x`为`torch.float16`/`torch.bfloat16`、`weight`为`torch.int32`（`torch_npu.int4`）时。
  > - 伪量化场景不支持激活函数计算。全量化场景下，当x和weight为`torch.int8`，量化模式为静态T-C量化或动态K-C量化，且scale类型为`torch.float32`/`torch.bfloat16`时，`act_type`参数支持传入0、1、2、4、5。其余全量化场景`act_type`参数仅支持传入0。
  > - 非量化场景在x、weight、out数据类型为`torch.float16`/`torch.bfloat16`、数据格式为ND且`group_type`为0/-1场景下支持`act_type`参数传入0/2；其余场景`act_type`参数仅支持传入0。
  > x、weight、y 输入为 TensorList。"单" = 单元素TensorList，"多" = 多元素TensorList。

  | group_type | x | weight | y | split_item | group_list | 转置 | 其余场景限制 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | -1 | 多 | 多 | 多 | 0/1（非量化仅支持0） | 必须传空 | x不转置；weight可转置（各tensor需统一） | 1）非量化/伪量化支持`bias`（1维tensorList[(n),...]，长度与`weight`相同）；<br>  2）pergroup伪量化`weight`最后一维需为偶数且K为group_size整数倍；<br>  3）`x`中各tensor维度需一致（非量化与pergroup伪量化仅2维，其他场景2~6维），`weight`需2维，`y`与`x`一致 |
  | 0 | 单 | 单 | 单 | 2/3 | 必须传；`group_list_type`为0/1/2时的取值要求见[`group_list`](#参数说明)；最大1024组 | x不转置；weight可转置或不转置（`x`、`weight`均为`torch_npu.int4`且`weight`为ND时仅支持不转置） | 1）`weight`单tensor需3维，`x`、`y`需2维；<br>  2）K轴不能为0；<br>  3）MX/K-CG伪量化`weight`后2维需32B对齐；<br>  4）量化场景仅`x`为`torch.int8`/float4系列时支持`bias`，非量化与伪量化均支持；<br>  5）`bias` shape仅（g, n）；<br>  6）MX、perchannel的dtype与转置特例见各场景约束 |
  | 0 | 单 | 多 | 单 | 2/3 | 必须传；`group_list_type`为0/1/2时的取值要求见[`group_list`](#参数说明)；最大128组（非量化1024） | x不转置；weight可转置（各tensor需统一） | 1）`x`、`y`需2维（M, K）/（M, N）；<br>  2）`weight`需2维（N, K）或（K, N）且各tensor的N轴相等；<br>  3）全量化MX量化且`weight`为$FRACTAL\_NZ$格式时不支持`bias`；<br>  4）非量化/伪量化支持`bias`（1维tensorList） |
  | 0 | 多 | 多 | 单 | 2/3 | 可选，若传需满足`group_list_type`为0/1的取值要求；最大128组（非量化1024） | x不转置；weight可转置（各tensor需统一） | 1）`x`、`weight`、`y`需2维；<br>  2）`weight`中各tensor的N轴必须相等；<br>  3）非量化场景支持`bias`（1维tensorList，长度与`weight`相同） |
  | 2 | 单 | 单 | 单 | 2/3 | 必须传；`group_list_type`为0/1时的取值要求见[`group_list`](#参数说明)；最大1024组 | x必须转置；weight不转置 | 1）`x`、`weight`需2维，`y`需3维；<br>  2）单算子模式下支持总`K`=0，图模式场景下K轴不能为0；<br>  3）全量化场景下不支持`scale`为`torch.int64`；<br>  4）不支持`bias`；<br>  5）仅支持ND进ND出 |
  | 2 | 单 | 多 | 多 | 0/1 | 可选，若传需满足`group_list_type`为0/1的取值要求；最大1024组 | x必须转置；weight不转置 | 1）`x`、`weight`、`y`需2维；<br>  2）非分组轴的shape不能为0；<br>  3）仅支持ND进ND出；<br>  4）不支持`bias`；<br>  5）仅支持非量化场景 |

  该产品下各量化场景约束见：[非量化](#非量化场景约束)、[静态量化（T-T、T-C）](#静态量化t-tt-c场景约束)、[动态量化（K-T、K-C、T-T、T-C）](#动态量化k-tk-ct-tt-c场景约束)、[伪量化（perchannel、pergroup）](#伪量化perchannelpergroup场景约束)、[动态量化（G-B、MX）](#动态量化g-bmx场景约束)、[伪量化（MX）](#伪量化mx场景约束)、[伪量化（K-CG）](#伪量化k-cg场景约束)。
<!-- end id68 -->

### 非量化场景约束

支持的Tensor数量组合见[支持场景约束](#支持场景约束)。

- 输入`x`、`weight`与输出`y`的数据类型组合、`bias`及`output_dtype`需满足下表。

  <!-- npu="950" id69 -->
  - <term>Ascend 950PR&950DT系列产品</term>：

    | x | weight | bias | output_dtype | y |
    | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.float16` | `torch.float16`/`torch.float32` | None | `torch.float16` |
    | `torch.bfloat16` | `torch.bfloat16` | `torch.float16`/`torch.float32` | None | `torch.bfloat16` |
    | `torch.float32` | `torch.float32` | `torch.float32` | None | `torch.float32` |

    - 列表支持的最大长度为1024。
    - `weight`输入为`torch.float16`或`torch.bfloat16`且shape后两维对32B对齐时，数据格式支持$FRACTAL\_NZ$。
    - `group_type`支持取-1、0、2，`weight`为$FRACTAL\_NZ$格式时不支持取2。
    - `x`、`weight`、`y`数据类型为`torch.float16`/`torch.bfloat16`、数据格式为$ND$且`group_type`为0/-1时，`act_type`支持传入0/1/2；其余场景`act_type`仅支持传入0。
    - `split_item`支持取0/1/2/3，`x`输入多tensor且`y`输出多tensor时仅支持0，`x`单tensor且`y`多tensor时仅支持1，`y`单tensor时支持2/3。
    - 示例：[非量化场景示例](#非量化场景示例)、[非量化weightNZ示例](#非量化weightnz示例)
    - 图模式场景约束：
      - `x_dtype`只支持传入None，`weight_dtype`只支持传入None。
      - `x`输入多tensor且`y`输出多tensor场景下，`split_item`参数仅支持输入0。`x`输入单tensor且`y`输出多tensor场景下，`split_item`参数仅支持输入1。`y`输出单tensor场景下，`split_item`参数支持输入2/3。
      - 示例：[图模式非量化示例](#图模式非量化示例)
  <!-- end id69 -->

  <!-- npu="A3,910b" id70 -->
  - <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

    - `group_list`输入类型为`List[int]`时：

      | x | weight | bias | output_dtype | y |
      | --- | --- | --- | --- | --- |
      | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` |
      | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` |
      | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` | `torch.float32` |

    - `group_list`输入类型为`Tensor`时：

      | x | weight | bias | output_dtype | y |
      | --- | --- | --- | --- | --- |
      | `torch.float16` | `torch.float16` | `torch.float16` | None/`torch.float16` | `torch.float16` |
      | `torch.bfloat16` | `torch.bfloat16` | `torch.float32` | None/`torch.bfloat16` | `torch.bfloat16` |
      | `torch.float32` | `torch.float32` | `torch.float32` | None/`torch.float32`（仅x/weight/y均为单张量） | `torch.float32` |

    - `group_type`支持取-1、0、2，`x`不支持转置。
    - `act_type`取值范围为0-5。
  <!-- end id70 -->

  <!-- npu="310p" id71 -->
  - <term>Atlas推理系列产品</term>：
    - `per_token_scale`只支持`torch.float32`数据类型，`x`、`weight`、`bias`、`output_dtype`、`y`只支持`torch.float16`数据类型。
    - 输出`y`的n轴大小需要是16的倍数。
    - `group_type`仅支持取0，`act_type`仅支持传入0，`split_item`仅支持2或3。
    - 支持`weight`转置，不支持`x`转置。
  <!-- end id71 -->

### 静态量化（T-T、T-C）场景约束

支持的Tensor数量组合见[支持场景约束](#支持场景约束)。

<!-- npu="950" id72 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  - 输入`x`、`weight`与输出`y`的数据类型组合需满足下表（`group_list`输入类型为`Tensor`）：

    | x | weight | bias | scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int32`/None | `torch.int64` | `torch.float16`/`torch.bfloat16`/`torch.int8` | `torch.float16`/`torch.bfloat16`/`torch.int8` |
    | `torch.int8` | `torch.int8` | `torch.int32`/`torch.bfloat16`/`torch.float32`/None | `torch.bfloat16`/`torch.float32` | `torch.bfloat16` | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.int32`/`torch.float16`/`torch.float32`/None | `torch.float32` | `torch.float16` | `torch.float16` |
    | `torch.int8` | `torch.int8` | `torch.int32`/None | None/`torch.int64` | `torch.int32` | `torch.int32` |
    | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.int64`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.int64`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |

  - 单单单场景下`scale`与`per_token_scale`的shape限制（其中g为`group_list` shape大小，M与x的shape m一致，K与x的shape k一致，N与weight的shape n一致）：
    - `scale`为单tensor，2维或1维（C量化场景shape为（g, N）；T量化场景shape为（g, 1）或（g,））。`torch.int8`输出场景下，仅支持C量化，shape为（g, N）。
    - `per_token_scale`无需赋值。

  - `group_type`支持取0（M轴分组）、2（K轴分组）：`group_type`为0时`x`仅支持不转置、shape为（M, K）；`group_type`为2时`x`必须转置、shape为（K, M）。
  - `bias`仅在`x`输入`dtype`为`torch.int8`时支持。
  - `act_type`：当`x`与`weight`为`torch.int8`、量化模式为静态T-C量化且`scale`类型为`torch.float32`/`torch.bfloat16`时，支持传入0、1、2、4、5；其余全量化场景仅支持传入0。
  - `group_type`为2的单单单场景不支持`scale`为`torch.int64`。
  - 示例：[全量化weightnz示例](#全量化weightnz示例)
  - 图模式场景约束：
    - 若输出数据类型`y`为`torch.int8`，则不支持图模式运行。
    - 示例：[图模式全量化示例](#图模式全量化示例)、[图模式全量化weightnz示例](#图模式全量化weightnz示例)

<!-- end id72 -->

<!-- npu="A3,910b" id73 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

  - `group_list`输入类型为`List[int]`时：

    | x | weight | bias | scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int64` | `torch.int8` | `torch.int8` |

  - `group_list`输入类型为`Tensor`时：

    | x | weight | bias | scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.int64` | None/`torch.int8` | `torch.int8` |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |
    | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float32` | `torch.float16` | `torch.float16` |

  - 量化、伪量化仅支持`group_type`为-1和0场景；`x`不支持转置。
<!-- end id73 -->

<!-- npu="310p" id74 -->
- <term>Atlas推理系列产品</term>：暂不支持当前场景约束。
<!-- end id74 -->

### 动态量化（K-T、K-C、T-T、T-C）场景约束

支持的Tensor数量组合见[支持场景约束](#支持场景约束)。

<!-- npu="950" id75 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  - 输入`x`、`weight`与输出`y`的数据类型组合需满足下表（`group_list`输入类型为`Tensor`）：

    | 场景 | x | weight | bias | scale | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | K-T && K-C动态量化 | `torch.int8` | `torch.int8` | `torch.int32`/`torch.bfloat16`/`torch.float32`/None | `torch.bfloat16`/`torch.float32` | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` |
    | K-T && K-C动态量化 | `torch.int8` | `torch.int8` | `torch.int32`/`torch.float16`/`torch.float32`/None | `torch.float32` | `torch.float32` | `torch.float16` | `torch.float16` |
    | K-T && K-C动态量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | K-T && K-C动态量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | K-C动态量化 | `torch_npu.int4` | `torch_npu.int4` | None | `torch.uint64` | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | K-G量化 | `torch_npu.int4` | `torch_npu.int4` | None | `torch.uint64` | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |
    | T-C && T-T动态量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | T-C && T-T动态量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |

  - 单单单场景下`scale`与`per_token_scale`的shape限制（其中g为`group_list` shape大小，M与x的shape m一致，K与x的shape k一致，N与weight的shape n一致）：

    | group_type | 量化模式 | shape限制 |
    | --- | --- | --- |
    | 0/2 | K-T && K-C动态量化 | <li>scale为单tensor，2维或1维（C量化场景shape为（g, N）；T量化场景shape为（g, 1）或（g,））。</li><li>per_token_scale为单tensor，group_type等于0时，shape为1维（M,）；group_type等于2时，shape为2维（g, M）。</li> |
    | 0/2 | T-T动态量化 | scale和per_token_scale均为单tensor，2维时shape为（g, 1）或1维时shape为（g, ）。 |
    | 0/2 | T-C动态量化 | <li>scale为单tensor，shape为2维（g, N）。</li><li>per_token_scale为单tensor，2维或1维（2维时shape为（g, 1），1维时shape为（g, ））。</li> |
    | 0 | K-G量化 | <li>scale为单tensor，shape为3维（E, G, N），$G$必须要能整除$K$，且$k/G$需为偶数</li><li>per_token_scale为单tensor，shape为1维（M,）。</li> |

  - `group_type`支持取0、2。
  - `act_type`仅支持传入0（K-C动态量化且`scale`为`torch.float32`/`torch.bfloat16`时支持0、1、2、4、5）。
  - 单单单的全量化特殊场景说明：

    | `group_type` | 触发条件 | 量化模式选择 |
    | --- | --- | --- |
    | 0或2 | N=1，`scale`的shape为（g, 1），`weight`既可pertensor量化也可perchannel量化 | 优先选择pertensor量化模式 |
    | 2 | M=1，`per_token_scale`的shape为（g, 1），`x`既可pertoken量化也可pertensor量化 | 优先选择pertensor量化模式 |
    | 2 | K<128且N≤128，`scale`的shape为2维（g, 1） | 一律按G-B动态量化处理；如需非G-B动态量化，`scale`推荐1维（g,） |
    | 0 | g=M，且`per_token_scale`的shape为（g,）或（g, 1） | `per_token_scale`为（g,）（即K>128）时，`x`选择pertoken量化；为（g, 1）（即K≤128）时，按`weight`量化模式选择：perchannel/pertensor→pertensor，perblock→pergroup |
    | 2 | K<128、M≠1 | N≤128时，`x`选择pergroup量化；N>128时，按`weight`量化模式选择：perchannel/pertensor→pertoken，perblock→pergroup |
    | 2 | K<128、M=1，`per_token_scale`的shape为（g, 1） | N≤128时，`x`选择pergroup量化；N>128时，按`weight`量化模式选择：perchannel/pertensor→pertensor，perblock→pergroup |
    | — | 全量化`torch_npu.int4`场景，`x`为pertoken量化、`weight`为pergroup量化 | pergroup数G需整除K且K/G需为偶数；`weight`为weightNZ转置后，K/G按64对齐、K按64对齐、N按16对齐 |

  - 示例：[k-t动态量化示例](#k-t动态量化示例)、[全量化int8-动态k-c示例](#全量化int8-动态k-c示例)、[全量化int4-perchannel示例](#全量化int4-perchannel示例)
  - 图模式场景约束（暂未提供图模式示例）：
    - 若`x`和`weight`输入类型都为`torch_npu.int4`，则暂不支持图模式。
<!-- end id75 -->

<!-- npu="A3,910b" id76 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

  - `group_list`输入类型为`Tensor`时：

    | 场景 | x | weight | bias | scale | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | pertoken全量化 | `torch.int8` | `torch.int8` | `torch.int32` | `torch.bfloat16` | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` |
    | pertoken全量化 | `torch.int8` | `torch.int8` | `torch.int32` | `torch.float32` | `torch.float32` | `torch.float16` | `torch.float16` |

  - 量化、伪量化仅支持`group_type`为-1和0场景；`x`不支持转置。
<!-- end id76 -->

<!-- npu="310p" id77 -->
- <term>Atlas推理系列产品</term>：暂不支持当前场景约束。
<!-- end id77 -->

### 伪量化（perchannel、pergroup）场景约束

支持的Tensor数量组合见[支持场景约束](#支持场景约束)。

<!-- npu="950" id78 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  - 输入`x`、`weight`与输出`y`的数据类型组合需满足下表（`group_list`输入类型为`Tensor`）；K-CG伪量化场景（含S8S4 K-CG）的数据类型、shape及必传参数（`scale`、`per_token_scale`等）详见[伪量化（K-CG）场景约束](#伪量化k-cg场景约束)：

    | 场景 | x | weight | bias | antiquant_scale | antiquant_offset | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | perchannel伪量化 | `torch.float16`/`torch.bfloat16` | `torch.float8_e4m3fn`/`torch.float8_e5m2`/`torch_npu.hifloat8` | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | `torch.float16`/`torch.bfloat16` | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | perchannel伪量化 | `torch.float16`/`torch.bfloat16` | `torch.int8`/`torch.int32`（`torch_npu.int4`） | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | pergroup伪量化 | `torch.float16`/`torch.bfloat16` | `torch.int32`(`torch_npu.int4`) | `torch.float16`/`torch.float32`/None | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | S8S4 K-C伪量化 | `torch.int8` | `torch_npu.int4` | `torch.float32` | `torch.uint64` | None | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |

  - 各模式对`antiquant_scale`、`scale`和`per_token_scale`的shape限制如下（其中g为`group_list`的shape大小，M和K与x的m和k一致，N与weight的n一致）：

    | group_type | 量化模式 | shape限制 |
    | --- | --- | --- |
    | -1 | perchannel | antiquant_scale为多tensor，每个tensor为1维，shape为（ni）。 |
    | 0 | perchannel | antiquant_scale为单tensor，shape为2维（g, N）。 |
    | -1 | pergroup | <li>group_size支持32/64/128/256，pergroup数G=k/gs，要求K可以被gs整除。</li><li>antiquant_scale/antiquant_offset为多tensor，每个tensor为2维，shape为 (Gi, ni)，固定为非转置。</li> |
    | 0 | pergroup | <li>group_size支持32/64/128/256，pergroup数G=k/gs，要求K可以被gs整除。</li><li>antiquant_scale/antiquant_offset为单tensor，shape为3维（g, G, N），固定为非转置。</li> |
    | 0 | S8S4 K-C | scale和offset均为单tensor，shape为（g, 1, N）；per_token_scale为单tensor，shape为（M）。 |

  - `group_type`支持取-1、0，不支持激活函数计算（`act_type`仅支持传入0）。
  - `offset`仅S8S4 K-C支持`torch.float32`。
  - 当`weight`为`torch.float32`/`torch.int32`类型时，可通过[torch\_npu.npu\_convert\_weight\_to\_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)实现1个`torch.float32`承载8个`torch.float4_e2m1fn_x2`、1个`torch.int32`承载8个`torch_npu.int4`的输入，此时不传入`weight_dtype`；`weight`为`torch_npu.int4`时也需进行pack操作。
  - MxA8W4数据流支持`weight`使用`torch.uint8`承载`torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`（1个`torch.uint8`承载两个float4），该场景下静态图/动态图不支持E=1或K=64（E表示Group大小）。
  - pergroup场景，`weight`最后一维需为偶数，K为`group_size`整数倍；多多多场景下不支持`weight`非转置。
  - 图模式场景约束：
    - `offset`、`activation_input`、`activation_quant_scale`、`activation_quant_offset`只支持传入None。
    - `x_dtype`只支持传入None，`weight_dtype`支持传入None或`torch_npu.hifloat8`。
    - `x_dtype`为`torch.float16`/`torch.bfloat16`、`weight_dtype`为`torch_npu.int4`的pergroup量化场景下，不支持图模式运行。
  - 示例：[a16w4伪量化示例](#a16w4伪量化示例)、[pergroup伪量化示例](#pergroup伪量化示例)、[s8s4-k-c伪量化示例](#s8s4-k-c伪量化示例)
<!-- end id78 -->

<!-- npu="A3,910b" id79 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：

  - `group_list`输入类型为`List[int]`时：

    | x | weight | bias | antiquant_scale | antiquant_offset | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.int8` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` | `torch.float16` |
    | `torch.bfloat16` | `torch.int8` | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` | `torch.bfloat16` |

  - `group_list`输入类型为`Tensor`时：

    | x | weight | bias | antiquant_scale | antiquant_offset | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16` | `torch.int8`/`torch_npu.int4` | `torch.float16` | `torch.float16` | `torch.float16` | None/`torch.float16` | `torch.float16` |
    | `torch.bfloat16` | `torch.int8`/`torch_npu.int4` | `torch.float32` | `torch.bfloat16` | `torch.bfloat16` | None/`torch.bfloat16` | `torch.bfloat16` |

  - 若`weight`的类型为`torch.int8`，仅支持perchannel模式；若`weight`的类型为`torch_npu.int4`，支持perchannel和pergroup两种模式。若为pergroup，pergroup数G或G<sub>i</sub>必须要能整除对应的k<sub>i</sub>。若`weight`为多tensor，定义pergroup长度s<sub>i</sub> = k<sub>i</sub> / G<sub>i</sub>，要求所有s<sub>i</sub>(i=1,2,...g)都相等。
  - 若`weight`的类型为`torch_npu.int4`，则`weight`中每一组tensor的最后一维大小都应是偶数。`weight`<sub>i</sub>的最后一维指`weight`不转置时`weight`<sub>i</sub>的N轴或当`weight`转置时`weight`<sub>i</sub>的K轴。并且在pergroup场景下，当`weight`转置时，要求pergroup长度s<sub>i</sub>是偶数。tensor转置：指若tensor shape为\[M,K\]时，则stride为\[1,M\]，数据排布为\[K,M\]的场景，即非连续tensor。
  - 当前PyTorch不支持`torch_npu.int4`类型数据，需要使用时可以通过[torch\_npu.npu\_quantize](torch_npu-npu_quantize.md)接口使用`torch.int32`数据表示`torch_npu.int4`。
  - 量化、伪量化仅支持`group_type`为-1和0场景；`x`不支持转置。
<!-- end id79 -->

<!-- npu="310p" id80 -->
- <term>Atlas推理系列产品</term>：暂不支持当前场景约束。
<!-- end id80 -->

### 动态量化（G-B、MX）场景约束

支持的Tensor数量组合见[支持场景约束](#支持场景约束)。

<!-- npu="950" id81 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  - 输入`x`、`weight`与输出`y`的数据类型组合需满足下表（`group_list`输入类型为`Tensor`）：

    | 场景 | x | weight | bias | scale | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | MxA8W8（全量化） | $ND$支持`torch.float8_e4m3fn`/`torch.float8_e5m2`，$FRACTAL\_NZ$支持`torch.float8_e4m3fn` | $ND$支持`torch.float8_e4m3fn`/`torch.float8_e5m2`，$FRACTAL\_NZ$支持`torch.float8_e4m3fn` | None | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | MxA4W4（全量化） | $ND$支持`torch.float4_e2m1fn_x2`，$FRACTAL\_NZ$支持`torch_npu.float4_e1m2fn_x2` | $ND$支持`torch.float4_e2m1fn_x2`，$FRACTAL\_NZ$支持`torch_npu.float4_e1m2fn_x2` | `torch.float32`/None | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | G-B动态量化 | `torch_npu.hifloat8` | `torch_npu.hifloat8` | None | `torch.float32` | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |
    | G-B动态量化 | `torch.float8_e4m3fn`/`torch.float8_e5m2` | `torch.float8_e4m3fn`/`torch.float8_e5m2` | None | `torch.float32` | `torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` | `torch.float16`/`torch.bfloat16`/`torch.float32` |

  - 单单单场景下`scale`与`per_token_scale`的shape限制（其中g为`group_list` shape大小，M与x的shape m一致，K与x的shape k一致，N与weight的shape n一致）：

    | group_type | 量化模式 | shape限制 |
    | --- | --- | --- |
    | 0 | MX量化 | 计算公式（5）中gsm = gsn = 1，gsk = 32。scale和per_token_scale均为单tensor，scale的shape支持（g, N, ceil(K/64), 2）或 (g, ceil(K/64), N, 2)，per_token_scale的shape仅支持（M, ceil(K/64), 2），输入x为`torch.float4_e2m1fn_x2`时，需要满足K为偶数并且K不为2。当weight非转置时还需满足N为偶数。 |
    | 2 | MX量化 | 计算公式（5）中gsm = gsn = 1，gsk = 32。scale和per_token_scale均为单tensor。scale shape仅支持(K//64+g, N, 2)，per_token_scale shape仅支持(K//64 + g, M, 2)，输入x为`torch.float4_e2m1fn_x2`且K为1时，需要满足M和N不为2。 |
    | 0 | G-B动态量化 | 计算公式（5）中gsm = 1，gsn = gsk = 128。scale为单tensor，scale为3维，shape为（g, ceil(K/128), ceil(N/128)）或（g, ceil(N/128), ceil(K/128)）。<br>per_token_scale为单tensor，per_token_scale为2维，shape为（M,ceil(K/128)）。 |
    | 2 | G-B动态量化 | 计算公式（5）中gsm = 1，gsn = gsk = 128。scale为单tensor，scale为2维，shape为（K//128+g, ceil(N/128)）。<br>per_token_scale为单tensor，per_token_scale为2维，shape为（K//128+g, M） |

  - MX量化的M轴分组场景时，`scale`每个张量的维度需要为4，在`weight`多tensor输入场景下每个张量维度为3维，并且与`weight`具有相同的转置属性；MX量化的K轴分组场景时，`scale`每个张量的维度需要为3，并且与`weight`具有相同转置属性。
  - MX量化时，`per_token_scale`每个张量需要为3维，且与`x`具有相同的转置属性。
  - `group_type`支持取0（M轴分组）、2（K轴分组）：`group_type`为0时`x`仅支持不转置、shape为（M, K）；`group_type`为2时`x`必须转置、shape为（K, M）。
  - MX全量化且`weight`为$FRACTAL\_NZ$格式时，支持多`weight`多tensor输入，列表最大长度为128。
  - `group_type`为2的单单单场景不支持`scale`为`torch.int64`；`group_type`为2的单单单场景仅支持ND进ND出。
  - scale NZ亲和格式使用约束：
    - 仅支持MX量化场景，`scale`数据类型为`torch.float8_e8m0fnu`。
    - `scale`使用NZ亲和格式时，`weight`必须为$FRACTAL\_NZ$格式。
    - `x`和`weight`数据类型仅支持`torch.float8_e4m3fn`。
    - 仅支持`weight`不转置场景。
    - 仅支持`group_type`为0，即M轴分组。
    - 不支持图模式，仅支持单算子模式。
    - 单tensor场景下，scale NZ亲和格式的storage shape为$[E, \lceil N/16 \rceil, \lceil K/64 \rceil, 16, 2]$；多tensor场景下，每个scale tensor的storage shape为$[\lceil N_i/16 \rceil, \lceil K_i/64 \rceil, 16, 2]$。
  - 示例：[MX量化示例](#MX量化示例)、[G-B动态量化示例](#g-b动态量化示例)、[MxA4W4全量化示例](#MX-a4w4全量化示例)
  - 图模式场景约束：
    - MX全量化且`weight`输入格式为$FRACTAL\_NZ$格式，若`weight`多tensor输入，则不支持图模式运行。
    - 示例：[图模式G-B动态量化示例](#图模式g-b动态量化示例)、[图模式MxA4W4全量化示例](#图模式MX-a4w4全量化示例)
<!-- end id81 -->

<!-- npu="A3,910b" id82 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持当前场景约束。
<!-- end id82 -->

<!-- npu="310p" id83 -->
- <term>Atlas推理系列产品</term>：暂不支持当前场景约束。
<!-- end id83 -->

### 伪量化（MX）场景约束

支持的Tensor数量组合见[支持场景约束](#支持场景约束)。

<!-- npu="950" id84 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  - 输入`x`、`weight`与输出`y`的数据类型组合需满足下表（`group_list`输入类型为`Tensor`）：

    | x | weight | bias | antiquant_scale | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- |
    | `torch.float16`/`torch.bfloat16` | `torch.float32`（`torch.float4_e2m1fn_x2`） | `torch.float16`/`torch.bfloat16`/`torch.float32`/None | `torch.float8_e8m0fnu` | None | `torch.float16`/`torch.bfloat16`/None | `torch.float16`/`torch.bfloat16` |
    | `torch.float8_e4m3fn` | `torch.uint8`/`torch.int8`（`torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`） | `torch.float16`/`torch.bfloat16` | `torch.float8_e8m0fnu` | `torch.float8_e8m0fnu` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |

  - `antiquant_scale`与`per_token_scale`的shape限制（其中g为`group_list`的shape大小，M和K分别与`x`的m和k一致，N与`weight`的n一致；`group_type`为0，pergroup数G=K/32，要求K可以被32整除）：
    - `x`为`torch.float16`/`torch.bfloat16`、`weight`为`torch.float32`（`torch.float4_e2m1fn_x2`）场景：`antiquant_scale`为单tensor，shape为3维（g, K/32, N）。
    - `x`为`torch.float8_e4m3fn`、`weight`为`torch.uint8`/`torch.int8`（承载`torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`）场景：
      - G需为偶数。
      - 单单单场景：`antiquant_scale`为单tensor，shape为4维（g, N, K/64, 2）。
      - 单单单场景：`per_token_scale`为单tensor，shape为3维（M, K/64, 2）。
      - 单多单场景：每个`weight`对应一个`antiquant_scale`（列表长度与`weight`相同），`antiquant_scale`中每个tensor为3维、shape为（N, K/64, 2），转置属性与对应`weight`相同。
      - 单多单场景：`per_token_scale`对应单`x`，为单tensor、shape为3维（M, K/64, 2）。

  - `group_type`支持取-1、0；`group_list_type`支持取0、1，仅全量化场景且`group_type`为0时支持取2。
  - 不支持激活函数计算（`act_type`仅支持传入0）。
  - `weight`为$FRACTAL\_NZ$格式时适用[WeightNZ场景说明](#通用约束)。
  - 转置要求按上表两种数据类型组合区分：
    - `x`为`torch.float16`/`torch.bfloat16`、`weight`为`torch.float32`（承载`torch.float4_e2m1fn_x2`）：仅支持`x`不转置、`weight`不转置。
    - `x`为`torch.float8_e4m3fn`、`weight`为`torch.uint8`/`torch.int8`（承载`torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`）：仅支持`x`不转置、`weight`转置。
  - `x`为`torch.float8_e4m3fn`、`weight`为`torch.uint8`/`torch.int8`（承载`torch.float4_e2m1fn_x2`/`torch_npu.float4_e1m2fn_x2`）：必须显式传入`output_dtype`，`bias`的数据类型须与`output_dtype`一致。
  - `weight`的最后2维需要满足32B对齐；`weight`为`torch.float32`/`torch.int32`类型时需按[伪量化场景约束](#伪量化perchannelpergroup场景约束)进行pack操作。
  - 全量化MX量化且输入`w`为$FRACTAL\_NZ$格式时，支持单多单场景，此时不支持传入`bias`。
  - 示例：[MX伪量化示例](#MX伪量化示例)、[MxA8W4伪量化单多单示例](#MXa8w4伪量化单多单示例)
  - 图模式示例见[图模式MX伪量化示例](#图模式MX伪量化示例)，其约束与本节一致。
<!-- end id84 -->

<!-- npu="A3,910b" id85 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持当前场景约束。
<!-- end id85 -->

<!-- npu="310p" id86 -->
- <term>Atlas推理系列产品</term>：暂不支持当前场景约束。
<!-- end id86 -->

### 伪量化（K-CG）场景约束

支持的Tensor数量组合见[支持场景约束](#支持场景约束)。

<!-- npu="950" id87 -->
- <term>Ascend 950PR&950DT系列产品</term>：

  - 输入`x`、`weight`与输出`y`的数据类型组合需满足下表（`group_list`输入类型为`Tensor`）：

    | x | weight | bias | antiquant_scale | scale | per_token_scale | output_dtype | y |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | `torch.int8` | `torch.int32`（`torch_npu.int4`） | `torch.float32` | `torch.float16` | `torch.float32` | `torch.float32` | `torch.float16`/`torch.bfloat16` | `torch.float16`/`torch.bfloat16` |

  - `group_list_type`仅支持取1，此时`group_list`中的数值表示分组轴上每组大小。
  - `antiquant_scale`为单tensor，shape为3维（g, G, N），其中pergroup数G=K/gs，gs支持取值为128、192、256、512，要求K可以被gs整除。
  - `per_token_scale`为单tensor，shape为1维（M）。
  - 常规K-CG子场景：`scale`为单tensor，shape为2维（g, N）。
  - S8S4 K-CG子场景：`scale`为单tensor，shape为3维（g, K/256, N）。
  - `group_type`支持取0（M轴分组）。
  - 不支持激活函数计算（`act_type`仅支持传入0）。
  - `weight`的最后2维需要满足32B对齐。
<!-- end id87 -->

<!-- npu="A3,910b" id88 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：暂不支持当前场景约束。
<!-- end id88 -->

<!-- npu="310p" id89 -->
- <term>Atlas推理系列产品</term>：暂不支持当前场景约束。
<!-- end id89 -->

## 调用示例

- 以下示例按量化场景索引：

  - 通用示例（适用于所有产品）：[非量化场景示例](#非量化场景示例)、[图模式非量化示例](#图模式非量化示例)

  <!-- npu="950" id90 -->
  - 非量化：[非量化weightnz示例](#非量化weightnz示例)
  - 静态量化（T-T、T-C）：[全量化weightnz示例](#全量化weightnz示例)
  - 动态量化（K-T、K-C、T-T、T-C）：[k-t动态量化示例](#k-t动态量化示例)、[全量化int8-动态k-c示例](#全量化int8-动态k-c示例)、[全量化int4-perchannel示例](#全量化int4-perchannel示例)
  - 伪量化（perchannel、pergroup）：[a16w4伪量化示例](#a16w4伪量化示例)、[pergroup伪量化示例](#pergroup伪量化示例)、[s8s4-k-c伪量化示例](#s8s4-k-c伪量化示例)
  - 动态量化（G-B、MX）：[MX量化示例](#MX量化示例)、[G-B动态量化示例](#g-b动态量化示例)、[MxA4W4全量化示例](#MX-a4w4全量化示例)
  - 伪量化（MX）：[MX伪量化示例](#MX伪量化示例)、[MX伪量化MxA8W4单多单示例](#MXa8w4伪量化单多单示例)
  - 图模式（各场景）：[图模式全量化示例](#图模式全量化示例)、[图模式全量化weightnz示例](#图模式全量化weightnz示例)、[图模式MX伪量化示例](#图模式MX伪量化示例)、[图模式G-B动态量化示例](#图模式g-b动态量化示例)、[图模式MxA4W4全量化示例](#图模式MX-a4w4全量化示例)
  <!-- end id90 -->

- 单算子模式调用

  - 非量化场景示例：<a id="非量化场景示例"></a>

    ```python
    import torch
    import torch_npu
    M = 256
    K = 256
    N = 256
    g = 2
    x = torch.randint(-1, 1, (M, K), dtype=torch.float16).npu()
    weight = torch.randint(-1, 1, (g, K, N), dtype=torch.float16).npu().transpose(1,2)
    group_list = torch.Tensor([128, 256]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)
    ```

  <!-- npu="950" id91 -->
  - <term>Ascend 950PR&950DT系列产品</term>：`torch_npu.hifloat8`输入，K-T动态量化场景<a id="k-t动态量化示例"></a>

    ```python
    import torch
    import torch_npu

    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, 1), dtype=torch.float32).npu()
    x1_scale = torch.randint(-1, 1, (M,), dtype=torch.float32).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch_npu.hifloat8, weight_dtype=torch_npu.hifloat8, group_list_type=0)
    ```
  <!-- end id91 -->
  <!-- npu="950" id92 -->
  - <term>Ascend 950PR&950DT系列产品</term>：MX量化场景示例-MXFP8<a id="MX量化示例"></a>

    ```python
    import math
    import torch
    import torch_npu
    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, N, math.ceil(K/64), 2), dtype=torch.int8).npu().transpose(1,2)
    x1_scale = torch.randint(-1, 1, (M, math.ceil(K/64), 2), dtype=torch.int8).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)
    ```
  <!-- end id92 -->
  <!-- npu="950" id93 -->
  - <term>Ascend 950PR&950DT系列产品</term>：G-B动态量化场景示例<a id="g-b动态量化示例"></a>

    ```python
    # G-B
    import math
    import torch
    import torch_npu

    M = 576
    K = 512
    N = 7168
    g = 4
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu()
    weight = torch.randint(-1, 1, (g, N, K), dtype=torch.int8).to(torch.float8_e4m3fn).npu().transpose(1,2)
    x2_scale = torch.randint(-1, 1, (g, math.ceil(N/128), math.ceil(K/128)), dtype=torch.float32).npu().transpose(1,2)
    x1_scale = torch.randint(-1, 1, (M, math.ceil(K/128)), dtype=torch.float32).npu()

    group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[x2_scale], per_token_scale = [x1_scale], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0)
    ```
  <!-- end id93 -->
  <!-- npu="950" id94 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化weightNZ场景示例（x/weight为`torch.int8`）<a id="全量化weightnz示例"></a>

    ```python
    import torch
    import torch_npu
    M = 4
    K = 4
    N = 5
    g = 2
    x = torch.randint(-1, 1, (M, K), dtype=torch.int8).npu()
    weight1 = torch.randint(-1, 1, (g, K, N), dtype=torch.int8).npu()
    weightnz = torch_npu.npu_format_cast(weight1, 29)
    scale = torch.randint(-1, 1, (g, N), dtype=torch.bfloat16).npu()
    pertoken = torch.randint(-1, 1, (M,), dtype=torch.float32).npu()
    group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
    split_item = 3
    bias = torch.randint(-1, 1, (g, N), dtype=torch.int32).npu()

    npu_out = torch_npu.npu_grouped_matmul([x], [weightnz], scale=[scale], bias = [bias], per_token_scale = [pertoken], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0)
    ```
  <!-- end id94 -->
  <!-- npu="950" id95 -->
  - <term>Ascend 950PR&950DT系列产品</term>：非量化weightNZ场景示例<a id="非量化weightnz示例"></a>

    ```python
    import torch
    import torch_npu
    M = 1
    K = 256
    N = 256
    g = 1
    x = torch.randint(-1, 1, (M, K), dtype=torch.float16).npu()
    weight = torch.randint(-1, 1, (g, K, N), dtype=torch.float16).npu().transpose(1,2)

    group_list = torch.Tensor([1]).to(torch.int64).npu()
    split_item = 2
    weight_flag = 29
    weight = torch_npu.npu_format_cast(weight, weight_flag)
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)
    ```
  <!-- end id95 -->
  <!-- npu="950" id96 -->
  - <term>Ascend 950PR&950DT系列产品</term>：MX（A4W4）全量化场景示例-MXFP4<a id="MX-a4w4全量化示例"></a>

    ```python
    import math
    import torch
    import torch_npu
    M = 64
    N = 128
    K = 128
    g = 1
    #使用uint8代替两个mxfp4，此处uint8的内轴需要扩展2倍才是真实的mxfp4的内轴
    x = torch.randint(1, 3, (M, int(K/2)), dtype=torch.uint8).npu()

    weight = torch.randint(1, 3, (g, K, int(N/2)), dtype=torch.uint8).npu()

    scale = torch.randint(-1, 1, (g, math.ceil(K/64),N,2), dtype=torch.int8).npu()
    per_token_scale = torch.randint(-1, 1, (M, math.ceil(K/64),2), dtype=torch.int8).npu()

    group_list = torch.Tensor([1]).to(torch.int64).npu()
    split_item = 2
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[scale], per_token_scale = [per_token_scale], group_list=group_list,
    split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch.float4_e2m1fn_x2, weight_dtype=torch.float4_e2m1fn_x2,
    scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)
    ```
  <!-- end id96 -->
  <!-- npu="950" id114 -->
  - <term>Ascend 950PR&950DT系列产品</term>：mx（A4W4）全量化K轴分组场景示例-mxfp4（`group_type`=2，TN，`x`物理排布为\[K, M/2\]的转置视图，`per_token_scale`物理排布为\[K//64+g, M, 2\]的转置视图）

    ```python
    import torch
    import torch_npu
    M = 128
    N = 256
    K = 160            # 各组K长度之和，此处两组分别为64和96
    g = 2
    S = K // 64 + g    # scale/per_token_scale的K轴行数
    # 使用uint8承载mxfp4（1字节承载2个float4），物理排布[K, M/2]，转置视图为[M/2, K]
    x = torch.randint(1, 3, (K, M // 2), dtype=torch.uint8).npu().view(torch.float4_e2m1fn_x2).t()
    weight = torch.randint(1, 3, (K, N // 2), dtype=torch.uint8).npu().view(torch.float4_e2m1fn_x2)
    scale = torch.randint(124, 130, (S, N, 2), dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)
    # per_token_scale物理排布[S, M, 2]，转置视图为[M, S, 2]
    per_token_scale = torch.randint(124, 130, (S, M, 2), dtype=torch.uint8).npu().view(torch.float8_e8m0fnu).transpose(0, 1)

    group_list = torch.Tensor([64, 160]).to(torch.int64).npu()
    split_item = 3
    npu_out = torch_npu.npu_grouped_matmul([x], [weight], scale=[scale], per_token_scale = [per_token_scale], group_list=group_list,
    split_item=split_item, group_type=2, output_dtype=torch.bfloat16, x_dtype=torch.float4_e2m1fn_x2, weight_dtype=torch.float4_e2m1fn_x2,
    scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)
    ```
  <!-- end id114 -->
  <!-- npu="950" id97 -->
  - <term>Ascend 950PR&950DT系列产品</term>：（A16W4）伪量化场景示例<a id="a16w4伪量化示例"></a>

    ```python
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    def call_pta(x, weight, bias, antiquant_scale, antiquant_offset, group_type, split_item, group_list, group_list_type):
            for i in range(len(weight)):
                weight[i] = weight[i].transpose(-2, -1)
            return torch_npu.npu_grouped_matmul(x, weight, bias=bias, antiquant_scale=antiquant_scale,
                                                antiquant_offset=antiquant_offset, group_list=group_list,
                                                split_item=split_item, group_type=group_type, output_dtype=torch.float16,
                                                group_list_type=group_list_type)

    def main():
        g, m, k, n = 2, 5, 64, 128
        x1 = torch.ones(m, k, dtype=torch.float16)
        x = [x1.npu()]
        weight1 = torch.ones(g, n, k, dtype=torch.int32)
        weight_packed = torch_npu.npu_convert_weight_to_int4pack(weight1.npu())
        weight = [weight_packed.npu()]
        bias1 = torch.ones(g, n, dtype=torch.float16)
        bias = [bias1.npu()]
        antiquant_scale1 = torch.ones(g, n, dtype=torch.float16)
        print(f"antiquant_scale1 shape {antiquant_scale1.shape}")
        antiquant_scale = [antiquant_scale1.npu()]
        antiquant_offset1 = torch.zeros(g, n, dtype=torch.float16)
        antiquant_offset = [antiquant_offset1.npu()]
        group_list = torch.Tensor([2, 3]).to(torch.int64).npu()
        group_list_type = 1
        split_item = 3
        group_type = 0
        custom_output = call_pta(x, weight, bias, antiquant_scale, antiquant_offset, group_type, split_item, group_list,
                              group_list_type)

    if __name__ == '__main__':
        main()
    ```
  <!-- end id97 -->
  <!-- npu="950" id98 -->
  - <term>Ascend 950PR&950DT系列产品</term>MX伪量化场景示例（x为`torch.float8_e4m3fn`，weight使用`torch.uint8`承载`torch.float4_e2m1fn_x2`（1个`torch.uint8`承载两个float4\_e2m1），antiquant\_scale和per\_token\_scale使用`torch.uint8`承载`torch.float8_e8m0fnu`）<a id="MX伪量化示例"></a>

    ```python
    import torch
    import torch_npu

    MX_GROUP_SIZE = 32

    class NetPTA(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, weight, per_token_scale, antiquant_scale, bias, group_list, group_list_type,
                    output_dtype=torch.bfloat16):
            weight = weight.transpose(-2, -1)
            antiquant_scale = antiquant_scale.transpose(-3, -2)
            bias = [bias] if bias is not None else None
            output = torch_npu.npu_grouped_matmul([x], [weight], bias=bias, group_list=group_list, group_type=0,
                                                  group_list_type=group_list_type, per_token_scale=[per_token_scale],
                                                  antiquant_scale=[antiquant_scale], split_item=2,
                                                  output_dtype=output_dtype, per_token_scale_dtype=torch_npu.float8_e8m0fnu, weight_dtype=torch.float4_e2m1fn_x2)
            return output

    def main():
        g, m, k, n = 4, 16, 512, 128
        output_dtype = torch.bfloat16
        x = torch.ones((m, k)).to(torch.float8_e4m3fn).npu()
        weight = torch.randint(0, 256, (g, n, k//2), dtype=torch.uint8).npu()
        weight = torch_npu.npu_format_cast(weight, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2)
        per_token_scale = torch.randint(124, 130, (m, k // MX_GROUP_SIZE // 2 , 2)).to(torch.uint8).npu()
        antiquant_scale = torch.randint(124, 130, (g, n, k // MX_GROUP_SIZE // 2, 2), dtype=torch.uint8).npu()
        bias = torch.randint(-5, 5, (g, n)).to(output_dtype).npu()
        group_list = torch.Tensor([2, 4, 6, 16]).to(torch.int64).npu()
        group_list_type = 0
        model = NetPTA().npu()
        out = model(x, weight, per_token_scale, antiquant_scale, bias, group_list, group_list_type, output_dtype)
        print("output")
        print(out[0].cpu())

    if __name__ == '__main__':
        main()
    ```
  <!-- end id98 -->
  <!-- npu="950" id99 -->
  - <term>Ascend 950PR&950DT系列产品</term>：pergroup伪量化场景示例（x为`torch.float16`/`torch.bfloat16`，weight使用`torch.int32`承载`torch_npu.int4`，即1个`torch.int32`承载8个`torch_npu.int4`）<a id="pergroup伪量化示例"></a>

    ```python
    import torch
    import torch_npu

    E, M, N, K, group_size = 2, 32, 128, 256, 32
    group_list = [M // E] * E
    xdtype = torch.bfloat16
    is_multi = True
    transpose_weight = True

    if __name__ == "__main__":
        assert not (is_multi and not transpose_weight), "multi/multi/multi and non-transpose is not supported"

        x = [torch.rand((group_list[i], K), dtype=xdtype, device="npu") for i in range(E)] if is_multi else \
            [torch.rand((M, K), dtype=xdtype, device="npu")]
        weight = [torch_npu.npu_convert_weight_to_int4pack(torch.randint(-8, 7, (N, K) if transpose_weight else (K, N), dtype=torch.int32, device="npu"))
                    for _ in range(E)] if is_multi else \
                  [torch_npu.npu_convert_weight_to_int4pack(torch.randint(-8, 7, (E, N, K) if transpose_weight else (E, K, N), dtype=torch.int32, device="npu"))]
        weight = [w.transpose(-1, -2) if transpose_weight else w for w in weight]
        bias = [torch.rand((N, ), dtype=xdtype if xdtype == torch.float16 else torch.float32, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, N), dtype=xdtype if xdtype == torch.float16 else torch.float32, device="npu")]
        antiquantScale = [torch.rand((K // group_size, N), dtype=xdtype, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, K // group_size, N), dtype=xdtype, device="npu")]
        antiquantOffset = [torch.rand((K // group_size, N), dtype=xdtype, device="npu") for _ in range(E)] if is_multi else \
                [torch.rand((E, K // group_size, N), dtype=xdtype, device="npu")]

        out = torch_npu.npu_grouped_matmul(x, weight, bias=bias,
                                            antiquant_scale=antiquantScale, antiquant_offset=antiquantOffset,
                                            group_list=None if is_multi else torch.Tensor(group_list).to(torch.int64).npu(),
                                            split_item=0 if is_multi else 2,
                                            group_type=-1 if is_multi else 0,
                                            group_list_type=1)
        print([o.shape for o in out])
    ```

  <!-- end id99 -->
  <!-- npu="950" id100 -->
  - <term>Ascend 950PR&950DT系列产品</term>：S8S4 K-C伪量化示例<a id="s8s4-k-c伪量化示例"></a>

    ```python
    import os
    import numpy as np
    import torch
    import torch_npu

    def encode_a8w4_scale(scale_fp32: torch.Tensor) -> torch.Tensor:
        """Encode FP32 scales as the UINT64 carrier required by A8W4."""
        scale_np = scale_fp32.cpu().contiguous().numpy().astype(np.float32)
        scale_bits = scale_np.view(np.uint32)
        scale_bits &= np.uint32(0xFFFFE000)
        encoded = scale_bits.astype(np.uint64)
        encoded |= np.uint64(1 << 46)
        return torch.from_numpy(encoded.copy())

    def main() -> None:
        device_id = int(os.getenv("ASCEND_DEVICE_ID", "0"))
        torch.npu.set_device(device_id)

        m, n, k, expert_num = 64, 128, 256, 1

        x = torch.randint(-128, 128, (m, k), dtype=torch.int8).npu()

        # INT32 carries logical INT4 values in [-8, 7] before NPU packing.
        weight_i32 = torch.randint(
            -8, 8, (expert_num, k, n), dtype=torch.int32
        ).npu()
        weight_i4 = torch_npu.npu_convert_weight_to_int4pack(weight_i32)

        # A non-empty offset selects Per-channel mode. Both tensors are [E, 1, N].
        scale_fp32 = torch.rand(expert_num, 1, n) * 0.01 + 0.001
        scale = encode_a8w4_scale(scale_fp32).npu()
        offset = torch.zeros((expert_num, 1, n), dtype=torch.float32).npu()
        per_token_scale = (torch.rand(m, dtype=torch.float32) * 0.01 + 0.001).npu()
        bias = torch.zeros((expert_num, n), dtype=torch.float32).npu()
        group_list = torch.tensor([m], dtype=torch.int64).npu()

        output = torch_npu.npu_grouped_matmul(
            [x],
            [weight_i4],
            bias=[bias],
            scale=[scale],
            offset=[offset],
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=[per_token_scale],
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
            act_type=0,
            tuning_config=[0],
            # Asymmetric S8S4 K-C (offset is non-empty) requires FP16 output.
            output_dtype=torch.float16,
        )[0]

        torch.npu.synchronize()
    ```
  <!-- end id100 -->
  <!-- npu="950" id101 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化`torch_npu.int4`，perchannel示例<a id="全量化int4-perchannel示例"></a>

    ```python
    import numpy as np
    import torch
    import torch_npu

    # npu_quantize 打 int4 包时需要开启内部格式 (allow_internal_format)
    torch_npu.npu.config.allow_internal_format = True


    def main():
        m, k, n, e = 2048, 1024, 1024, 16
        out_dtype = torch.bfloat16
        npu = "npu"
        print(f"[run] perchannel  shape M={m} K={k} N={n} E={e}  out={out_dtype}")

        x_int = torch.full((m, k), 1, dtype=torch.int8)
        w_int = torch.full((e, k, n), 1, dtype=torch.int8)
        scale_ref = torch.full((e, 1, n), 0.1, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        pt_scale = torch.full((m,), 1.0, dtype=torch.float32)
        group_list = torch.arange(1, e + 1) * (m // e)

        # ---- NPU 侧: int4 量化打包 (ND, 非转置) ----
        x_quant = torch_npu.npu_quantize(
            x_int.to(torch.float32).to(npu),
            torch.tensor([1.], device=npu), None, torch.quint4x2, -1, False)
        weight_quant = torch_npu.npu_quantize(
            w_int.to(torch.float32).to(npu),
            torch.tensor([1.], device=npu), None, torch.quint4x2, -1, False)

        # ---- scale 打包: (E,1,N) fp32 比特位放入 int64 低 32 位, 高 32 位=0 ----
        scale_arr = scale_ref.cpu().numpy().astype(np.float32)
        scale_arr.dtype = np.uint32                               # 比特重解释, 不改数值
        packed = np.zeros((e, 1, n * 2), dtype=np.uint32)
        packed[..., ::2] = scale_arr
        packed.dtype = np.int64
        scale_i64 = torch.from_numpy(packed).to(npu)

        y = torch_npu.npu_grouped_matmul(
            [x_quant], [weight_quant],
            bias=None,
            scale=[scale_i64],
            offset=None,
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=[pt_scale.to(npu)],
            group_list=group_list.to(npu),
            split_item=3,
            group_type=0,
            group_list_type=0,
            act_type=0,
            output_dtype=out_dtype,
        )[0]
        torch_npu.npu.synchronize()

    if __name__ == "__main__":
        main()
    ```
  <!-- end id101 -->
  <!-- npu="950" id102 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化`torch.int8`，动态K-C量化支持激活示例<a id="全量化int8-动态k-c示例"></a>

    ```python
    import torch
    import torch_npu
    x = torch.randint(-1, 1, (4, 4), dtype=torch.int8).npu()
    weight1 = torch.randint(-1, 1, (2, 4, 5), dtype=torch.int8).npu()
    scale = torch.randint(-1, 1, (2, 5), dtype=torch.float32).npu()
    pertoken = torch.randint(-1, 1, (4,), dtype=torch.float32).npu()
    group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
    split_item = 3
    bias = torch.randint(-1, 1, (2, 5), dtype=torch.int32).npu()
    npu_out = torch_npu.npu_grouped_matmul([x], [weight1], scale=[scale], bias = [bias],    per_token_scale = [pertoken],group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, act_type=1)
    print("npu_out[0].shape: ", npu_out[0].shape)
    ```
  <!-- end id102 -->
  <!-- npu="950" id103 -->
  - <term>Ascend 950PR&950DT系列产品</term>：伪量化MxA8W4，单多单场景<a id="MXa8w4伪量化单多单示例"></a>

    ```python
    import torch
    import torch_npu
    MX_GROUP_SIZE = 32

    def test():
        # 1. 基础参数
        g, m, k, n = 4, 16, 512, 128
        output_dtype = torch.bfloat16
        # 2. 生成 Input x: (m, k)
        x_list = [torch.ones((m, k)).to(torch.float8_e4m3fn).npu()]
        # 3. 生成 group_list
        m_per_group = m // g
        group_list = torch.tensor([m_per_group] * g, dtype=torch.int64).npu()
        group_list_type = 1  # group_list中数值为分组轴上每组大小。
        # 4. 生成参数
        weight_list = []
        antiquant_scale_list = []
        bias_list = []
        for i in range(g):
            # 单组权重形状：(n, k)
            w = torch.randint(0, 256, (n, k//2), dtype=torch.uint8).npu()
            w_cast = torch_npu.npu_format_cast(w, 29,
                                                customize_dtype=torch.float8_e4m3fn,
                                                input_dtype=torch.float4_e2m1fn_x2).t()
            weight_list.append(w_cast)
            # 假设 scale 也是按组或全局，这里生成列表
            s_k = k // MX_GROUP_SIZE // 2
            s = torch.randint(124, 130, (n, s_k, 2), dtype=torch.uint8).npu().permute(1, 0, 2)
            antiquant_scale_list.append(s)
            # Bias 列表 (g 个 (n))
            b = torch.randint(-5, 5, (n,), dtype=output_dtype).npu()
            bias_list.append(b)
        # 5. 准备 per_token_scale
        per_token_scale_list = [torch.randint(124, 130, (m, k // MX_GROUP_SIZE // 2, 2), dtype=torch.uint8).npu()]
        # 6. 执行
        out_list = torch_npu.npu_grouped_matmul(
            x_list,
            weight_list,
            bias=bias_list,
            group_list=group_list,
            group_type=0,
            group_list_type=group_list_type,
            per_token_scale=per_token_scale_list,
            antiquant_scale=antiquant_scale_list,
            split_item=2,
            output_dtype=output_dtype,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            weight_dtype=torch.float4_e2m1fn_x2
        )
    if __name__ == '__main__':
        test()
    ```
  <!-- end id103 -->

- 图模式调用

  - 非量化场景示例：（约束见[非量化场景约束](#非量化场景约束)）<a id="图模式非量化示例"></a>

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    M = 256
    N = 256
    K = 256
    g = 2
    class GMMModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, group_list, split_item):
            return torch_npu.npu_grouped_matmul(x, weight, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.float16, group_list_type=0)

    def main():
        x1 = torch.ones(M, K, dtype=torch.float16)
        x = [x1.npu()]
        weight1 = torch.ones(g, K, N, dtype=torch.float16)
        weight = [weight1.npu()]
        group_list = torch.Tensor([128, 256]).to(torch.int64).npu()
        split_item = 3
        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        custom_output = model(x, weight, group_list, split_item)
        print(custom_output[0].cpu())

    if __name__ == '__main__':
        main()
    ```

  <!-- npu="950" id104 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化场景示例（约束见[静态量化（T-T、T-C）场景约束](#静态量化t-tt-c场景约束)）<a id="图模式全量化示例"></a>

      ```python
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 4
      N = 5
      K = 4
      g = 2
      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, group_list, split_item, pertoken):
              return torch_npu.npu_grouped_matmul(x, weight, scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)

      def main():
          x1 = torch.ones(M, K, dtype=torch.int8)
          x = [x1.npu()]
          weight1 = torch.ones(g, K, N, dtype=torch.int8)
          weight = [weight1.npu()]
          scale1 = torch.ones(g, N, dtype=torch.bfloat16).npu()
          scale = [scale1]
          pertoken1 = torch.ones(M, dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id104 -->
  <!-- npu="950" id105 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化weightNz场景示例（x/weight为`torch.int8`）（约束见[静态量化（T-T、T-C）场景约束](#静态量化t-tt-c场景约束)）<a id="图模式全量化weightnz示例"></a>

      ```python
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os
      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 4
      N = 5
      K = 4
      g = 2

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, group_list, split_item, pertoken):
              return torch_npu.npu_grouped_matmul(x, [weight.transpose(1,2)], scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)
      def main():
          x1 = torch.ones(M, K, dtype=torch.int8)
          x = [x1.npu()]

          weight3 = torch.ones(g, K, N, dtype=torch.int8).npu().transpose(1,2)
          weight = torch_npu.npu_format_cast(weight3, 29) # 将weight转为nz格式

          scale1 = torch.ones(g, N, dtype=torch.bfloat16).npu()
          scale = [scale1]
          pertoken1 = torch.ones(M, dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([2, 2]).to(torch.int64).npu()
          split_item = 3
          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id105 -->
  <!-- npu="950" id106 -->
  - <term>Ascend 950PR&950DT系列产品</term>：MX伪量化场景示例（x为`torch.float16`，weight为`torch.float32`（`torch.float4_e2m1fn_x2`），antiquant\_scale使用`torch.uint8`承载`torch.float8_e8m0fnu`）（约束见[伪量化（MX）场景约束](#伪量化mx场景约束)）<a id="图模式MX伪量化示例"></a>

      ```python
      import numpy as np
      from ml_dtypes import float4_e2m1fn
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      m = 1
      k = 64
      n = 64
      g = 2
      E2M1_MIN, E2M1_MAX = -6, 6
      antiquant_group_size = 32
      trans_weight = False

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, antiquant_scale, group_list, split_item):
              return torch_npu.npu_grouped_matmul(x, weight, antiquant_scale=antiquant_scale,
                                                  group_list=group_list,
                                                  split_item=split_item,
                                                  group_type=0, output_dtype=torch.float16, group_list_type=1)

      def main():
          x1 = torch.ones(m, k, dtype=torch.float16)
          x = [x1.npu()]
          weight1 = (E2M1_MIN + (E2M1_MAX - E2M1_MIN) * np.random.random(g * k * n).reshape((g, k, n))).astype(float4_e2m1fn)
          weight1 = torch.from_numpy(weight1.astype(np.float32))
          npu_weight = torch_npu.npu_format_cast(weight1.npu(), 29, customize_dtype=torch.float16)
          weight_packed = torch_npu.npu_convert_weight_to_int4pack(npu_weight)
          weight = [weight_packed.npu()]
          antiquant_scale1 = torch.randint(127, 128, (g, k // antiquant_group_size, n), dtype=torch.uint8)
          print(antiquant_scale1)
          antiquant_scale = [antiquant_scale1.npu()]
          group_list = torch.Tensor([0, 1]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=True)
          custom_output = model(x, weight, antiquant_scale, group_list, split_item)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id106 -->
  <!-- npu="950" id107 -->
  - <term>Ascend 950PR&950DT系列产品</term>：G-B动态量化场景示例（约束见[动态量化（G-B、MX）场景约束](#动态量化g-bmx场景约束)）<a id="图模式g-b动态量化示例"></a>

      ```python
      import math
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"
      M = 576
      N = 7168
      K = 512
      g = 4

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight1, scale1, group_list, split_item, pertoken):
              weight1 = weight1.transpose(1,2)
              scale1 = scale1.transpose(1,2)
              scale = [scale1]
              weight = [weight1]
              return torch_npu.npu_grouped_matmul(x, weight, scale=scale, group_list=group_list, split_item=split_item, group_type=0, output_dtype=torch.bfloat16, group_list_type=0, per_token_scale=pertoken)

      def main():
          x1 = torch.ones(M, K, dtype=torch.int8).to(torch.float8_e4m3fn)
          x = [x1.npu()]
          weight1 = torch.ones(g, N, K, dtype=torch.int8).to(torch.float8_e4m3fn).npu()
          scale1 = torch.randint(-1, 1, (g, math.ceil(N/128), math.ceil(K/128)), dtype=torch.float32).npu()
          pertoken1 = torch.randint(-1, 1, (M, math.ceil(K/128)), dtype=torch.float32).npu()
          pertoken = [pertoken1]
          group_list = torch.Tensor([8, 181, 415, 576]).to(torch.int64).npu()
          split_item = 3

          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight1, scale1, group_list, split_item, pertoken)
          print(custom_output[0].cpu())

      if __name__ == '__main__':
          main()
      ```
  <!-- end id107 -->
  <!-- npu="950" id108 -->
  - <term>Ascend 950PR&950DT系列产品</term>：MxA4W4全量化场景示例（约束见[动态量化（G-B、MX）场景约束](#动态量化g-bmx场景约束)）<a id="图模式MX-a4w4全量化示例"></a>

      ```python
      import math
      import torch
      import torch.nn as nn
      import torch_npu
      import torchair as tng
      from torchair.configs.compiler_config import CompilerConfig
      import os

      config = CompilerConfig()
      npu_backend = tng.get_npu_backend(compiler_config=config)
      #os.environ["ENABLE_ACLNN"] = "true"

      class GMMModel(nn.Module):
          def __init__(self):
              super().__init__()
          def forward(self, x, weight, scale, per_token_scale, group_list, split_item):
              return torch_npu.npu_grouped_matmul([x], [weight], scale=[scale], per_token_scale = [per_token_scale], group_list=group_list,
        split_item=split_item, group_type=0, output_dtype=torch.bfloat16, x_dtype=torch.float4_e2m1fn_x2, weight_dtype=torch.float4_e2m1fn_x2,
        scale_dtype=torch_npu.float8_e8m0fnu, per_token_scale_dtype=torch_npu.float8_e8m0fnu, group_list_type=0)

      def main():
          M = 64
          N = 128
          K = 128
          g = 1
          x = torch.randint(1, 3, (M, int(K/2)), dtype=torch.uint8).npu()
          weight = torch.randint(1, 3, (g, K, int(N/2)), dtype=torch.uint8).npu()
          scale = torch.randint(-1, 1, (g, math.ceil(K/64),N,2), dtype=torch.int8).npu()
          per_token_scale = torch.randint(-1, 1, (M,math.ceil(K/64),2), dtype=torch.int8).npu()
          group_list = torch.Tensor([1]).to(torch.int64).npu()
          split_item = 2
          model = GMMModel().npu()
          model = torch.compile(model, backend=npu_backend, dynamic=False)
          custom_output = model(x, weight, scale, per_token_scale, group_list, split_item)

      if __name__ == '__main__':
          main()
      ```
  <!-- end id108 -->
