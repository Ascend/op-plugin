# torch_npu.npu_grouped_matmul_swiglu_quant_v2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas 350 加速卡</term>            |    √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |

## 功能说明

- 接口功能：`npu_grouped_matmul_swiglu_quant_v2`是一种融合分组矩阵乘法（GroupedMatmul）、反量化（dequant）、SwiGLU混合激活函数、量化（quant）的计算方法。该方法适用于需要对矩阵乘法结果进行SwiGLU激活函数激活的场景，融合算子在底层能够对部分过程并行，达到性能优化的效果。支持以下量化场景：

- 计算公式：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    <details>
    <summary>量化场景A8W8（A指激活矩阵，W指权重矩阵，8指int8数据类型）：</summary>

      - **输入**：
        * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_8}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵的逐通道缩放因子。
        * $x\_scale∈\mathbb{R}^{M}$：激活矩阵的逐token缩放因子。
        * $groupList∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：
        * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
      - **计算过程**：
        1. 根据groupList\[i\]确定当前分组的token，$i \in [0,Len(groupList)]$。
        2. 根据分组确定的入参进行如下计算：

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$
        3. 量化输出结果：

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

    </details>

    <details>
    <summary>MSD场景A8W4（A指激活矩阵，W指权重矩阵，8指int8数据类型，4指int4数据类型）：</summary>

      - **输入**：
        * $X∈\mathbb{Z_8}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $weightAssistMatrix∈\mathbb{R}^{E \times N}$：计算矩阵乘时的辅助矩阵（离线生成，非算子内部完成）。
        * $w\_scale$：分组权重矩阵的缩放因子，perchannel时shape为$\mathbb{R}^{E \times N}$，pergroup时shape为$\mathbb{R}^{E \times K\_group\_num \times N}$。
        * $x\_scale∈\mathbb{R}^{M}$：激活矩阵的逐token缩放因子。
        * $groupList∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：
        * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
      - **计算过程**：
        1. 根据groupList\[i\]确定当前分组的token，分组逻辑与A8W8相同。
        2. 将左矩阵int8拆为高低4bit两部分：

          $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$，$X\_low\_4bits_{i} = X_{i}\ \&\ 0x0f - 8$
        3. 分别与权重做矩阵乘并应用perchannel或pergroup量化缩放，合并高低位结果：

          $C_{i} = (C\_high_{i} * 16 + C\_low_{i} + weightAssistMatrix_{i}) \odot x\_scale_{i}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$
        4. 量化输出结果：

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

    </details>

    <details>
    <summary>量化场景A4W4（A指激活矩阵，W指权重矩阵，4指int4数据类型）：</summary>

      - **输入**：
        * $X∈\mathbb{Z_4}^{M \times K}$：激活矩阵（左矩阵），M是总token数，K是特征维度。
        * $W∈\mathbb{Z_4}^{E \times K \times N}$：分组权重矩阵（右矩阵），E是专家个数，K是特征维度，N是输出维度。
        * $w\_scale∈\mathbb{R}^{E \times N}$：分组权重矩阵的逐通道缩放因子。
        * $x\_scale∈\mathbb{R}^{M}$：激活矩阵的逐token缩放因子。
        * $smoothScale∈\mathbb{R}^{E \times N/2}$：平滑缩放因子，E是专家个数，N是输出维度。支持shape为(E,)时广播。
        * $groupList∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
      - **输出**：
        * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
        * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
      - **计算过程**：
        1. 根据groupList\[i\]确定当前分组的token，分组逻辑与A8W8相同。
        2. 根据分组确定的入参进行如下计算：

          $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

          $C_{i,act}, gate_{i} = split(C_{i})$

          $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$

          $S_{i} = S_{i} \odot smoothScale_{i\ Broadcast}$

          注：当smoothScale形状为(E,)时，会对其进行广播，使其与$S_{i}$的形状匹配。
        3. 量化输出结果：

          $Q\_scale_{i} = \frac{max(|S_{i}|)}{127}$

          $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

    </details>
  
  - <term>Atlas 350 加速卡</term>：  
    <details>
    <summary>MX量化场景：</summary>

      1. 根据分组确定的入参进行如下计算：

         $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i\ Broadcast} \odot w\_scale_{i\ Broadcast}$

         $C_{i,act}, gate_{i} = split(C_{i})$

         $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$

      2. 量化输出结果：

         $shared\_exp = \left\lfloor \log_2(max_i(|S_i|)) \right\rceil - emax$

         $Q\_scale = 2 ^ {shared\_exp}$

         $Q_i = quantize\_to\_element\_format(S_i/Q\_scale), \space i\space from\space 1\space to\space blocksize$

         其中，$emax$表示对应数据类型的最大正则数的指数位：

         |   DataType    | emax |
         | :-----------: | :--: |
         | FLOAT8_E4M3FN |  8   |
         |  FLOAT8_E5M2  |  15  |
         |  FLOAT4_E2M1  |  2   |

         其中，$blocksize$表示每次量化的元素个数，仅支持32。

    </details>

    <details>
    <summary>Pertoken量化场景：</summary>

      1. 根据分组确定的入参进行如下计算：

         $C_{i} = (X_{i}\cdot W_{i} )\odot x\_scale_{i} \odot w\_scale_{i}$

         $C_{i,act}, gate_{i} = split(C_{i})$

         $S_{i}=Swish(C_{i,act})\odot gate_{i}$，其中$Swish(x)=\frac{x}{1+e^{-x}}$

         其中，$x\_scale_{i}$表示对应token的量化因子。

      2. 量化输出结果：

         $Q\_scale_{i} = \frac{max(|S_{i}|)}{max(type)}$

         $Q_{i} = \left\lfloor \frac{S_{i}}{Q\_scale_{i}} \right\rceil$

    </details>

## 函数原型

```python
torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None, weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0, group_list_type=0, tuning_config=None) -> (Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选输入，矩阵乘法的左矩阵，对应公式中的$X$。shape支持2维[m,k]，数据格式支持$ND$，支持非连续的Tensor。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`int32`。
  - <term>Atlas 350 加速卡</term>：数据类型支持torch.float8\_e5m2、torch.float8\_e4m3fn、torch\_npu.float4\_e2m1fn\_x2、torch.int8、torch\_npu.hifloat8，其中torch\_npu.hifloat8和float4系列需配置可选参数x\_dtype为对应类型，此时x本身的dtype不再生效，但仍需保证x本身的dtype为8bit位的数据类型，以保证shape正确；其中float4内轴K需为偶数，以保证8bits可以转换为2个float4。数据格式支持ND。

- **weight**（`TensorList`）：必选输入，权重矩阵（矩阵乘法右矩阵），对应公式中的$W$。目前仅支持TensorList长度为1。shape支持3维[e,k,n]（$ND$格式）或5维NZ格式，数据格式支持$ND$和FRACTAL_NZ（通过接口npu\_format\_cast，可实现格式转换），支持非连续的Tensor。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 数据类型支持`int8`。`int32`为A8W4和A4W4场景下的适配用途，实际1个`int32`会被解释为8个int4数据。
    - A8W8场景下，weight仅支持NZ格式（FRACTAL\_NZ），不支持$ND$数据格式。
  - <term>Atlas 350 加速卡</term>：
    - 数据格式为ND时，shape支持3维，非转置shape\[\[e, k, n\]\]，转置shape\[\[e, n, k\]\]。数据类型支持torch.float8\_e5m2、torch.float8\_e4m3fn、torch\_npu.float4\_e2m1fn\_x2、torch.int8、torch\_npu.hifloat8，其中torch\_npu.hifloat8和float4系列需配置可选参数weight\_dtype为对应类型，此时weight本身的dtype不再生效，但仍需保证weight本身的dtype为8bit位的数据类型，以保证shape正确；其中float4内轴需为偶数，以保证8bits可以转换为2个float4。
    - 数据格式为FRACTAL\_NZ\(通过接口npu\_format\_cast，可实现格式转换\)时，shape支持5维，非转置shape\[e, n/32, k/16, 16, 32\], 转置shape\[e, k/32, n/16, 16, 32\]；数据类型仅支持torch.float8\_e4m3fn。
  
- **weight\_scale**（`TensorList`）：必选输入，右矩阵的量化因子，对应公式中的$w_{scale}$。目前仅支持TensorList长度为1。数据格式支持$ND$，支持非连续的Tensor。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：`weight`数据类型为`int8`时，`weight_scale`的shape支持2维；`weight`数据类型为`int32`时，`weight_scale`的shape支持2维和3维。数据类型支持`float32`、`float16`、`bfloat16`、`uint64`。
  - <term>Atlas 350 加速卡</term>：MX量化场景下：shape支持4维，非转置shape\[\[e, ceil\(k / 64\), n, 2\]\]，转置shape\[\[e, n, ceil\(k / 64\), 2\]\]，数据类型支持torch\_npu.float8\_e8m0fnu。pertoken量化场景下：shape支持2维，shape\[\[e, n\]\]，当x为torch.int8时，weightScale需支持torch.bfloat16、torch.float32、torch.float16，当x为torch.float8\_e4m3fn/torch.float8\_e5m2/torch\_npu.hifloat8时，weightScale支持torch.bfloat16、torch.float32。

- **x\_scale**（`Tensor`）：必选输入，左矩阵的量化因子，对应公式中的$x_scale$。数据格式支持$ND$，支持非连续的Tensor。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：shape支持1维[m]，数据类型支持`float32`。
  - <term>Atlas 350 加速卡</term>：MX量化场景下：shape支持3维\[m, ceil\(k / 64\), 2\]，数据类型支持torch\_npu.float8\_e8m0fnu。pertoken量化场景下：shape支持1维\[m\]，数据类型支持torch.float32。

- **group\_list**（`Tensor`）：必选输入，指示每个分组参与计算的Token个数，对应公式中的$groupList$。shape支持1维[e]，长度需与`weight`的首轴维度相等。数据类型支持`int64`，数据格式支持$ND$，支持非连续的Tensor。
- **smooth\_scale**（`Tensor`）：可选输入，平滑缩放因子，对应公式中的$smoothScale$。数据类型为`float32`，数据格式支持$ND$。仅A4W4场景下需传入，首轴长度需与`weight`的首轴维度相等，支持两种shape：(E, N/2)或(E,)，当使用(E,)时会进行广播乘法。其他场景传入默认值None。
- **weight\_assist\_matrix**（`TensorList`）：可选输入，右矩阵的辅助矩阵，对应公式中的$weightAssistMatrix$。数据类型支持`float32`，数据格式支持$ND$，shape支持2维。仅A8W4场景下需传入，首轴长度需与`weight`的首轴维度相等，尾轴长度需要与`weight`还原为ND格式的尾轴相同。其他场景传入默认值None。
- **bias**（`Tensor`）：可选输入，矩阵乘计算的偏移值，对应公式中的$bias$，shape支持2维，数据类型支持`int32`，当前仅支持传入默认值None。
- **dequant\_mode**（`int`）：可选输入，表示反量化模式，数据类型为`int32`，默认值为0。取值为0时，表示激活矩阵pertoken，权重矩阵perchannel。取值为1时，表示激活矩阵pertoken，权重矩阵pergroup。取值为2时，表示mx量化。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：A8W4场景下，dequant_mode支持取值0和1；A8W8和A4W4场景下，dequant_mode仅支持取值0。
  - <term>Atlas 350 加速卡</term>：当前仅支持传入0以及2。

- **dequant\_dtype**（`int`）：可选输入，表示反量化类型，数据类型为`int32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当前仅支持传入默认值0（表示`float32`）。
  - <term>Atlas 350 加速卡</term>：默认值为torch.int8，当前仅支持传入torch.float32、torch.bfloat16、torch.float16。

- **quant\_mode**（`int`）：可选输入，参数表示SwiGLU后的量化模式。数据类型为`int32`。支持取值：0（默认）表示pertoken量化；1表示pergroup量化；2表示mx量化。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当前仅支持传入默认值0（表示pertoken）。
  - <term>Atlas 350 加速卡</term>：当前仅支持传入0以及2。

- **quant\_dtype**（`int`）：可选输入，参数表示量化后低比特数据类型。数据类型为`int32`。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当前仅支持传入默认值0（表示`int8`）。
  - <term>Atlas 350 加速卡</term>：默认值为torch.int8，当前支持传入torch.float8\_e5m2、torch.float8\_e4m3fn、torch\_npu.float4\_e2m1fn\_x2、torch.int8、torch\_npu.hifloat8。

- **group\_list\_type**（`int`）：可选输入，参数表示`group_list`的输入类型，数据类型为`int32`，默认值为0。
    - 取值为0时，表示cumsum模式，`group_list`中的每个元素代表当前分组的累计长度。
    - 取值为1时，表示count模式，`group_list`中的每个元素代表该分组包含多少元素。
- **tuning\_config**（`List[int]`）：可选输入，参数数组中的第一个元素表示各个专家处理的token数的预期值。从第二个元素开始预留，用户无须填写，未来会进行扩展。默认设置为None。

- **x\_dtype**（int）：可选参数，指定输入x的真实数据类型。当前仅支持默认值None，表示输入x真实的数据类型与输入x的dtype相同。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：暂不支持该参数，使用默认值。
  - <term>Atlas 350 加速卡</term>：当x为float4\_e2m1fn\_x2、hifloat8时，x\_dtype需要传入torch\_npu.float4\_e2m1fn\_x2、torch\_npu.hifloat8。

- **weight\_dtype**（int）：可选参数，指定输入weight的真实数据类型。当前仅支持默认值None，表示输入weight真实的数据类型与输入weight的dtype相同。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：暂不支持该参数，使用默认值。
  - <term>Atlas 350 加速卡</term>：当weight为float4\_e2m1fn\_x2、hifloat8时，weight\_dtype需要传入torch\_npu.float4\_e2m1fn\_x2、torch\_npu.hifloat8。

- **weight\_scale\_dtype**（int）：可选参数，指定输入weight\_scale的真实数据类型。默认值None，表示输入weight\_scale真实的数据类型与输入weight\_scale的dtype相同。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：暂不支持该参数，使用默认值。
  - <term>Atlas 350 加速卡</term>：当weight\_scale为float8\_e8m0fnu时，weight\_scale\_dtype需要传入torch\_npu.float8\_e8m0fnu。

- **x\_scale\_dtype**（int）：可选参数，指定输入x\_scale的真实数据类型。默认值None，表示输入x\_scale真实的数据类型与输入x\_scale的dtype相同。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：暂不支持该参数，使用默认值。
  - <term>Atlas 350 加速卡</term>：当x\_scale为float8\_e8m0fnu时，x\_scale\_dtype需要传入torch\_npu.float8\_e8m0fnu。

## 返回值说明

- **output**（`Tensor`）：输出的量化结果，对应公式中的$Q$。数据格式支持$ND$，支持非连续的Tensor。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`，shape支持2维[m, n/2]。
  - <term>Atlas 350 加速卡</term>：数据类型支持torch.float8\_e4m3fn、torch.float8\_e5m2、torch\_npu.float4\_e2m1fn\_x2、torch.int8、torch\_npu.hifloat8，shape支持2维\[m，n / 2\]。

- **output\_scale**（`Tensor`）：输出的量化因子，对应公式中的$Q_{scale}$。数据格式支持$ND$，支持非连续的Tensor。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`，shape支持1维[m]。
  - <term>Atlas 350 加速卡</term>：MX量化场景：数据类型支持torch\_npu.float8\_e8m0fnu，shape支持3维\[m, ceil\(\(n / 2\) / 64\), 2\]。pertoken量化场景：shape支持1维\[m\]，数据类型支持torch.float32。

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口支持图模式。
- 确定性计算：该接口默认为确定性实现，即对于相同的输入，多次执行会产生相同的结果，确保计算结果的可重复性。
- MX量化场景下，需满足n为128对齐。
- MXFP4场景不支持k=2，MXFP4场景需满足K为偶数。
- group\_list第1维最大支持1024，即最多支持1024个group。
- 参数说明里Shape使用的变量说明：
  - e：表示分组数目，取值范围为1-1024。
  - m：输出矩阵output的倒数第二维大小，取值范围为1-2147483647。
  - n：输出矩阵output的倒数第一维大小的两倍，取值范围为1-2147483647。<term>Atlas 350 加速卡</term> mx量化场景下要求n为128对齐。
  - k：矩阵乘法reduce轴的大小，取值范围为1-2147483647。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 支持A8W8、A8W4、A4W4量化场景，输入和输出Tensor支持的数据类型组合如下：

        |量化场景|x|weight|weight\_scale|x\_scale|smooth\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|--------|--------|
        |A8W8|`int8`|`int8`|`float32`、`float16`、`bfloat16`|`float32`|-|`int8`|`float32`|
        |A8W4|`int8`|`int4`、`int32`|`uint64`|`float32`|-|`int8`|`float32`|
        |A4W4|`int4`、`int32`|`int4`、`int32`|`float32`|`float32`|`float32`|`int8`|`float32`|

    - shape约束如下：

        |量化场景|x|weight|weight\_scale|x\_scale|smooth\_scale|output|output\_scale|
        |--------|--------|--------|--------|--------|--------|--------|--------|
        |A8W8|(M, K)|NZ格式shape形如{(E, N/32, K/16, 16, 32)}|{(E, N)}|(M,)|-|(M, N/2)|(M,)|
        |A8W4|(M, K)|$ND$格式{(E, K, N)}或NZ格式|perchannel:{(E, N)}; pergroup:{(E, K\_group\_num, N)}|(M,)|-|(M, N/2)|(M,)|
        |A4W4|(M, K)|$ND$格式{(E, K, N)}或NZ格式|{(E, N)}|(M,)|(E, N/2)或(E,)|(M, N/2)|(M,)|

    - A8W8场景下，不支持N轴长度超过10240，不支持`x`的尾轴长度大于等于65536。
    - A8W4场景下，不支持N轴长度超过10240，不支持`x`的尾轴长度大于等于20000。
    - A4W4场景下，不支持N轴长度超过10240，不支持`x`的尾轴长度大于等于20000。
- <term>Atlas 350 加速卡</term>：
    - 输入和输出Tensor支持的数据类型组合如下：

      - MX量化场景：

        | 量化模式 | x | weight | group_list | weight_scale | x_scale | bias | weight_assist_matrix | smooth_scale | output | output_scale |
        | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
        | MXFP8量化（ND格式） | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.int64` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` |
        | MXFP4量化（ND格式） | `torch_npu.float4_e2m1fn_x2` | `torch_npu.float4_e2m1fn_x2` | `torch.int64` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch_npu.float4_e2m1fn_x2` / `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch_npu.float8_e8m0fnu` |
        | MXFP8量化（FRACTAL_NZ格式） | `torch.float8_e4m3fn` | `torch.float8_e4m3fn` | `torch.int64` | `torch_npu.float8_e8m0fnu` | `torch_npu.float8_e8m0fnu` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float8_e4m3fn` | `torch_npu.float8_e8m0fnu` |

      - Pertoken量化场景：

        | x | weight | group_list | weight_scale | x_scale | bias | weight_assist_matrix | smooth_scale | output | output_scale |
        | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
        | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.int64` | `torch.float32` / `torch.bfloat16` | `torch.float32` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.float8_e4m3fn` / `torch.float8_e5m2` | `torch.float32` |
        | `torch.int8` | `torch.int8` | `torch.int64` | `torch.float32` / `torch.float16` / `torch.bfloat16` | `torch.float32` | 暂不支持 | 暂不支持 | 暂不支持 | `torch.int8` | `torch.float32` |
        | `torch_npu.hifloat8` | `torch_npu.hifloat8` | `torch.int64` | `torch.float32` / `torch.bfloat16` | `torch.float32` | 暂不支持 | 暂不支持 | 暂不支持 | `torch_npu.hifloat8` | `torch.float32` |

    - 输入和输出Tensor支持的shape组合如下：

      | 量化模式 | x | weight | weight_scale | x_scale | output | output_scale |
      | --- | --- | --- | --- | --- | --- | --- |
      | MX量化（ND格式） | `(m, k)` | 非转置shape形如`{(e, k, n)}`<br>转置shape形如`{(e, n, k)}` | 非转置shape形如`{(e, ceil(k / 64), n, 2)}`<br>转置shape形如`{(e, n, ceil(k / 64), 2)}` | `(m, ceil(k / 64), 2)` | `(m, n / 2)` | `(m, ceil((n / 2) / 64), 2)` |
      | MX量化（FRACTAL_NZ格式） | `(m, k)` | 非转置shape形如`{(e, n / 32, k / 16, 16, 32)}`<br>转置shape形如`{(e, k / 32, n / 16, 16, 32)}` | 非转置shape形如`{(e, ceil(k / 64), n, 2)}`<br>转置shape形如`{(e, n, ceil(k / 64), 2)}` | `(m, ceil(k / 64), 2)` | `(m, n / 2)` | `(m, ceil((n / 2) / 64), 2)` |
      | Pertoken量化 | `(m, k)` | 非转置shape形如`{(e, k, n)}`<br>转置shape形如`{(e, n, k)}` | `shape`形如`{(e, n)}` | `(m,)` | `(m, n / 2)` | `(m,)` |

## 调用示例

- 单算子模式调用：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    ```python
    import numpy as np
    import torch
    import torch_npu
    from scipy.special import softmax

    torch.npu.config.allow_internal_format = True

    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList
    E = 2
    M = 512
    K = 7168
    N = 4096
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu())
    ```
  
  - <term>Atlas 350 加速卡</term>：mx量化场景示例-mxfp8

    ```python
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch_npu
    import math

    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8).to(torch.float8_e4m3fn)

        weightScale = torch.randint(low=-128, high=127, size=(E, math.ceil(K / 64), N, 2), dtype=torch.int8)
        xScale = torch.randint(low=-128, high=127, size=(M, math.ceil(K / 64), 2), dtype=torch.int8)
        groupList = torch.tensor([int(M / 2), int(M / 2)], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    K = 2
    E = 2
    M = 16
    N = 128
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = weight.npu()
    weightScale = weightScale.npu()
    output0_npu, output1_npu = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x.npu(),
        [weight_npu],
        [weightScale],
        xScale.npu(),
        groupList.npu(),
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=torch.float8_e4m3fn,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu)
    ```

  - <term>Atlas 350 加速卡</term>：mx量化场景示例-mxfp4

    ```python
    import numpy as np
    import torch
    import torch_npu
    import math

    K = 9
    E = 2
    M = 2255
    N = 896

    x = torch.randint(0, 256, (M, K), dtype=torch.uint8).npu()
    weight = torch.randint(0, 256, (E, K * 2, N), dtype=torch.uint8).npu()
    weightScale = torch.randint(0, 256, (E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8).npu()
    xScale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8).npu()
    groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64).npu()

    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
        x,
        [weight], [weightScale],
        xScale, groupList,
        dequant_mode=2,
        dequant_dtype=torch.float32,
        quant_mode=2,
        quant_dtype=torch_npu.float4_e2m1fn_x2,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu,
        x_dtype=torch_npu.float4_e2m1fn_x2,
        weight_dtype=torch_npu.float4_e2m1fn_x2,
        group_list_type=1)

    print("y.shape: ", y.shape)
    print("y_scale.shape: ", y_scale.shape)
    ```

  - <term>Atlas 350 加速卡</term>：pertoken量化场景示例

    ```python
    import numpy as np
    import torch
    import torch_npu
    import math

    K = 9
    E = 2
    M = 2255
    N = 896
    x = torch.randint(0, 256, (M, K), dtype=torch.uint8).to(torch.float8_e5m2).npu()
    weight = torch.randint(0, 256, (E, K, N), dtype=torch.uint8).to(torch.float8_e5m2).npu()
    weightScale = torch.randint(0, 256, (E, N), dtype=torch.float).npu()
    xScale = torch.randint(0, 256, (M,), dtype=torch.float).npu()
    groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64).npu()
    y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x,
        [weight], [weightScale],
        xScale, groupList,
        dequant_mode=0,
        quant_mode=0,
        quant_dtype=torch_npu.float8_e5m2,
        dequant_dtype=torch.float,
        group_list_type=1)
    print("y.shape: ", y.shape)
    print("y_scale.shape: ", y_scale.shape)
    ```

- 图模式调用：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from scipy.special import softmax
    from torchair.configs.compiler_config import CompilerConfig
    
    torch.npu.config.allow_internal_format = True
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
     
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight, weightscale, xscale, group_list, quant_dtype):
            output = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weightscale, xscale, group_list, quant_dtype=quant_dtype)
            return output    
     
    def gen_input_data(E, M, K, N):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        weight = torch.randint(-128, 127, (E, K, N), dtype=torch.int8)
        weightScale = torch.randn(E, N)
        xScale = torch.randn(M)
        groupList = torch.tensor([128, 128], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList    
    E = 2
    M = 512
    K = 7168
    N = 4096
    quant_dtype = 2
    x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
    weight_npu = torch_npu.npu_format_cast(weight.npu(), 29)
     
    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), [weight_npu], [weightScale.npu()], xScale.npu(), groupList.npu(), quant_dtype)
    ```

  - <term>Atlas 350 加速卡</term>：mx量化场景示例-mxfp8

    ```python
    import os
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from typing import Tuple
    import logging
    import torch_npu
    from torchair import logger
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()

    npu_backend = tng.get_npu_backend(compiler_config=config)

    os.environ["ENABLE_ACLNN"] = "false"

    class GMMModel(nn.Module):
        def __init__(self, weight_npu, weightScale, xScale, transpose=True):
            super().__init__()
            self.transpose = transpose
            self.weight = nn.Parameter(weight_npu, requires_grad=False)
            self.weightScale = nn.Parameter(weightScale, requires_grad=False)
            self.xScale = nn.Parameter(xScale, requires_grad=False)

        def forward(self, x_npu：Torch.Tensor, w：Torch.Tensor, group_list_npu：Torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                weight = self.weight
                weightScale = self.weightScale.npu()
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x_npu, [weight.transpose(1, 2)], [weightScale.transpose(1, 2)], xScale.npu(), group_list_npu, quant_mode=2, quant_dtype=torch.float8_e5m2, dequant_mode=2, dequant_dtype=torch.float32,weight_scale_dtype=torch_npu.float8_e8m0fnu, x_scale_dtype=torch_npu.float8_e8m0fnu)
                return y, y_scale

    def gen_input_data(E, M, K, N, transpose):
        if transpose:
            x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
            weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn)
            weightScale = torch.randint(low=0, high=256, size=(E, N, math.ceil(K / 64), 2), dtype=torch.uint8)
            xScale = torch.randint(low=0, high=256, size=(M, math.ceil(K / 64), 2), dtype=torch.uint8)
            groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    def run_npu(x, weight_npu, weightScale, xScale, groupList, transpose):
        model = GMMModel(weight_npu, weightScale, xScale, transpose).npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)

        for k in range(1):
            torch_npu.npu.synchronize()
            custom_output, y_scale = model(x, None, groupList)
            torch_npu.npu.synchronize()

    if __name__ == "__main__":
        K = 1
        E = 2
        M = 16
        N = 128
        transpose = True
        x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N, transpose)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weightScale_npu = weightScale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        run_npu(x_npu, weight_npu, weightScale_npu, xScale_npu, groupList_npu, transpose)
    ```

  - <term>Atlas 350 加速卡</term>：mx量化场景示例-mxfp4

    ```python
    import os
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    os.environ["ENABLE_ACLNN"] = "false"

    class GMMModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,
                    x,
                    weight,
                    weight_scale,
                    x_scale,
                    group_list,
                    dequant_mode,
                    dequant_dtype,
                    quant_mode,
                    quant_dtype,
                    group_list_type,
                    weight_scale_dtype,
                    x_scale_dtype,
                    x_dtype=None,
                    weight_dtype=None,
                    transpose_w=False):
            if quant_dtype is None:
                quant_dtype = torch.float8_e5m2
            if weight_scale_dtype is None:
                weight_scale_dtype = torch_npu.float8_e8m0fnu
            if x_scale_dtype is None:
                x_scale_dtype = torch_npu.float8_e8m0fnu

            processed_weight = []
            for w in weight:
                if transpose_w:
                    w = w.transpose(1, 2)
                processed_weight.append(w)

            processed_weight_scale = []
            for ws in weight_scale:
                if transpose_w:
                    ws = ws.transpose(1, 2)
                processed_weight_scale.append(ws)

            with torch.no_grad():
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
                    x,
                    processed_weight,
                    processed_weight_scale,
                    x_scale,
                    group_list,
                    dequant_mode=dequant_mode,
                    dequant_dtype=dequant_dtype,
                    quant_mode=quant_mode,
                    quant_dtype=quant_dtype,
                    group_list_type=group_list_type,
                    weight_scale_dtype=weight_scale_dtype,
                    x_scale_dtype=x_scale_dtype,
                    x_dtype=x_dtype,
                    weight_dtype=weight_dtype
                )
                return y, y_scale

    def gen_input_data(E, M, K, N):
        x = torch.randint(0, 256, (M, K), dtype=torch.uint8)
        weight = torch.randint(0, 256, (E, K * 2, N), dtype=torch.uint8)
        weightScale = torch.randint(0, 256, (E, math.ceil(K / 64), N * 2, 2), dtype=torch.uint8)
        xScale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8)
        groupList = torch.tensor([int(M/2), int(M/2) + 1], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList

    if __name__ == "__main__":
        K = 9
        E = 2
        M = 2255
        N = 896
        transpose = False

        x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weightScale_npu = weightScale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        weight_list = [weight_npu]
        weight_scale_list = [weightScale_npu]

        model = GMMModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)

        y, y_scale = model(
            x_npu,
            weight_list,
            weight_scale_list,
            xScale_npu,
            groupList_npu,
            dequant_mode=2,
            dequant_dtype=torch.float32,
            quant_mode=2,
            quant_dtype=torch.float8_e4m3fn,
            group_list_type=1,
            weight_scale_dtype=torch_npu.float8_e8m0fnu,
            x_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch_npu.float4_e2m1fn_x2,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            transpose_w=transpose
        )

        print("y shape: ", y.shape)
        print("y_scale shape: ", y_scale.shape)
    ```

  - <term>Atlas 350 加速卡</term>：Pertoken量化场景示例

    ```python
    import os
    import unittest
    import itertools
    import numpy as np
    import torch
    import torch.nn as nn
    import torch_npu
    import math
    import torchair as tng
    from typing import Tuple
    import logging
    import torch_npu
    from torchair import logger
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    os.environ["ENABLE_ACLNN"] = "false"

    class GMMModel(nn.Module):
        def __init__(self, weight_npu, weightScale, xScale, transpose=True):
            super().__init__()
            self.transpose = transpose
            self.weight = nn.Parameter(weight_npu, requires_grad=False)
            self.weightScale = nn.Parameter(weightScale, requires_grad=False)
            self.xScale = nn.Parameter(xScale, requires_grad=False)
        def forward(self, x_npu：Torch.Tensor, w：Torch.Tensor, group_list_npu：Torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                weight = self.weight
                weightScale = self.weightScale.npu()
                y, y_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(x_npu, [weight.transpose(1, 2)], [weightScale], xScale.npu(), group_list_npu, quant_mode=0, quant_dtype=torch.float8_e5m2, dequant_mode=0, dequant_dtype=torch.float)
                return y, y_scale
    def gen_input_data(E, M, K, N, transpose):
        x = torch.randint(-128, 127, (M, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weight = torch.randint(-128, 127, (E, N, K), dtype=torch.int8).to(torch.float8_e4m3fn)
        weightScale = torch.randint(low=0, high=256, size=(E, N), dtype=torch.float)
        xScale = torch.randint(low=0, high=256, size=(M,), dtype=torch.float)
        groupList = torch.tensor([M//2, M//2], dtype=torch.int64)
        return x, weight, weightScale, xScale, groupList
    def run_npu(x, weight_npu, weightScale, xScale, groupList, transpose):
        model = GMMModel(weight_npu, weightScale, xScale, transpose).npu()
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        for k in range(1):
            torch_npu.npu.synchronize()
            customyy_output, y_scale = model(x, None, groupList)
            print(customyy_output, y_scale)
            torch_npu.npu.synchronize()
    if __name__ == "__main__":
        K = 1
        E = 2
        M = 16
        N = 128
        transpose = False
        x, weight, weightScale, xScale, groupList = gen_input_data(E, M, K, N, transpose)
        x_npu = x.npu()
        weight_npu = weight.npu()
        weightScale_npu = weightScale.npu()
        xScale_npu = xScale.npu()
        groupList_npu = groupList.npu()
        run_npu(x_npu, weight_npu, weightScale_npu, xScale_npu, groupList_npu, transpose)
    ```
