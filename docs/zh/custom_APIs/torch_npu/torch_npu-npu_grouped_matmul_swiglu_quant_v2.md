# torch_npu.npu_grouped_matmul_swiglu_quant_v2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |

## 功能说明

`npu_grouped_matmul_swiglu_quant_v2`是一种融合分组矩阵乘法（GroupedMatmul）、反量化（dequant）、SwiGLU混合激活函数、量化（quant）的计算方法。该方法适用于需要对矩阵乘法结果进行SwiGLU激活函数激活的场景，融合算子在底层能够对部分过程并行，达到性能优化的效果。支持以下量化场景：

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
    * $w\_scale$：分组权重矩阵的缩放因子，per-channel时shape为$\mathbb{R}^{E \times N}$，per-group时shape为$\mathbb{R}^{E \times K\_group\_num \times N}$。
    * $x\_scale∈\mathbb{R}^{M}$：激活矩阵的逐token缩放因子。
    * $groupList∈\mathbb{N}^{E}$：cumsum或count的分组索引列表。
  - **输出**：
    * $Q∈\mathbb{Z_8}^{M \times N / 2}$：量化后的输出矩阵。
    * $Q\_scale∈\mathbb{R}^{M}$：量化缩放因子。
  - **计算过程**：
    1. 根据groupList\[i\]确定当前分组的token，分组逻辑与A8W8相同。
    2. 将左矩阵int8拆为高低4bit两部分：

       $X\_high\_4bits_{i} = \lfloor \frac{X_{i}}{16} \rfloor$，$X\_low\_4bits_{i} = X_{i}\ \&\ 0x0f - 8$

    3. 分别与权重做矩阵乘并应用per-channel或per-group量化缩放，合并高低位结果：

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

## 函数原型

```python
torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None, weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0, group_list_type=0, tuning_config=None) -> (Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选输入，矩阵乘法的左矩阵，对应公式中的$X$。shape支持2维[m,k]，数据类型支持`int8`、`int32`，数据格式支持$ND$，支持非连续的Tensor。
- **weight**（`TensorList`）：必选输入，权重矩阵(矩阵乘法右矩阵)，对应公式中的$W$。目前仅支持TensorList长度为1。shape支持3维[e,k,n]（$ND$格式）或5维NZ格式，数据类型支持`int8`，数据格式支持$ND$和FRACTAL_NZ(通过接口npu\_format\_cast，可实现格式转换)，支持非连续的Tensor。A8W8场景下，weight仅支持NZ格式（FRACTAL\_NZ），不支持$ND$数据格式。`int32`为A8W4和A4W4场景下的适配用途，实际1个`int32`会被解释为8个int4数据。
- **weight\_scale**（`TensorList`）：必选输入，右矩阵的量化因子，对应公式中的$w_scale$。目前仅支持TensorList长度为1。`weight`数据类型为`int8`时，`weight_scale`的shape支持2维；`weight`数据类型为`int32`时，`weight_scale`的shape支持2维和3维。数据类型支持`float32`、`float16`、`bfloat16`。数据格式支持$ND$，支持非连续的Tensor。
- **x\_scale**（`Tensor`）：必选输入，左矩阵的量化因子，对应公式中的$x_scale$。shape支持1维[m]，数据类型支持`float32`，数据格式支持$ND$，支持非连续的Tensor。
- **group\_list**（`Tensor`）：必选输入，指示每个分组参与计算的Token个数，对应公式中的$groupList$。shape支持1维[e]，长度需与`weight`的首轴维度相等。数据类型支持`int64`，数据格式支持$ND$，支持非连续的Tensor。
- **smooth\_scale**（`Tensor`）：可选输入，平滑缩放因子，对应公式中的$smoothScale$。数据类型为`float32`，数据格式支持$ND$。仅A4W4场景下需传入，首轴长度需与`weight`的首轴维度相等，支持两种shape：(E, N/2)或(E,)，当使用(E,)时会进行广播乘法。其他场景传入默认值None。
- **weight\_assist\_matrix**（`TensorList`）：可选输入，右矩阵的辅助矩阵，对应公式中的$weightAssistMatrix$。数据类型支持`float32`，数据格式支持$ND$，shape支持2维。仅A8W4场景下需传入，首轴长度需与`weight`的首轴维度相等，尾轴长度需要与`weight`还原为ND格式的尾轴相同。其他场景传入默认值None。
- **bias**（`Tensor`）：可选输入，矩阵乘计算的偏移值，对应公式中的$bias$，shape支持2维，数据类型支持`int32`，当前仅支持传入默认值None。
- **dequant\_mode**（`int`）：可选输入，表示反量化模式，数据类型为`int32`，默认值为0。A8W4场景下，dequant_mode支持取值0和1；A8W8和A4W4场景下，dequant_mode仅支持取值0。
    - 取值为0时，表示激活矩阵per-token，权重矩阵per-channel。
    - 取值为1时，表示激活矩阵per-token，权重矩阵per-group。
- **dequant\_dtype**（`int`）：可选输入，表示反量化类型，数据类型为`int32`，当前仅支持传入默认值0（表示`float32`）。
- **quant\_mode**（`int`）：可选输入，参数表示SwiGLU后的量化模式。数据类型为`int32`，当前仅支持传入默认值0（表示per-token）。
- **quant\_dtype**（`int`）：可选输入，参数表示量化后低比特数据类型。数据类型为`int32`，当前仅支持传入默认值0（表示`int8`）。
- **group\_list\_type**（`int`）：可选输入，参数表示`group_list`的输入类型，数据类型为`int32`，默认值为0。
    - 取值为0时，表示cumsum模式，`group_list`中的每个元素代表当前分组的累计长度。
    - 取值为1时，表示count模式，`group_list`中的每个元素代表该分组包含多少元素。
- **tuning\_config**（`List[int]`）：可选输入，参数数组中的第一个元素表示各个专家处理的token数的预期值。从第二个元素开始预留，用户无须填写，未来会进行扩展。默认设置为None。

## 返回值说明

- **output**（`Tensor`）：输出的量化结果，对应公式中的$Q$，数据类型支持`int8`，shape支持2维[m, n/2]。数据格式支持$ND$，支持非连续的Tensor。
- **output\_scale**（`Tensor`）：输出的量化因子，对应公式中的$Q_scale$，数据类型支持`float32`，shape支持1维[m]。数据格式支持$ND$，支持非连续的Tensor。

## 约束说明

- 该接口支持推理和训练场景下使用。
- 该接口支持图模式。
- 确定性计算：该接口默认为确定性实现，即对于相同的输入，多次执行会产生相同的结果，确保计算结果的可重复性。
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
        |A8W4|(M, K)|$ND$格式{(E, K, N)}或NZ格式|per-channel:{(E, N)}; per-group:{(E, K\_group\_num, N)}|(M,)|-|(M, N/2)|(M,)|
        |A4W4|(M, K)|$ND$格式{(E, K, N)}或NZ格式|{(E, N)}|(M,)|(E, N/2)或(E,)|(M, N/2)|(M,)|

    - A8W8场景下，不支持N轴长度超过10240，不支持`x`的尾轴长度大于等于65536。
    - A8W4场景下，不支持N轴长度超过10240，不支持`x`的尾轴长度大于等于20000。
    - A4W4场景下，不支持N轴长度超过10240，不支持`x`的尾轴长度大于等于20000。

## 调用示例

- 单算子模式调用

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

- 图模式调用：

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
