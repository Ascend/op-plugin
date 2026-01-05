# torch_npu.npu_grouped_matmul_swiglu_quant_v2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    | √  |

## 功能说明

`npu_grouped_matmul_swiglu_quant_v2`是一种融合分组矩阵乘法（GroupedMatmul）、SwiGLU混合激活函数、量化（quant）的计算方法。该方法适用于需要对矩阵乘法结果进行SwiGLU激活函数激活的场景，融合算子在底层能够对部分过程并行，达到性能优化的效果。

## 函数原型

```
torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, weight, weight_scale, x_scale, group_list, *, smooth_scale=None, weight_assist_matrix=None, bias=None, dequant_mode=0, dequant_dtype=0, quant_mode=0, quant_dtype=0, group_list_type=0, tuning_config=None) -> (Tensor, Tensor)
```

## 参数说明

- **x**（`Tensor`）：必选输入，矩阵乘法的左矩阵。shape支持2维[m,k]，数据类型支持`int8`，数据格式支持ND，支持非连续的Tensor。
- **weight**（`TensorList`）：必选输入，权重矩阵（矩阵乘法右矩阵），shape支持3维[e,k,n]，数据类型支持`int8`、`int32`，数据格式支持FRACTAL_NZ(通过接口npu_format_cast，可实现格式转换)，支持非连续的Tensor。
- **weight\_scale**（`TensorList`）：必选输入，右矩阵的量化因子。`weight`数据类型为`int8`时，`weight_scale`的shape支持2维，`weight`数据类型为`int32`时，`weight_scale`的shape支持2维和3维。数据类型支持`float32`，数据格式支持ND，支持非连续的Tensor。
- **x\_scale**（`Tensor`）：必选输入，左矩阵的量化因子。shape支持1维[m]，数据类型支持`float32`，数据格式支持ND，支持非连续的Tensor。
- **group\_list**（`Tensor`）：必选输入，指示每个分组参与计算的Token个数。shape支持1维[e]，数据类型支持`int64`，数据格式支持ND，支持非连续的Tensor。
- **smooth\_scale**（`Tensor`）：可选输入，量化的smooth_scales。数据类型为`float32`，当前仅支持传入默认值None。
- **weight\_assist\_matrix**（`TensorList`）：可选输入，右矩阵的辅助矩阵，数据类型支持`float32`，当前仅支持传入默认值None。
- **bias**（`Tensor`）：可选输入，矩阵乘计算的偏移值，公式中的bias，shape支持2维，数据类型支持`int32`，当前仅支持传入默认值None。
- **dequant\_mode**（`int`）：可选输入，表示反量化模式，数据类型为`int32`。`weight`数据类型为`int8`时仅支持传入默认值0，`weight`数据类型为`int32`时支持传入0和1。
    - 取值为0时，表示左pertoken，右perchannel。
    - 取值为1时，表示左pertoken，右pergroup。
- **dequant\_dtype**（`int`）：可选输入，表示反量化类型，数据类型为`int32`，预留输入，当前仅支持传入默认值0。
    - 取值为0时，表示pertoken。
    - 取值为1时，表示pergroup。
- **quant\_dtype**（`int`）：可选输入，参数表示量化后低比特数据类型。0：`int8`；1：`float8_e8m0`；2：`float8_e5m2`；3：`float8_e4m3`，数据类型为`int32`，当前仅支持传入默认值0。
- **quant\_mode**（`int`）：可选输入，参数表示SwiGLU后的量化模式。数据类型为`int32`，当前仅支持传入默认值0。
    - 取值为0时，表示pertoken。
    - 取值为1时，表示perchannel。
- **group\_list\_type**（`int`）：可选输入，参数表示grouplist的输入类型。取值为0时，表示cumsum；取值为1时，表示count。数据类型为`int32`，当前仅支持传入默认值0。
- **tuning\_config**（`List[int]`）：可选输入，参数数组中的第一个元素表示各个专家处理的token数的预期值。从第二个元素开始预留，用户无须填写，未来会进行扩展。默认设置为None。

## 返回值说明

- **output**（`Tensor`）：输出的量化结果，数据类型支持`int8`，shape支持2维[m,n]。数据格式支持ND，支持非连续的Tensor。
- **output_scale**（`Tensor`）：输出的量化因子，数据类型支持`float`，shape支持1维[m]。数据格式支持ND，支持非连续的Tensor。

## 约束说明

-   该接口支持推理和训练场景下使用。
-   该接口支持图模式。
-   输入和输出Tensor支持的数据类型组合如下：
    |x|weight|group_list|weight_scale|x_scale|bias|weight_assist_matrix|smooth_scale|y|y_scale|
    |--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
    |`int8`|`int8`|`int64`|`float32`|`float`|`int32`|`float32`|`float32`|`int8`|`float`|

## 调用示例

-   单算子模式调用

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

-   图模式调用：

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

