# torch_npu.npu_dynamic_quant_asymmetric

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>            |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |

## 功能说明

-   API功能：

    对输入的张量进行动态非对称量化。支持pertoken、pertensor和MoE（Mixture of Experts，混合专家模型）场景。

-   计算公式：
    
    pertoken场景，rowMax、rowMin代表按行取最大值、按行取最小值，此处的“行”对应`x`最后一个维度的数据，即一个token。DST_MAX、DST_MIN分别对应量化后dtype的最大值和最小值，公式如下：

    $$
    \text{scale} = \frac{\text{rowMax}(\mathbf{x}) - \text{rowMin}(\mathbf{x})}{DST\_MAX - DST\_MIN}\\
    \text{offset} = DST\_MAX - \frac{\text{rowMax}(\mathbf{x})}{\text{scale}}\\
    y = \text{round}(\frac{\mathbf{x}}{\text{scale}} + \text{offset})
    $$

    -   若使用smooth quant，非MoE（Mixture of Experts，混合专家模型）场景下，会引入smooth_scales输入，其形状与x最后一个维度大小一致，在进行量化前，会先令x乘以smooth_scales，再按上述公式进行量化。MoE（Mixture of Experts，混合专家模型）场景下会同时引入smooth_scales和group_index，此时smooth_scales中包含多组smooth向量，按group_index中的数值作用到x的不同行上。具体地，假如x包含m个token，smooth_scales有n行，smooth_scales[0]会作用到x[0:group_index[0]]上，smooth_scales[i]会作用到x[group_index[i-1]: group_index[i]]上，i=1,2, ...,n-1。
    -   在pertensor场景下，rowMax、rowMin表示求整个tensor的最大值、最小值。

## 函数原型

```
torch_npu.npu_dynamic_quant_asymmetric(x, *, smooth_scales=None, group_index=None, dst_type=None, quant_mode="pertoken") -> (Tensor, Tensor, Tensor)
```

## 参数说明

- **x** (`Tensor`)：必选参数，需要进行量化的源数据张量，数据类型支持`float16`、`bfloat16`，数据格式支持ND，支持非连续的Tensor。输入`x`的维度必须大于1。进行int4量化时，要求x形状的最后一维是8的整数倍。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **smooth_scales** (`Tensor`)：可选参数，对`x`进行scales的张量，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，支持非连续的Tensor。
    -   在非MoE场景shape必须是1维，和`x`的最后一维相等。
    -   在MoE场景shape是2维[E, H]。其中E是专家数，取值范围在[1, 1024]且与group_index的第一维相同；H是x的最后一维。
    -   单算子模式下`smooth_scales`的dtype必须和`x`保持一致，图模式下可以不一致。
- **group_index** (`Tensor`)：可选参数，对`smooth_scales`进行分组下标（代表`x`的行数索引），仅在MoE场景下生效。数据类型支持`int32`，数据格式支持$ND$，支持非连续的Tensor。`group_index`的shape为[E,]，E的取值范围在[1, 1024]且与smooth_scales第一维相同。tensor的取值必须递增且范围为[1, S]，最后一个值必须等于S（S代表输入`x`的行数，是`x`的shape除最后一维度外的乘积）。
- **dst_type** (`ScalarType`)：可选参数，指定量化输出的类型，传None时当作`int8`处理。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`quint4x2`。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`、`quint4x2`。
- **quant_mode** (`str`)：可选参数，量化模式，支持"pertoken"、"pertensor"。默认值为"pertoken"。若`group_index`不为None，只支持"pertoken"。

## 返回值说明

- **y** (`Tensor`)：量化后的输出，数据类型由`dst_type`指定。当`dst_type`是`quint4x2`时，`y`的数据类型为`int32`，形状最后一维为`x`最后一维除以8，其余维度与`x`一致，每个`int32`元素包含8个`int4`结果。其他场景下`y`形状与输入`x`一致，数据类型由`dst_type`指定。
- **scale** (`Tensor`)：非对称动态量化过程中计算出的缩放系数，数据类型为`float32`。如果`quant_mode`是"pertoken"，shape为`x`的形状剔除最后一维。如果`quant_mode`是"pertensor"，shape为(1,)。
- **offset** (`Tensor`)：非对称动态量化过程中计算出的偏移系数，数据类型为`float32`，shape和`scale`一致。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   使用可选参数`smooth_scales`、`group_index`、`dst_type`时，必须使用关键字传参。

## 调用示例

-   单算子模式调用
    -   只有一个输入`x`，进行`int8`量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x)
        print(y, scale, offset)
        ```

    -   只有一个输入`x`，进行`int4`量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, dst_type=torch.quint4x2)
        print(y, scale, offset)
        ```

    -   使用`smooth_scales`输入，非MoE场景（不使用`group_index`），进行`int8`量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        smooth_scales = torch.rand((8,), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales)
        print(y, scale, offset)
        ```

    -   使用`smooth_scales`输入，MoE场景（使用`group_index`），进行`int8`量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        smooth_scales = torch.rand((2, 8), dtype=torch.half).npu()
        group_index = torch.Tensor([1, 3]).to(torch.int32).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales, group_index=group_index)
        print(y, scale, offset)
        ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    torch_npu.npu.set_compile_mode(jit_compile=True)
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    
    device=torch.device(f'npu:0')
    
    torch_npu.npu.set_device(device)
    
    class DynamicQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
    
        def forward(self, input_tensor, smooth_scales=None, group_index=None, dst_type=None):
            out, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(input_tensor, smooth_scales=smooth_scales, group_index=group_index, dst_type=dst_type)
            return out, scale, offset
    
    x = torch.randn((2, 4, 6),device='npu',dtype=torch.float16).npu()
    smooth_scales = torch.randn((6),device='npu',dtype=torch.float16).npu()
    dynamic_quant_model = DynamicQuantModel().npu()
    dynamic_quant_model = torch.compile(dynamic_quant_model, backend=npu_backend, dynamic=True)
    out, scale, offset = dynamic_quant_model(x, smooth_scales=smooth_scales)
    print(out)
    print(scale)
    print(offset)
    ```

