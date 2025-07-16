# torch\_npu.npu\_dynamic\_quant\_asymmetric<a name="ZH-CN_TOPIC_0000002153860105"></a>

## 功能说明<a name="zh-cn_topic_0000002057983185_section14441124184110"></a>

-   算子功能：

    对输入的张量进行动态非对称量化。支持pertoken、pertensor和MoE（Mixture of Experts，混合专家模型）场景。

-   计算公式：假设待量化张量为x，

    ![](./figures/zh-cn_formulaimage_0000002128390261.png)

    -   rowMax、rowMin代表按行取最大值、按行取最小值，此处的“行”对应x最后一个维度的数据，即一个token。
    -   DST\_MAX、DST\_MIN分别对应量化后dtype的最大值和最小值。
    -   若使用smooth quant，会引入smooth\_scales输入，其形状与x最后一个维度大小一致，在进行量化前，会先令x乘以smooth\_scales，再按上述公式进行量化
    -   若使用smooth quant，MoE（Mixture of Experts，混合专家模型）场景下会引入smooth\_scales和group\_index，此时smooth\_scales中包含多组smooth向量，按group\_index中的数值作用到x的不同行上。具体的，假如x包含m个token，smooth\_scales有n行，smooth\_scales\[0\]会作用到x\[0:group\_index\[0\]\]上，smooth\_scales\[i\]会作用到x\[group\_index\[i-1\]: group\_index\[i\]\]上，i=\[1, 2, ..., n-1\]。
    -   在pertensor场景下，rowMax、rowMin表示求整个tensor的最大值、最小值。

## 函数原型<a name="zh-cn_topic_0000002057983185_section45077510411"></a>

```
torch_npu.npu_dynamic_quant_asymmetric(Tensor x, *, Tensor? smooth_scales=None, Tensor? group_index=None, ScalarType? dst_type=None) -> (Tensor, Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002057983185_section112637109429"></a>

-   x：Tensor类型，需要进行量化的源数据张量，必选输入，数据类型支持torch.float16、torch.bfloat16，数据格式支持ND，支持非连续的Tensor。输入x的维度必须大于1。进行int4量化时，要求x形状的最后一维是8的整数倍。
-   smooth\_scales：Tensor类型，对x进行scales的张量，可选参数，数据类型支持torch.float16、torch.bfloat16，数据格式支持ND，支持非连续的Tensor。
    -   在非MoE场景shape必须是1维，和x的最后一维相等。
    -   在MoE场景shape是2维\[E, H\]。其中E是专家数，取值范围在\[1, 1024\]且与group\_index的第一维相同；H是x的最后一维。
    -   单算子模式下smooth\_scales的dtype必须和x保持一致，图模式下可以不一致。

-   group\_index：Tensor类型，对smooth\_scales进行分组下标（代表x的行数索引），可选参数，仅在MoE场景下生效。数据类型支持int32，数据格式支持ND，支持非连续的Tensor。group\_index的shape为\[E,\]，E的取值范围在\[1, 1024\]且与smooth\_scales第一维相同。tensor的取值必须递增且范围为\[1, S\]，最后一个值必须等于S（S代表输入x的行数，是x的shape除最后一维度外的乘积）。
-   dst\_type：ScalarType类型，指定量化输出的类型，可选参数，传None时当作torch.int8处理。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：支持取值torch.int8、torch.quint4x2。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持取值torch.int8、torch.quint4x2。

-   quant\_mode：str类型，量化模式，支持"pertoken"、"pertensor"。默认值为"pertoken"。
    -   如果group\_index不为None，只支持"pertoken"。

## 输出说明<a name="zh-cn_topic_0000002057983185_section22231435517"></a>

-   y：量化后的输出Tensor，数据类型由dst\_type指定。当dst\_type是torch.quint4x2时，y的数据类型为int32，形状最后一维为x最后一维除以8，其余维度与x一致，每个int32元素包含8个int4结果。其他场景下y形状与输入x一致，数据类型由dst\_type指定。
-   scale：Tensor类型，非对称动态量化过程中计算出的缩放系数，数据类型为float32。如果quant\_mode是"pertoken"，shape为x的形状剔除最后一维。如果quant\_mode是"pertensor"，shape为\(1,\)。

-   offset：Tensor类型，非对称动态量化过程中计算出的偏移系数，数据类型为float32，shape和scale一致。

## 约束说明<a name="zh-cn_topic_0000002057983185_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   使用可选参数smooth\_scales、group\_index、dst\_type时，必须使用关键字传参。

## 支持的型号<a name="zh-cn_topic_0000002057983185_section3995315192919"></a>

-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

## 调用示例<a name="zh-cn_topic_0000002057983185_section14459801435"></a>

-   单算子模式调用
    -   只有一个输入x，进行int8量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x)
        print(y, scale, offset)
        ```

    -   只有一个输入x，进行int4量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, dst_type=torch.quint4x2)
        print(y, scale, offset)
        ```

    -   使用smooth\_scales输入，非MoE场景（不使用group\_index），进行int8量化

        ```python
        import torch
        import torch_npu
        x = torch.rand((3, 8), dtype=torch.half).npu()
        smooth_scales = torch.rand((8,), dtype=torch.half).npu()
        y, scale, offset = torch_npu.npu_dynamic_quant_asymmetric(x, smooth_scales=smooth_scales)
        print(y, scale, offset)
        ```

    -   使用smooth\_scales输入，MoE场景（使用group\_index），进行int8量化

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

