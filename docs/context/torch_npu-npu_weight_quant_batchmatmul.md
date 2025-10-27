# torch\_npu.npu\_weight\_quant\_batchmatmul<a name="ZH-CN_TOPIC_0000002231202136"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>      |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  | √   |
|<term>Atlas 推理系列加速卡产品</term>  | √   |

## 功能说明

- API功能：该接口用于实现矩阵乘计算中`weight`输入和输出的量化操作，支持pertensor、perchannel、pergroup多场景量化。

- 计算公式：

     $$
     y = x @ ANTIQUANT(weight) + bias 
     $$
     公式中的$weight$为伪量化场景的输入，其反量化公式$ANTIQUANT(weight)$ 为:
     $$
     ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
     $$
     当配置了`quant_scale`时，会对输出进行量化处理，其量化公式为:
     $$
     y = QUANT(x @ ANTIQUANT(weight) + bias) \\
     = (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset
     $$
     当`quant_scale`配置为None时，则直接输出：
     $$
     y = x @ ANTIQUANT(weight) + bias
     $$

## 函数原型

```
torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset=None, quant_scale=None, quant_offset=None, bias=None, antiquant_group_size=0, inner_precise=0) -> Tensor
```

## 参数说明

-   **x** (`Tensor`)：必选参数。即矩阵乘中的左矩阵。对应公式中的$x$。数据格式支持$ND$，支持带transpose的非连续的Tensor，支持输入维度为两维\(M, K\)。
    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持`float16`。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`。

-   **weight** (`Tensor`)：必选参数。即矩阵乘中的右矩阵。对应公式中的$weight$。支持带transpose的非连续的Tensor，支持输入维度为两维\(K, N\)，维度需与`x`保持一致。当数据格式为$ND$时，perchannel场景下为提高性能推荐使用transpose后的`weight`输入。
    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持`int8`。数据格式支持$ND$、$FRACTAL\_NZ$，其中$FRACTAL\_NZ$格式只在“图模式”有效，需依赖接口torch\_npu.npu\_format\_cast完成$ND$到$FRACTAL\_NZ$的转换，可参考[调用示例](#zh-cn_topic_0000001771071862_section14459801435)。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`int32`（通过`int32`承载`int4`的输入，可参考[torch\_npu.npu\_convert\_weight\_to\_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)的调用示例）。数据格式支持$ND$、$FRACTAL\_NZ$。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`、`int32`（通过`int32`承载`int4`的输入，可参考[torch\_npu.npu\_convert\_weight\_to\_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)的调用示例）。数据格式支持$ND$、$FRACTAL\_NZ$。

-   **antiquant\_scale** (`Tensor`)：必选参数。反量化的缩放因子，用于weight矩阵反量化，对应反量化公式中的$antiquantScale$，数据格式支持$ND$。支持带transpose的非连续的Tensor。`antiquant_scale`支持的shape与量化方式相关：

    -   per\_tensor模式：输入shape为\(1,\)或\(1, 1\)。
    -   per\_channel模式：输入shape为\(1, N\)或\(N,\)。
    -   per\_group模式：输入shape为\(ceil\(K, antiquant\_group\_size\),  N\)。

    `antiquant_scale`支持的dtype如下：

    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持`float16`，其数据类型需与`x`保持一致。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`bfloat16`、`int64`。
        -   若输入为`float16`、`bfloat16`，其数据类型需与`x`保持一致。
        -   若输入为`int64`，`x`数据类型必须为`float16`且不带transpose输入，同时`weight`数据类型必须为`int8`、数据格式为$ND$、带transpose输入，可参考[调用示例](#zh-cn_topic_0000001771071862_section14459801435)。此时只支持perchannel场景，M范围为\[1, 96\]，且K和N要求64对齐。

    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`、`int64`。
        -   若输入为`float16`、`bfloat16`，其数据类型需与`x`保持一致。
        -   若输入为`int64`，`x`数据类型必须为`float16`且不带transpose输入，同时`weight`数据类型必须为`int8`、数据格式为$ND$、带transpose输入，可参考[调用示例](#zh-cn_topic_0000001771071862_section14459801435)。此时只支持perchannel场景，M范围为\[1, 96\]，且K和N要求64对齐。

-   **antiquant\_offset** (`Tensor`)：可选参数。反量化的偏移量，用于weight矩阵反量化。对应反量化公式中的$antiquantOffset$，默认值为None，数据格式支持$ND$，支持带transpose的非连续的Tensor，支持输入维度为两维\(1, N\)或一维\(N, \)、\(1, \)。
    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持`float16`，其数据类型需与`antiquant_scale`保持一致。pergroup场景shape要求为\(ceil\_div\(K, antiquant\_group\_size\), N\)。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`bfloat16`、`int32`。pergroup场景shape要求为\(ceil\_div\(K, antiquant\_group\_size\), N\)。
        -   若输入为`float16`、`bfloat16`，其数据类型需与`antiquant_scale`保持一致。
        -   若输入为`int32`，`antiquant_scale`的数据类型必须为`int64`。

    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`bfloat16`、`int32`。pergroup场景shape要求为\(ceil\_div\(K, antiquant\_group\_size\), N\)。
        -   若输入为`float16`、`bfloat16`，其数据类型需与`antiquant_scale`保持一致。
        -   若输入为`int32`，`antiquant_scale`的数据类型必须为`int64`。

-   **quant\_scale** (`Tensor`)：可选参数。量化的缩放因子，用于输出矩阵的量化，默认值为None，仅在`weight`格式为$ND$时支持。数据类型支持`float32`、`int64`，数据格式支持$ND$，支持输入维度为两维\(1, N\)或一维\(N, \)、\(1, \)。当`antiquant_scale`的数据类型为`int64`时，此参数必须为空。
    -   <term>Atlas 推理系列加速卡产品</term>：暂不支持此参数。

-   **quant\_offset** (`Tensor`)：可选参数。量化的偏移量，用于输出矩阵的量化，对应量化公式中的$quantOffset$，默认值为None，仅在`weight`格式为$ND$时支持。数据类型支持`float32`，数据格式支持$ND$，支持输入维度为两维\(1, N\)或一维\(N, \)、\(1, \)。当`antiquant_scale`的数据类型为`int64`时，此参数必须为空。
    -   <term>Atlas 推理系列加速卡产品</term>：暂不支持此参数。

-   **bias** (`Tensor`)：可选参数。即矩阵乘中的偏置项，对应公式中的$bias$。默认值为None，数据格式支持$ND$，不支持非连续的Tensor，支持输入维度为两维\(1, N\)或一维\(N, \)。
    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持`float16`。
    -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float16`、`float32`。当`x`数据类型为`bfloat16`，`bias`需为`float32`；当`x`数据类型为`float16`，`bias`需为`float16`。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float16`、`float32`。当`x`数据类型为`bfloat16`，`bias`需为`float32`；当`x`数据类型为`float16`，`bias`需为`float16`。

-   **antiquant\_group\_size** (`int`)：可选参数。用于控制pergroup场景下group大小，其他量化场景不生效。默认值为0，pergroup场景下要求传入值的范围为\[32, K-1\]且必须是32的倍数。
-   **inner\_precise** (`int`)：可选参数。计算模式选择，默认为0。0表示高精度模式，1表示高性能模式，可能会影响精度。当`weight`以`int32`类型且以$FRACTAL\_NZ$格式输入，M不大于16的pergroup场景下可以设置为1，提升性能。其他场景不建议使用高性能模式。

## 返回值说明<a name="zh-cn_topic_0000001771071862_section22231435517"></a>
`Tensor`

当输入存在`quant_scale`时输出数据类型为`int8`，当输入不存在`quant_scale`时输出数据类型和输入`x`一致。

## 约束说明<a name="zh-cn_topic_0000001771071862_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。当输入`weight`为$FRACTAL\_NZ$格式时暂不支持单算子调用，只支持图模式调用。
-   `x`和`weight`后两维必须为\(M, K\)和\(K, N\)格式，K、N的范围为\[1, 65535\]；在`x`为非转置时，M的范围为\[1, 2^31-1\]，在`x`为转置时，M的范围为\[1, 65535\]。
-   不支持空Tensor输入。
-   `antiquant_scale`和`antiquant_offset`的输入shape要保持一致。
-   `quant_scale`和`quant_offset`的输入shape要保持一致，且`quant_offset`不能独立于`quant_scale`存在。
-   如需传入`int64`数据类型的`quant_scale`，需要提前调用`torch_npu.npu_trans_quant_param`接口将数据类型为`float32`的`quant_scale`和`quant_offset`转换为数据类型为`int64`的`quant_scale`输入，可参考[调用示例](#zh-cn_topic_0000001771071862_section14459801435)。
-   当输入`weight`为$FRACTAL\_NZ$格式且类型为`int32`时，perchannel场景需满足`weight`为转置输入；pergroup场景需满足`x`为转置输入，`weight`为非转置输入，`antiquant_group_size`为64或128，K为`antiquant_group_size`对齐，N为64对齐。
-   不支持输入`weight`` shape为\(1, 8\)且类型为`int4`，同时`weight`带有transpose的场景，否则会报错`x`矩阵和`weight`矩阵K轴不匹配，该场景建议走非量化算子获取更高精度和性能。
-   当`antiquant_scale`为`float16`、`bfloat16`，单算子模式要求`x`和`antiquant_scale`数据类型一致，图模式允许不一致，如果出现不一致，接口内部会自行判断是否转换成一致的数据类型。用户可dump图信息查看实际参与计算的数据类型。

## 调用示例<a name="zh-cn_topic_0000001771071862_section14459801435"></a>

-   单算子模式调用
    -   weight非transpose+quant\_scale场景，仅支持如下产品：

        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

            ```python
             import torch
             import torch_npu
             # 输入int8+ND 
             cpu_x = torch.randn((8192, 320),dtype=torch.float16)
             cpu_weight = torch.randint(low=-8, high=8, size=(320, 256),dtype=torch.int8)
             cpu_antiquantscale = torch.randn((1, 256),dtype=torch.float16)
             cpu_antiquantoffset = torch.randn((1, 256),dtype=torch.float16)
             cpu_quantscale = torch.randn((1, 256),dtype=torch.float32)
             cpu_quantoffset = torch.randn((1, 256),dtype=torch.float32)
             quantscale= torch_npu.npu_trans_quant_param(cpu_quantscale.npu(), cpu_quantoffset.npu())
             npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(),quantscale.npu())
            ```

    -   weight transpose+antiquant\_scale场景，仅支持如下产品：

        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
        -   <term>Atlas 推理系列加速卡产品</term>

            ```python
             import torch
             import torch_npu
             cpu_x = torch.randn((96, 320),dtype=torch.float16)
             cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
             cpu_antiquantscale = torch.randn((256),dtype=torch.float16)
             # 构建int64类型的scale参数
             antiquant_scale = torch_npu.npu_trans_quant_param(cpu_antiquantscale.to(torch.float32).npu()).reshape(256, 1)
             cpu_antiquantoffset = torch.randint(-128, 127, (256, 1), dtype=torch.int32)
             npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.transpose(-1,-2).npu(), antiquant_scale.transpose(-1,-2).npu(), cpu_antiquantoffset.transpose(-1,-2).npu())
            ```

    -   weight transpose+antiquant\_scale场景，仅支持如下产品：

        -   <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
        -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

            ```python
             import torch
             import torch_npu
             # 输入int8+ND 
             cpu_x = torch.randn((96, 320),dtype=torch.float16)
             cpu_weight = torch.randint(low=-8, high=8, size=(256, 320),dtype=torch.int8)
             cpu_antiquantscale = torch.randn((256,1),dtype=torch.float16)
             cpu_antiquantoffset = torch.randint(-128, 127, (256,1), dtype=torch.float16)
             npu_out = torch_npu.npu_weight_quant_batchmatmul(cpu_x.npu(), cpu_weight.npu().transpose(-1, -2), cpu_antiquantscale.npu().transpose(-1, -2), cpu_antiquantoffset.npu().transpose(-1, -2))
            ```

-   图模式调用
    -   weight输入为ND格式

        ```python
         # 图模式
         import torch
         import torch_npu
         import  torchair as tng
         from torchair.configs.compiler_config import CompilerConfig
         config = CompilerConfig()
         config.debug.graph_dump.type = "pbtxt"
         npu_backend = tng.get_npu_backend(compiler_config=config)
        
         cpu_x = torch.randn((8192, 320),device='npu',dtype=torch.bfloat16)
         cpu_weight = torch.randint(low=-8, high=8, size=(320, 256), dtype=torch.int8, device='npu')
         cpu_antiquantscale = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
         cpu_antiquantoffset = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
        
         class MyModel(torch.nn.Module):
             def __init__(self):
                super().__init__()
        
             def forward(self, x, weight, antiquant_scale, antiquant_offset, quant_scale,quant_offset, bias, antiquant_group_size):
                return torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset, quant_scale ,quant_offset, bias, antiquant_group_size)
        
         cpu_model = MyModel()
         model = cpu_model.npu()
         model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
         npu_out = model(cpu_x.npu(), cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu(), None, None, None, 0)
        ```

    -   weight输入为FRACTAL\_NZ格式，仅支持<term>Atlas 推理系列加速卡产品</term>。

        ```python
         import torch_npu
         import torch
         from torchair.configs.compiler_config import CompilerConfig
         import torchair as tng
         config = CompilerConfig()
         config.debug.graph_dump.type = "pbtxt"
         npu_backend = tng.get_npu_backend(compiler_config=config)
         class NPUQuantizedLinearA16W8(torch.nn.Module):
             def __init__(self,
                         weight,
                         antiquant_scale,
                         antiquant_offset,
                         quant_offset=None,
                         quant_scale=None,
                         bias=None,
                         transpose_x=False,
                         transpose_weight=True,
                         w4=False):
                super().__init__()
        
                self.dtype = torch.float16
                self.weight = weight.to(torch.int8).npu()
                self.transpose_weight = transpose_weight
        
                if self.transpose_weight:
                    self.weight = torch_npu.npu_format_cast(self.weight.contiguous(), 29)
                else:
                    self.weight = torch_npu.npu_format_cast(self.weight.transpose(0, 1).contiguous(), 29) # n,k ->nz
        
                self.bias = None
                self.antiquant_scale = antiquant_scale
                self.antiquant_offset = antiquant_offset
                self.quant_offset = quant_offset
                self.quant_scale = quant_scale
                self.transpose_x = transpose_x
        
             def forward(self, x):
                x = torch_npu.npu_weight_quant_batchmatmul(x.transpose(0, 1) if self.transpose_x else x,
                                                           self.weight.transpose(0, 1),
                                                           self.antiquant_scale.transpose(0, 1),
                                                           self.antiquant_offset.transpose(0, 1),
                                                           self.quant_scale,
                                                           self.quant_offset,
                                                           self.bias)
                return x
        
        
         m, k, n = 4, 1024, 4096
         cpu_x = torch.randn((m, k),dtype=torch.float16)
         cpu_weight = torch.randint(1, 10, (k, n),dtype=torch.int8)
         cpu_weight = cpu_weight.transpose(-1, -2)
        
         cpu_antiquantscale = torch.randn((1, n),dtype=torch.float16)
         cpu_antiquantoffset = torch.randn((1, n),dtype=torch.float16)
         cpu_antiquantscale = cpu_antiquantscale.transpose(-1, -2)
         cpu_antiquantoffset = cpu_antiquantoffset.transpose(-1, -2)
         model = NPUQuantizedLinearA16W8(cpu_weight.npu(), cpu_antiquantscale.npu(), cpu_antiquantoffset.npu())
         model = torch.compile(model, backend=npu_backend, dynamic=True)
         out = model(cpu_x.npu())
        ```

