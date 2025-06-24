# torch\_npu.npu\_weight\_quant\_batchmatmul<a name="ZH-CN_TOPIC_0000002231202136"></a>

## 功能说明<a name="zh-cn_topic_0000001771071862_section14441124184110"></a>

该接口用于实现矩阵乘计算中weight输入和输出的量化操作，支持per-tensor、per-channel、per-group多场景量化。

不同产品支持的量化算法不同，如[表1](#zh-cn_topic_0000001771071862_table178313019319)所示。

**表1** 支持的量化场景

<a name="zh-cn_topic_0000001771071862_table178313019319"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001771071862_row383408315"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001771071862_p36332517320"><a name="zh-cn_topic_0000001771071862_p36332517320"></a><a name="zh-cn_topic_0000001771071862_p36332517320"></a>产品型号</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001771071862_p1563312515310"><a name="zh-cn_topic_0000001771071862_p1563312515310"></a><a name="zh-cn_topic_0000001771071862_p1563312515310"></a>量化方式</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001771071862_row1083501834"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001771071862_p10633205234"><a name="zh-cn_topic_0000001771071862_p10633205234"></a><a name="zh-cn_topic_0000001771071862_p10633205234"></a><span id="zh-cn_topic_0000001771071862_ph6939149151811"><a name="zh-cn_topic_0000001771071862_ph6939149151811"></a><a name="zh-cn_topic_0000001771071862_ph6939149151811"></a><a name="zh-cn_topic_0000001771071862_zh-cn_topic_0000001312391781_term15651172142210"></a><a name="zh-cn_topic_0000001771071862_zh-cn_topic_0000001312391781_term15651172142210"></a><term>Atlas 推理系列加速卡产品</term></span></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001771071862_p17633051430"><a name="zh-cn_topic_0000001771071862_p17633051430"></a><a name="zh-cn_topic_0000001771071862_p17633051430"></a>per-tensor、per-channel<span id="zh-cn_topic_0000001771071862_ph17785145061412"><a name="zh-cn_topic_0000001771071862_ph17785145061412"></a><a name="zh-cn_topic_0000001771071862_ph17785145061412"></a>、per-group</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001771071862_row384120537"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001771071862_p563316513320"><a name="zh-cn_topic_0000001771071862_p563316513320"></a><a name="zh-cn_topic_0000001771071862_p563316513320"></a><span id="zh-cn_topic_0000001771071862_ph18633185637"><a name="zh-cn_topic_0000001771071862_ph18633185637"></a><a name="zh-cn_topic_0000001771071862_ph18633185637"></a><a name="zh-cn_topic_0000001771071862_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001771071862_zh-cn_topic_0000001312391781_term11962195213215"></a><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></span></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001771071862_p4633550315"><a name="zh-cn_topic_0000001771071862_p4633550315"></a><a name="zh-cn_topic_0000001771071862_p4633550315"></a>per-tensor、per-channel、per-group</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001771071862_row11841505315"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001771071862_p1163315519316"><a name="zh-cn_topic_0000001771071862_p1163315519316"></a><a name="zh-cn_topic_0000001771071862_p1163315519316"></a><span id="zh-cn_topic_0000001771071862_ph16331857319"><a name="zh-cn_topic_0000001771071862_ph16331857319"></a><a name="zh-cn_topic_0000001771071862_ph16331857319"></a><a name="zh-cn_topic_0000001771071862_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001771071862_zh-cn_topic_0000001312391781_term1253731311225"></a><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001771071862_p4633125733"><a name="zh-cn_topic_0000001771071862_p4633125733"></a><a name="zh-cn_topic_0000001771071862_p4633125733"></a>per-tensor、per-channel、per-group</p>
</td>
</tr>
</tbody>
</table>

## 函数原型<a name="zh-cn_topic_0000001771071862_section45077510411"></a>

-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas 推理系列加速卡产品</term>：

    ```
    torch_npu.npu_weight_quant_batchmatmul(Tensor x, Tensor weight, Tensor antiquant_scale, Tensor? antiquant_offset=None, Tensor? quant_scale=None, Tensor? quant_offset=None, Tensor? bias=None, int antiquant_group_size=0, int inner_precise=0) -> Tensor
    ```

## 参数说明<a name="zh-cn_topic_0000001771071862_section112637109429"></a>

-   x : Tensor类型，即矩阵乘中的x。数据格式支持ND，支持带transpose的非连续的Tensor，支持输入维度为两维\(M, K\) 。
    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持float16。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、bfloat16。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、bfloat16。

-   weight：Tensor类型，即矩阵乘中的weight。支持带transpose的非连续的Tensor，支持输入维度为两维\(K, N\)，维度需与x保持一致。当数据格式为ND时，per-channel场景下为提高性能推荐使用transpose后的weight输入。
    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持int8。数据格式支持ND、FRACTAL\_NZ，其中FRACTAL\_NZ格式只在“图模式”有效，需依赖接口torch\_npu.npu\_format\_cast完成ND到FRACTAL\_NZ的转换，可参考[调用示例](#zh-cn_topic_0000001771071862_section14459801435)。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持int8、int32（通过int32承载int4的输入，可参考[torch\_npu.npu\_convert\_weight\_to\_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)的调用示例）。数据格式支持ND、FRACTAL\_NZ。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持int8、int32（通过int32承载int4的输入，可参考[torch\_npu.npu\_convert\_weight\_to\_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)的调用示例）。数据格式支持ND、FRACTAL\_NZ。

-   antiquant\_scale：Tensor类型，反量化的scale，用于weight矩阵反量化，数据格式支持ND。支持带transpose的非连续的Tensor。antiquant\_scale支持的shape与量化方式相关：

    -   per\_tensor模式：输入shape为\(1,\)或\(1, 1\)。
    -   per\_channel模式：输入shape为\(1, N\)或\(N,\)。
    -   per\_group模式：输入shape为\(ceil\(K, antiquant\_group\_size\),  N\)。

    antiquant\_scale支持的dtype如下：

    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持float16，其数据类型需与x保持一致。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、bfloat16、int64。
        -   若输入为float16、bfloat16，其数据类型需与x保持一致。
        -   若输入为int64，x数据类型必须为float16且不带transpose输入，同时weight数据类型必须为int8、数据格式为ND、带transpose输入，可参考[调用示例](#zh-cn_topic_0000001771071862_section14459801435)。此时只支持per-channel场景，M范围为\[1, 96\]，且K和N要求64对齐。

    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、bfloat16、int64。
        -   若输入为float16、bfloat16，其数据类型需与x保持一致。
        -   若输入为int64，x数据类型必须为float16且不带transpose输入，同时weight数据类型必须为int8、数据格式为ND、带transpose输入，可参考[调用示例](#zh-cn_topic_0000001771071862_section14459801435)。此时只支持per-channel场景，M范围为\[1, 96\]，且K和N要求64对齐。

-   antiquant\_offset：Tensor类型，反量化的offset，用于weight矩阵反量化。可选参数，默认值为None，数据格式支持ND，支持带transpose的非连续的Tensor，支持输入维度为两维\(1, N\)或一维\(N, \)、\(1, \)。
    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持float16，其数据类型需与antiquant\_scale保持一致。per-group场景shape要求为\(ceil\_div\(K, antiquant\_group\_size\), N\)。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、bfloat16、int32。per-group场景shape要求为\(ceil\_div\(K, antiquant\_group\_size\), N\)。
        -   若输入为float16、bfloat16，其数据类型需与antiquant\_scale保持一致。
        -   若输入为int32，antiquant\_scale的数据类型必须为int64。

    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、bfloat16、int32。per-group场景shape要求为\(ceil\_div\(K, antiquant\_group\_size\), N\)。
        -   若输入为float16、bfloat16，其数据类型需与antiquant\_scale保持一致。
        -   若输入为int32，antiquant\_scale的数据类型必须为int64。

-   quant\_scale：Tensor类型，量化的scale，用于输出矩阵的量化，可选参数，默认值为None，仅在weight格式为ND时支持。数据类型支持float32、int64，数据格式支持ND，支持输入维度为两维\(1, N\)或一维\(N, \)、\(1, \)。当antiquant\_scale的数据类型为int64时，此参数必须为空。
    -   <term>Atlas 推理系列加速卡产品</term>：暂不支持此参数。

-   quant\_offset: Tensor类型，量化的offset，用于输出矩阵的量化，可选参数，默认值为None，仅在weight格式为ND时支持。数据类型支持float32，数据格式支持ND，支持输入维度为两维\(1, N\)或一维\(N, \)、\(1, \)。当antiquant\_scale的数据类型为int64时，此参数必须为空。
    -   <term>Atlas 推理系列加速卡产品</term>：暂不支持此参数。

-   bias：Tensor类型，即矩阵乘中的bias，可选参数，默认值为None，数据格式支持ND，不支持非连续的Tensor，支持输入维度为两维\(1, N\)或一维\(N, \)。
    -   <term>Atlas 推理系列加速卡产品</term>：数据类型支持float16。
    -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持float16、float32。当x数据类型为bfloat16，bias需为float32；当x数据类型为float16，bias需为float16。
    -   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持float16、float32。当x数据类型为bfloat16，bias需为float32；当x数据类型为float16，bias需为float16。

-   antiquant\_group\_size：int类型，用于控制per-group场景下group大小，其他量化场景不生效。可选参数。默认值为0，per-group场景下要求传入值的范围为\[32, K-1\]且必须是32的倍数。
-   inner\_precise:  int类型，计算模式选择，默认为0。0表示高精度模式，1表示高性能模式，可能会影响精度。当weight以int32类型且以FRACTAL\_NZ格式输入，M不大于16的per-group场景下可以设置为1，提升性能。其他场景不建议使用高性能模式。

## 输出说明<a name="zh-cn_topic_0000001771071862_section22231435517"></a>

输出为Tensor类型，代表计算结果。当输入存在quant\_scale时输出数据类型为int8，当输入不存在quant\_scale时输出数据类型和输入x一致。

## 约束说明<a name="zh-cn_topic_0000001771071862_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。当输入weight为FRACTAL\_NZ格式时暂不支持单算子调用，只支持图模式调用。
-   x和weight后两维必须为\(M, K\)和\(K, N\)格式，K、N的范围为\[1, 65535\]；在x为非转置时，M的范围为\[1, 2^31-1\]，在x为转置时，M的范围为\[1, 65535\]。
-   不支持空Tensor输入。
-   antiquant\_scale和antiquant\_offset的输入shape要保持一致。
-   quant\_scale和quant\_offset的输入shape要保持一致，且quant\_offset不能独立于quant\_scale存在。
-   如需传入int64数据类型的quant\_scale，需要提前调用torch\_npu.npu\_trans\_quant\_param接口将数据类型为float32的quant\_scale和quant\_offset转换为数据类型为int64的quant\_scale输入，可参考[调用示例](#zh-cn_topic_0000001771071862_section14459801435)。
-   当输入weight为FRACTAL\_NZ格式且类型为int32时，per-channel场景需满足weight为转置输入；per-group场景需满足x为转置输入，weight为非转置输入，antiquant\_group\_size为64或128，K为antiquant\_group\_size对齐，N为64对齐。
-   不支持输入weight shape为\(1, 8\)且类型为int4，同时weight带有transpose的场景，否则会报错x矩阵和weight矩阵K轴不匹配，该场景建议走非量化算子获取更高精度和性能。
-   当antiquant\_scale为float16、bfloat16，单算子模式要求x和antiquant\_scale数据类型一致，图模式允许不一致，如果出现不一致，接口内部会自行判断是否转换成一致的数据类型。用户可dump图信息查看实际参与计算的数据类型。

## 支持的型号<a name="zh-cn_topic_0000001771071862_section1414151813182"></a>

-   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
-   <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
-   <term>Atlas 推理系列加速卡产品</term>

## 调用示例<a name="zh-cn_topic_0000001771071862_section14459801435"></a>

-   单算子模式调用
    -   weight非transpose+quant\_scale场景，仅支持如下产品：

        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
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

        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
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

        -   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>
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

