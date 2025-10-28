# torch_npu.contrib.module.LinearWeightQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>    |    √     |
| <term>Atlas 推理系列产品</term>     |    √     |

## 功能说明

LinearWeightQuant是对torch_npu.npu_weight_quant_batchmatmul接口的封装类，完成矩阵乘计算中的weight输入和输出的量化操作，支持pertensor、perchannel、pergroup多场景量化。

当前<term>Atlas 推理系列产品</term>仅支持perchannel量化。

## 函数原型

```
torch_npu.contrib.module.LinearWeightQuant(in_features, out_features, bias=True, device=None, dtype=None, antiquant_offset=False, quant_scale=False, quant_offset=False, antiquant_group_size=0, inner_precise=0)
```

## 参数说明

### 计算参数
- **in_features**（`int`）：伪量化matmul计算中的k轴的值。
- **out_features**（`int`）：伪量化matmul计算中的n轴的值。
- **bias**（`bool`）：可选参数，默认为True，代表是否需要bias计算参数。如果设置成False，则bias不会加入伪量化matmul的计算。
- **device**（`string`）：可选参数，用于执行model的device名称，默认为None。
- **dtype**（`torch.dtype`）：伪量化matmul运算中的输入`x`的dtype，可选参数，默认为None。
- **antiquant_offset**（`bool`）：可选参数，默认为False，代表是否需要antiquant_offset计算参数。如果设置成False，则weight矩阵反量化时无需设置offset。
- **quant_scale**（`bool`）：可选参数，默认为False，代表是否需要quant_scale计算参数。如果设置成False，则伪量化输出不会进行量化计算。
- **quant_offset**（`bool`）：可选参数，默认为False，代表是否需要quant_offset计算参数。如果设置成False，则对伪量化输出进行量化计算时无需设置offset。
- **antiquant_group_size**（`int`）：可选参数，用于控制pergroup场景下的group大小，当前默认为0。传入值的范围为[32,K-1]且值要求是32的倍数。<term>Atlas 推理系列产品</term>：暂不支持此参数。
- **inner_precise**（`int`）：计算模式选择，默认为0。0表示高精度模式，1表示高性能模式，可能会影响精度。当`weight`以`int32`类型且以FRACTAL_NZ格式输入，M不大于16的pergroup场景下可以设置为1，提升性能。其他场景不建议使用高性能模式。

###  计算输入

**x**（`Tensor`）：Tensor类型，即矩阵乘中的x。数据格式支持$ND$，支持输入维度为两维(M, K) 。

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品</term>  ：数据类型支持`float16`、`bfloat16`。
- <term>Atlas 推理系列产品</term> ：数据类型仅支持`float16`。

## 变量说明

- **weight**（`Tensor`）：即矩阵乘中的weight。数据格式支持$ND$、FRACTAL_NZ，支持非连续的Tensor，支持输入维度为两维(N, K)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品</term>：数据类型支持`int8`、`int32`（通过`int32`承载`int4`的输入，可以参考[torch_npu.npu_convert_weight_to_int4pack](torch_npu-npu_convert_weight_to_int4pack.md)的调用示例）。
    - <term>Atlas 推理系列产品</term>：数据类型支持`int8`。weight FRACTAL_NZ格式只在图模式有效，依赖接口torchair.experimental.inference.use_internal_format_weight完成数据格式从ND到FRACTAL_NZ转换，可参考[调用示例](#section00001)。

- **antiquant_scale**（`Tensor`）：反量化的scale，用于weight矩阵反量化。数据格式支持$ND$。支持非连续的Tensor，支持输入维度为两维(N, 1)或一维(N,)、(1,)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品</term>  ：数据类型支持`float16`、`bfloat16`、`int64`。pergroup场景shape要求为(N, ceil_div(K, antiquant_group_size))。
         - 若数据类型为`float16`、`bfloat16`，其数据类型需要和`x`保持一致。
         - 若数据类型为`int64`，则`x`的数据类型必须为`float16`且不带transpose输入，同时`weight`的数据类型必须为`int8`、数据格式为$ND$、带transpose输入，可参考[调用示例](#section00001)。此时只支持perchannel场景，M范围为[1, 96]，且K和N要求64对齐。

     - <term>Atlas 推理系列产品</term> ：数据类型支持`float16`，其数据类型需要和`x`保持一致。

- **antiquant_offset**（`Tensor`）：反量化的offset，用于weight矩阵反量化。数据格式支持$ND$。支持非连续的Tensor，支持输入维度为两维(N, 1)或一维(N,)、(1,)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>/<term>Atlas A3 训练系列产品</term>  ：数据类型支持`float16`、`bfloat16`、`int32`。pergroup场景shape要求为(N, ceil_div(K, antiquant_group_size))。
        - 若数据类型为`float16`、`bfloat16`，其数据类型需要和`antiquant_scale`保持一致。
        - 若数据类型为`int32`，则`antiquant_scale`的数据类型必须为`int64`。

    - <term>Atlas 推理系列产品</term> ：数据类型仅支持`float16`，其数据类型需要和`antiquant_scale`保持一致。

- **quant_scale**（`Tensor`）：量化的scale，用于输出矩阵的量化。仅在weight格式为$ND$时支持，数据格式支持$ND$，数据类型支持`float32`、`int64`，支持输入维度为两维(1, N)或一维(N,)、(1,)。当`antiquant_scale`的数据类型为`int64`时，此参数必须为空。<term>Atlas 推理系列产品</term> ：暂不支持此参数。

- **quant_offset**（`Tensor`）：量化的offset，用于输出矩阵的量化。仅在weight格式为$ND$时支持，数据格式支持$ND$，数据类型支持`float32`，支持输入维度为两维(1, N)或一维(N,)、(1, )。当`antiquant_scale`的数据类型为`int64`时，此参数必须为空。<term>Atlas 推理系列产品</term> ：暂不支持此参数。

- **bias**（`Tensor`）：即矩阵乘中的bias，数据格式支持$ND$，数据类型支持`float16`、`float32`，支持非连续的Tensor，支持输入维度为两维(1, N)或一维(N,)、(1,)。
- **antiquant_group_size**（`int`）：用于控制pergroup场景下的group大小，默认为0。传入值的范围为[32, K-1]且值要求是32的倍数。<term>Atlas 推理系列产品</term> ：暂不支持此参数。

## 返回值说明
`Tensor`

代表计算结果。当输入存在`quant_scale`时输出数据类型为`int8`，当输入不存在`quant_scale`时输出数据类型和输入`x`一致。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。当输入`weight`为FRACTAL_NZ格式时暂不支持单算子调用，只支持图模式调用。
- `x`和`weight`后两维必须为(M, K)和(N, K)格式，K、N的范围为[1, 65535]；在`x`为非转置时，M的范围为[1, 2^31-1]；在`x`为转置时，M的范围为[1, 65535]。
- 不支持空Tensor输入。
- `antiquant_scale`和`antiquant_offset`的输入shape要保持一致。
- `quant_scale`和`quant_offset`的输入shape要保持一致，且`quant_offset`不能独立于`quant_scale`存在。
- 当`x`输入类型为`bfloat16`类型时，`bias`的输入类型为`float32`；当`x`输入类型为`float16`类型时，`bias`的输入类型为`float16`。
- 如需传入`int64`数据类型的`quant_scale`，需要提前调用torch_npu.npu_trans_quant_param接口将数据类型为`float32`的`quant_scale`和`quant_offset`转换为数据类型为`int64`的`quant_scale`输入，可参考[调用示例](#section00001)。
- 当输入`weight`为FRACTAL_NZ格式且类型为`int32`时，perchannel场景需满足`weight`为转置输入；pergroup场景需满足`x`为转置输入，`weight`为非转置输入，`antiquant_group_size`为64或128，K为`antiquant_group_size`对齐，N为64对齐。


## 调用示例<a name="section00001"></a>

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from torch_npu.contrib.module import LinearWeightQuant
    x = torch.randn((8192, 320),device='npu',dtype=torch.float16)
    weight = torch.randn((320, 256),device='npu',dtype=torch.int8)
    antiquantscale = torch.randn((1, 256),device='npu',dtype=torch.float16)
    antiquantoffset = torch.randn((1, 256),device='npu',dtype=torch.float16)
    quantscale = torch.randn((1, 256),device='npu',dtype=torch.float)
    quantoffset = torch.randn((1, 256),device='npu',dtype=torch.float)
    model = LinearWeightQuant(in_features=320,
                              out_features=256,
                              bias=False,
                              dtype=torch.float16,
                              antiquant_offset=True,
                              quant_scale=True,
                              quant_offset=True,
                              antiquant_group_size=0,
                              device=torch.device(f'npu:0')
                              )
    model.npu()
    model.weight.data = weight.transpose(-1, -2)
    model.antiquant_scale.data = antiquantscale.transpose(-1, -2)
    model.antiquant_offset.data = antiquantoffset.transpose(-1, -2)
    model.quant_scale.data = torch_npu.npu_trans_quant_param(quantscale, quantoffset)
    model.quant_offset.data = quantoffset
    out = model(x)
    ```

- 图模式调用

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torch_npu.contrib.module import LinearWeightQuant
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)
    x = torch.randn((8192, 320),device='npu',dtype=torch.bfloat16)
    weight = torch.randn((320, 256),device='npu',dtype=torch.int8)
    antiquantscale = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
    antiquantoffset = torch.randn((1, 256),device='npu',dtype=torch.bfloat16)
    quantscale = torch.randn((1, 256),device='npu',dtype=torch.float)
    quantoffset = torch.randn((1, 256),device='npu',dtype=torch.float)
    model = LinearWeightQuant(in_features=320,
                              out_features=256,
                              bias=False,
                              dtype=torch.bfloat16,
                              antiquant_offset=True,
                              quant_scale=True,
                              quant_offset=True,
                              antiquant_group_size=0,
                              device=torch.device(f'npu:0')
                              )
    model.npu()
    model.weight.data = weight.transpose(-1, -2)
    model.antiquant_scale.data = antiquantscale.transpose(-1, -2)
    model.antiquant_offset.data = antiquantoffset.transpose(-1, -2)
    model.quant_scale.data = quantscale
    model.quant_offset.data = quantoffset
    tng.experimental.inference.use_internal_format_weight(model) # 将ND的weight输入转为FRACTAL_NZ格式
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    out = model(x)
    ```

