# torch_npu.npu_quant_matmul

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √   |
|<term>Atlas 推理系列加速卡产品</term> | √   |

## 功能说明

- API功能：完成量化的矩阵乘计算，最小支持输入维度为2维，最大支持输入维度为6维。

- 计算公式：
    - 无`bias`：
    $$
    out = x1 \mathbin{@} x2 * \text{scale} + \text{offset}
    $$
    - `bias`为`int32`：
    $$
    out = (x1 \mathbin{@} x2 + \text{bias}) * \text{scale} + \text{offset}
    $$
    - `bias`为`bfloat16`或`float32`（无`offset`）：
    $$
    out = x1 \mathbin{@} x2 * \text{scale} + \text{bias}
    $$

## 函数原型

```
npu_quant_matmul(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None, output_dtype=None, x1_dtype=None, x2_dtype=None, pertoken_scale_dtype=None, scale_dtype=None, group_sizes=None) -> Tensor
```

## 参数说明

- **x1** (`Tensor`)：必选参数，输入张量，表示矩阵乘法中的左矩阵，数据格式支持$ND$，shape需要在2-6维范围。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型支持`int8`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`和`int32`。其中`int32`表示`int4`类型矩阵乘计算，每个`int32`数据存放8个`int4`数据。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`和`int32`。其中`int32`表示`int4`类型矩阵乘计算，每个`int32`数据存放8个`int4`数据。

- **x2** (`Tensor`)：必选参数，输入张量，表示矩阵乘法中的右矩阵，其与`x1`的数据类型须保持一致。数据格式支持$ND$，shape需要在2-6维范围。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型支持`int8`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`和`int32`（`int32`含义同`x1`，表示`int4`类型计算）。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`和`int32`（`int32`含义同`x1`，表示`int4`类型计算）。

- **scale** (`Tensor`)：必选参数，量化缩放因子。该参数支持1维shape $(t, )$（其中$t=1$或$n$）。此外，**仅当`x1`和`x2`均为`int32`类型时**，该参数还支持2维shape $(CeilDiv(k, k\_group\_size), n)$。参数$k$和$n$与`x2`的维度一致。如需传入`int64`数据类型的`scale`，需要提前调用`torch_npu.npu_trans_quant_param`来获取`int64`数据类型的`scale`。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型支持`float32`、`int64`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`float32`、`int64`、`bfloat16`。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`float32`、`int64`、`bfloat16`。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **offset** (`Tensor`)：仅当`scale`为2维时为必选参数，并且数据类型仅支持`float16`，shape需要是2维且与`scale`相同；其他场景下为可选参数，用于调整量化后的数值偏移量。数据类型支持`float32`，数据格式支持$ND$，shape需要是1维$(t,)$，$t=1$或$n$，其中$n$与`x2`的$n$一致。
- **pertoken_scale** (`Tensor`)：仅当`scale`为2维时为必选参数，shape需要是2维$(m,1)$，其中$m$与`x1`的$m$一致；其他场景为可选参数，用于缩放原数值以匹配量化后的范围值。数据类型支持`float32`，数据格式支持$ND$，shape需要是1维$(m,)$，其中$m$与`x1`的$m$一致。<term>Atlas 推理系列加速卡产品</term>当前不支持`pertoken_scale`。
- **bias** (`Tensor`)：可选参数，偏置项，数据格式支持$ND$，shape支持1维$(n,)$或3维$（batch, 1, n）$，$n$与`x2`的$n$一致，同时$batch$值需要等于`x1`和`x2` broadcast后推导出的$batch$值。当输出是2、4、5、6维情况下，`bias`的shape必须为1维。当输出是3维情况下，`bias`的shape可以为1维或3维。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型支持`int32`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int32`、`bfloat16`、`float16`、`float32`。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int32`、`bfloat16`、`float16`、`float32`。

- **output_dtype** (`int`)：可选参数，表示输出Tensor的数据类型。默认值为`None`，代表输出Tensor数据类型为`int8`。
    - <term>Atlas 推理系列加速卡产品</term>：数据类型支持`int8`、`float16`。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型支持`int8`、`float16`、`bfloat16`、`int32`。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持`int8`、`float16`、`bfloat16`、`int32`。

- **x1_dtype** (`int`)：可选参数，指定输入`x1` 的数据类型。

- **x2_dtype** (`int`)：可选参数，指定输入`x2`的数据类型。

- **pertoken_scale_dtype** (`int`)：可选参数，指定输入`pertoken_scale`的数据类型。

- **scale_dtype** (`int`)：可选参数，指定输入`scale`的数据类型。

- **group_sizes** (`list[int]`)：可选参数，表示分组量化粒度。

## 返回值说明

`Tensor`

代表量化matmul的计算结果。
- 如果`output_dtype`为`float16`，输出的数据类型为`float16`。
- 如果`output_dtype`为`int8`或者`None`，输出的数据类型为`int8`。
- 如果`output_dtype`为`bfloat16`，输出的数据类型为`bfloat16`。
- 如果`output_dtype`为`int32`，输出的数据类型为`int32`。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 传入的`x1`、`x2`、`scale`不能是空。
- `x1`、`x2`、`bias`、`scale`、`offset`、`pertoken_scale`、`output_dtype`的数据类型和数据格式需要在支持的范围之内。
- `x1`与`x2`最后一维的shape大小不能超过65535。
- 目前输出`int8`或`float16`且无`pertoken_scale`情况下，图模式不支持`scale`直接传入`float32`数据类型。
- 如果在PyTorch图模式中使用本接口，且环境变量`ENABLE_ACLNN=false`，则在调用接口前需要对shape为$(n, k//8)$的`x2`数据进行转置，转置过程应写在图中。
- 支持将`x2`转为昇腾亲和的数据排布以提高搬运效率。需要调用`torch_npu.npu_format_cast`完成输入`x2`（weight）为昇腾亲和的数据排布功能。
    - <term>Atlas 推理系列加速卡产品</term>：必须先将`x2`转置后再转亲和format。
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：推荐`x2`不转置直接转亲和format。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：推荐`x2`不转置直接转亲和format。

- **int4** 类型计算的额外约束：

    当`x1`、`x2`的数据类型均为`int32`，每个`int32`类型的数据存放8个`int4`数据。输入的`int32` shape需要将数据原本`int4`类型时shape的最后一维缩小8倍。`int4`数据的shape最后一维应为8的倍数，例如：进行$(m, k)$乘$(k, n)$的`int4`类型矩阵乘计算时，需要输入`int32`类型、shape为$(m, k//8)$、$(k, n//8)$的数据，其中$k$与$n$都应是8的倍数。`x1`只能接受shape为$(m, k//8)$且数据排布连续的数据，`x2`可以接受$(k, n[g1] //8)$且数据排布连续的数据或shape为$(k//8, n)$且是由数据连续排布的$(n, k//8)$转置而来的数据。

    > [!NOTE]  
    > 数据排布连续是指数组中所有相邻的数，包括换行时内存地址连续，使用`Tensor.is_contiguous`返回值为`True`则表明Tensor数据排布连续。

- 输入参数间支持的数据类型组合情况如下：

    **表 1** <term>Atlas 推理系列加速卡产品</term>
    |x1|x2|scale|offset|bias|pertoken_scale|output_dtype|
    |---------|--------|--------|--------|--------|--------|--------|
    |int8|int8|int64/float32|None|int32/None|None|float16|
    |int8|int8|int64/float32|float32/None|int32/None|None|int8|

    **表 2**  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

    |x1|x2|scale|offset|bias|pertoken_scale|output_dtype|
    |---------|--------|--------|--------|--------|--------|--------|
    |int8|int8|int64/float32|None|int32/None|None|float16|
    |int8|int8|int64/float32|float32/None|int32/None|None|int8|
    |int8|int8|float32/bfloat16|None|int32/bfloat16/float32/None|float32/None|bfloat16|
    |int8|int8|float32|None|int32/bfloat16/float32/None|float32|float16|
    |int32|int32|int64/float32|None|int32/None|None|float16|
    |int32|int32|float32|float16|None|float32|bfloat16/float16|
    |int8|int8|float32/bfloat16|None|int32/None|None|int32|


## 调用示例

- 单算子调用
    - `int8`类型输入场景：

        ```python
        >>> import torch
        >>> import torch_npu
        >>> import logging
        >>> import os
        >>>
        >>> cpu_x1 = torch.randint(-5, 5, (1, 256, 768), dtype=torch.int8)
        >>> cpu_x2 = torch.randint(-5, 5, (31, 768, 16), dtype=torch.int8)
        >>> scale = torch.randn(16, dtype=torch.float32)
        >>> offset = torch.randn(16, dtype=torch.float32)
        >>> bias = torch.randint(-5, 5, (31, 1, 16), dtype=torch.int32)
        >>> # Method 1：You can directly call npu_quant_matmul
        >>> npu_out = torch_npu.npu_quant_matmul(
        ...     cpu_x1.npu(), cpu_x2.npu(), scale.npu(), offset=offset.npu(), bias=bias.npu()
        ... )
        >>> npu_out
        tensor([[[  75, -128,   -7,  ...,   30, -128,  -27],
                [-128, -128,  -98,  ...,   -1, -128, -102],
                [-128,  127, -128,  ...,   32,  -12,  -11],
                ...,
                [  22,  119, -102,  ...,   57, -128,  -50],
                [-128,  127, -128,  ...,  -27, -128,  -18],
                [-128, -128,  114,  ...,    1,   39,  -16]],
                ...,
                [[-128, -128, -128,  ...,   -3,  -13,  -47],
                [-128, -117,  -35,  ...,   34,  127,   18],
                [ 127,  127,  -18,  ...,   30, -128,  -47],
                ...,
                [-128, -128, -128,  ...,   39, -104,   -6],
                [-128,  127,   55,  ...,    8,   -5,   17],
                [ 127, -128, -128,  ...,    4, -128,   -5]]], device='npu:0',
            dtype=torch.int8)
        >>>
        >>> # Method 2: You can first call npu_trans_quant_param to convert scale and offset from float32 to int64
        >>> # when output dtype is not torch.bfloat16 and pertoken_scale is none
        >>> scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu())
        >>> npu_out = torch_npu.npu_quant_matmul(
        ...     cpu_x1.npu(), cpu_x2.npu(), scale_1, bias=bias.npu()
        ... )
        >>> npu_out
        tensor([[[  75, -128,   -7,  ...,   30, -128,  -27],
                [-128, -128,  -98,  ...,   -1, -128, -102],
                [-128,  127, -128,  ...,   32,  -12,  -11],
                ...,
                [  22,  119, -102,  ...,   57, -128,  -50],
                [-128,  127, -128,  ...,  -27, -128,  -18],
                [-128, -128,  114,  ...,    1,   39,  -16]],
                ...,
                [[-128, -128, -128,  ...,   -3,  -13,  -47],
                [-128, -117,  -35,  ...,   34,  127,   18],
                [ 127,  127,  -18,  ...,   30, -128,  -47],
                ...,
                [-128, -128, -128,  ...,   39, -104,   -6],
                [-128,  127,   55,  ...,    8,   -5,   17],
                [ 127, -128, -128,  ...,    4, -128,   -5]]], device='npu:0',
            dtype=torch.int8)
        ```

- 图模式调用（ND数据格式）
    - 输出`float16`

        ```python
        import torch
        import torch_npu
        import torchair as tng
        from torchair.ge_concrete_graph import ge_apis as ge
        from torchair.configs.compiler_config import CompilerConfig
        import logging
        from torchair.core.utils import logger

        logger.setLevel(logging.DEBUG)
        import os
        import numpy as np

        # "ENABLE_ACLNN"是否使能走aclnn, true: 回调走aclnn, false: 在线编译
        os.environ["ENABLE_ACLNN"] = "true"
        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
    
            def forward(self, x1, x2, scale, offset, bias):
                return torch_npu.npu_quant_matmul(
                    x1, x2, scale, offset=offset, bias=bias, output_dtype=torch.float16
                )

        cpu_model = MyModel()
        model = cpu_model.npu()
        cpu_x1 = torch.randint(-1, 1, (15, 1, 512), dtype=torch.int8)
        cpu_x2 = torch.randint(-1, 1, (15, 512, 128), dtype=torch.int8)
        scale = torch.randn(1, dtype=torch.float32)
        # pertoken_scale为空时，输出fp16必须先调用npu_trans_quant_param，将scale(offset)从float转为int64.
        scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), None)
        bias = torch.randint(-1, 1, (15, 1, 128), dtype=torch.int32)
        # dynamic=True: 动态图模式， dynamic=False: 静态图模式
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale_1, None, bias.npu())
        print(npu_out.shape)
        print(npu_out)
    
        # 执行上述代码的输出类似如下
        torch.Size([15, 1, 128])
        [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
        tensor([[[-103.6875, -104.5000, -113.6250,  ..., -108.6875,  -99.5625,
                -101.1875]],
    
                [[ -92.9375,  -90.4375, -110.3125,  ..., -106.1875, -105.3750,
                -98.7500]],
    
                [[-102.8750,  -98.7500, -104.5000,  ..., -106.1875, -117.8125,
                -111.1875]],
    
                ...,
    
                [[-107.0000,  -92.9375, -113.6250,  ..., -107.8750,  -99.5625,
                -103.6875]],
    
                [[-117.0000, -115.3125, -120.3125,  ..., -126.1250, -109.5000,
                -103.6875]],
    
                [[-122.7500, -107.8750, -129.3750,  ..., -115.3125, -106.1875,
                -112.8125]]], device='npu:0', dtype=torch.float16)
        ```
    
    - 输出`bfloat16`，示例代码如下，仅支持如下产品：
    
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    
        ```python
        import torch
        import torch_npu
        import torchair as tng
        from torchair.ge_concrete_graph import ge_apis as ge
        from torchair.configs.compiler_config import CompilerConfig
        import logging
        from torchair.core.utils import logger
    
        logger.setLevel(logging.DEBUG)
        import os
        import numpy as np
    
        os.environ["ENABLE_ACLNN"] = "true"
        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
    
            def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
                return torch_npu.npu_quant_matmul(
                    x1,
                    x2.t(),
                    scale,
                    offset=offset,
                    bias=bias,
                    pertoken_scale=pertoken_scale,
                    output_dtype=torch.bfloat16,
                )

        cpu_model = MyModel()
        model = cpu_model.npu()
        m = 15
        k = 11264
        n = 6912
        bias_flag = True
        cpu_x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
        cpu_x2 = torch.randint(-1, 1, (n, k), dtype=torch.int8)
        scale = torch.randn((n,), dtype=torch.bfloat16)
        pertoken_scale = torch.randn((m,), dtype=torch.float32)
        bias = torch.randn((n,), dtype=torch.bfloat16)
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        if bias_flag:
            npu_out = model(
                cpu_x1.npu(), cpu_x2.npu(), scale.npu(), None, bias.npu(), pertoken_scale.npu()
            )
        else:
            npu_out = model(
                cpu_x1.npu(), cpu_x2.npu(), scale.npu(), None, None, pertoken_scale.npu()
            )
        print(npu_out.shape)
        print(npu_out)
    
        # 执行上述代码的输出类似如下
        torch.Size([15, 6912])
        [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
        tensor([[-1.0000e+00,  0.0000e+00, -1.0000e+00,  ...,  0.0000e+00,
                -1.0000e+00, -1.0000e+00],
                [ 2.8480e+03,  2.7840e+03, -1.0000e+00,  ...,  2.7840e+03,
                2.8160e+03,  2.8800e+03],
                [ 2.8320e+03,  2.8160e+03, -1.0000e+00,  ...,  2.8000e+03,
                2.8320e+03,  2.7840e+03],
                ...,
                [ 2.8800e+03,  2.8160e+03, -1.0000e+00,  ...,  2.8480e+03,
                2.9120e+03,  2.8480e+03],
                [-1.0000e+00,  0.0000e+00, -1.0000e+00,  ...,  0.0000e+00,
                -1.0000e+00, -1.0000e+00],
                [ 2.8320e+03,  2.8000e+03, -1.0000e+00,  ...,  2.7680e+03,
                2.8320e+03,  2.8640e+03]], device='npu:0', dtype=torch.bfloat16)
        ```

- 图模式调用（高性能数据排布方式）
    - 将x2转置$(batch,** n, k**)$后转format，示例代码如下，仅支持<term>Atlas 推理系列加速卡产品</term>。

        ```python
        import torch
        import torch_npu
        import torchair as tng
        from torchair.ge_concrete_graph import ge_apis as ge
        from torchair.configs.compiler_config import CompilerConfig
        import logging
        from torchair.core.utils import logger

        logger.setLevel(logging.DEBUG)
        import os
        import numpy as np

        os.environ["ENABLE_ACLNN"] = "true"
        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)
        
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
    
            def forward(self, x1, x2, scale, offset, bias):
                return torch_npu.npu_quant_matmul(
                    x1, x2.transpose(2, 1), scale, offset=offset, bias=bias
                )

        cpu_model = MyModel()
        model = cpu_model.npu()
        cpu_x1 = torch.randint(-1, 1, (15, 1, 512), dtype=torch.int8).npu()
        cpu_x2 = torch.randint(-1, 1, (15, 512, 128), dtype=torch.int8).npu()
        # Process x2 into a high-bandwidth format(29) offline to improve performance, please ensure that the input is continuous with (batch,n,k) layout
        cpu_x2_t_29 = torch_npu.npu_format_cast(cpu_x2.transpose(2, 1).contiguous(), 29)
        scale = torch.randn(1, dtype=torch.float32).npu()
        offset = torch.randn(1, dtype=torch.float32).npu()
        bias = torch.randint(-1, 1, (128,), dtype=torch.int32).npu()
        # Process scale from float32 to int64 offline to improve performance
        scale_1 = torch_npu.npu_trans_quant_param(scale, offset)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        npu_out = model(cpu_x1, cpu_x2_t_29, scale_1, offset, bias)
        print(npu_out.shape)
        print(npu_out)
    
        # 执行上述代码的输出类似如下
        torch.Size([15, 1, 128])
        tensor([[[110, 105,  96,  ...,  99, 108, 112]],
    
                [[103, 106, 103,  ..., 102,  99,  97]],
    
                [[107, 110, 100,  ..., 112, 116, 110]],
    
                ...,
    
                [[110, 101, 108,  ..., 101, 110, 105]],
    
                [[ 96,  95, 102,  ...,  99,  95,  99]],
    
                [[ 89, 113, 103,  ..., 101,  95, 102]]], device='npu:0',
            dtype=torch.int8)
        ```
    
    - 将x2非转置$(batch,** k, n**)$后转format，示例代码如下，仅支持如下产品：
    
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>
    
        ```python
        import torch
        import torch_npu
        import torchair as tng
        from torchair.ge_concrete_graph import ge_apis as ge
        from torchair.configs.compiler_config import CompilerConfig
        import logging
        from torchair.core.utils import logger
        logger.setLevel(logging.DEBUG)
        import os
        import numpy as np
        config = CompilerConfig()
        npu_backend = tng.get_npu_backend(compiler_config=config)
    
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x1, x2, scale, offset, bias, pertoken_scale):
                return torch_npu.npu_quant_matmul(x1, x2, scale, offset=offset, bias=bias, pertoken_scale=pertoken_scale, output_dtype=torch.bfloat16)
        cpu_model = MyModel()
        model = cpu_model.npu()
        m = 15
        k = 11264
        n = 6912
        bias_flag = True
        cpu_x1 = torch.randint(-1, 1, (m, k), dtype=torch.int8)
        cpu_x2 = torch.randint(-1, 1, (n, k), dtype=torch.int8)
        # Process x2 into a high-bandwidth format(29) offline to improve performance, please ensure that the input is continuous with (batch,k,n) layout
        x2_notranspose_29 = torch_npu.npu_format_cast(cpu_x2.npu().transpose(1,0).contiguous(), 29)
        scale = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
        pertoken_scale = torch.randint(-1,1, (m,), dtype=torch.float32)
    
        bias = torch.randint(-1,1, (n,), dtype=torch.bfloat16)
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        if bias_flag:
            npu_out = model(cpu_x1.npu(), x2_notranspose_29, scale.npu(), None, bias.npu(), pertoken_scale.npu())
        else:
            npu_out = model(cpu_x1.npu(), x2_notranspose_29, scale.npu(), None, None, pertoken_scale.npu())
        print(npu_out.shape)
        print(npu_out)
    
        # 执行上述代码的输出类似如下
        torch.Size([15, 6912])
        [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast repalce with float. (function operator())
        tensor([[ 0.0000e+00, -1.0000e+00, -1.0000e+00,  ..., -1.0000e+00,
                0.0000e+00, -1.0000e+00],
                [ 0.0000e+00, -1.0000e+00, -1.0000e+00,  ..., -1.0000e+00,
                0.0000e+00,  2.7840e+03],
                [ 0.0000e+00, -1.0000e+00, -1.0000e+00,  ..., -1.0000e+00,
                0.0000e+00,  2.7680e+03],
                ...,
                [ 0.0000e+00, -1.0000e+00, -1.0000e+00,  ..., -1.0000e+00,
                0.0000e+00,  2.7680e+03],
                [ 0.0000e+00, -1.0000e+00, -1.0000e+00,  ..., -1.0000e+00,
                0.0000e+00, -1.0000e+00],
                [ 0.0000e+00, -1.0000e+00, -1.0000e+00,  ..., -1.0000e+00,
                0.0000e+00, -1.0000e+00]], device='npu:0', dtype=torch.bfloat16)
        ```