# torch_npu.npu_ffn

## 功能说明

- 算子功能：该FFN算子提供MoeFFN和FFN的计算功能。在没有专家分组（`expert_tokens`为空）时是FFN，有专家分组时是MoeFFN。
- 计算公式：

    ![](figures/zh-cn_formulaimage_0000001850668725.png)

>**说明：**<br>
>激活层为geglu/swiglu/reglu时，性能使能需要满足门槛要求，即整网中FFN结构所对应的小算子中vector耗时30us且占比10%以上的用例方可尝试FFN融合算子；或在不知道小算子性能的情况下，尝试使能FFN，若性能劣化则不使能FFN。

## 函数原型

```
torch_npu.npu_ffn(x, weight1, weight2, activation, *, expert_tokens=None, expert_tokens_index=None, bias1=None, bias2=None, scale=None, offset=None, deq_scale1=None, deq_scale2=None, antiquant_scale1=None, antiquant_scale2=None, antiquant_offset1=None, antiquant_offset2=None, inner_precise=None, output_dtype=None) -> Tensor
```

## 参数说明

- **x** (`Tensor`)：输入参数，公式中的$x$，数据类型支持`float16`、`bfloat16`、`int8`，数据格式支持$ND$，支持输入的维度最少是2维$[M, K1]$，最多是8维。

- **weight1** (`Tensor`)：专家的权重数据，公式中的$W1$，数据类型支持`float16`、`bfloat16`、`int8`，数据格式支持$ND$，输入在有/无专家时分别为$[E, K1, N1]/[K1, N1]$。

- **weight2** (`Tensor`)：专家的权重数据，公式中的$W2$，数据类型支持`float16`、`bfloat16`、`int8`，数据格式支持$ND$，输入在有/无专家时分别为$[E, K2, N2]/[K2, N2]$。

    >**说明：**<br>
    >$M$表示token个数，对应transform中的BS（$B$：Batch，表示输入样本批量大小，$S$：Seq-Length，表示输入样本序列长度）；$K1$表示第一个matmul的输入通道数，对应transform中的$H$（Head-Size，表示隐藏层的大小）；$N1$表示第一个matmul的输出通道数；$K2$表示第二个matmul的输入通道数；$N2$表示第二个matmul的输出通道数，对应transform中的$H$；$E$表示有专家场景的专家数。

- **activation** (`str`)：代表使用的激活函数，即输入参数中的`activation`。当前仅支持`fastgelu、gelu、relu、silu、geglu、swiglu、reglu`。
- **expert_tokens** (`list`)：可选参数。代表各专家的token数，数据类型支持`int32`，数据格式支持$ND$，若不为空时可支持的最大长度为256个。
- **expert_tokens_index** (`list`)：可选参数。代表各专家计算token的索引值，数据类型支持`int32`，数据格式支持$ND$，若不为空时可支持的最大长度为256个。

- **bias1** (`Tensor`)：可选参数。权重数据修正值，公式中的$b1$，数据类型支持`float16`、`float32`、`int32`，数据格式支持$ND$，输入在有/无专家时分别为$[E, N1]/[N1]$。
- **bias2** (`Tensor`)：可选参数。权重数据修正值，公式中的$b2$，数据类型支持`float16`、`float32`、`int32`，数据格式支持$ND$，输入在有/无专家时分别为$[E, N2]/[N2]$。

- **scale** (`Tensor`)：可选参数，量化参数，量化缩放系数，数据类型支持`float32`，数据格式支持$ND$。per-tensor下输入在有/无专家时均为一维向量，输入元素个数在有/无专家时分别为$[E]/[1]$；per-channel下输入在有/无专家时为二维向量/一维向量，输入元素个数在有/无专家时分别为$[E, N1]/[N1]$。
- **offset** (`Tensor`)：可选参数，量化参数，量化偏移量，数据类型支持`float32`，数据格式支持$ND$，一维向量，输入元素个数在有/无专家时分别为$[E]/[1]$。
- **deq_scale1** (`Tensor`)：可选参数，量化参数，第一组matmul的反量化缩放系数，数据类型支持`int64`、`float32`、`bfloat16`，数据格式支持$ND$，输入在有/无专家时分别为$[E, N1]/[N1]$。
- **deq_scale2** (`Tensor`)：可选参数，量化参数，第二组matmul的反量化缩放系数，数据类型支持`int64`、`float32`、`bfloat16`，数据格式支持$ND$，输入在有/无专家时分别为$[E, N2]/[N2]$。
- **antiquant_scale1** (`Tensor`)：可选参数，伪量化参数，第一组matmul的缩放系数，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，per-channel下输入在有/无专家时分别为$[E, N1]/[N1]$。
- **antiquant_scale2** (`Tensor`)：可选参数，伪量化参数，第二组matmul的缩放系数，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，per-channel下输入在有/无专家时分别为$[E, N2]/[N2]$。
- **antiquant_offset1** (`Tensor`)：可选参数，伪量化参数，第一组matmul的偏移量，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，per-channel下输入在有/无专家时分别为$[E, N1]/[N1]$。
- **antiquant_offset2** (`Tensor`)：可选参数，伪量化参数，第二组matmul的偏移量，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，per-channel下输入在有/无专家时分别为$[E, N2]/[N2]$。

- **inner_precise** (`int`)：可选参数，表示高精度或者高性能选择。数据类型支持`int64`。该参数仅对`float16`生效，`bfloat16`和`int8`不区分高精度和高性能。

    - `inner_precise`为0时，代表开启高精度模式，算子内部采用`float32`数据类型计算。
    - `inner_precise`为1时，代表高性能模式。

  `inner_precise`参数在`bfloat16`非量化场景，只能配置为0；`float16`非量化场景，可以配置为0或者1；量化或者伪量化场景，0和1都可配置，但是配置后不生效。

- **output_dtype** (`ScalarType`)：可选参数，该参数只在量化场景生效，其他场景不生效。表示输出Tensor的数据类型，支持输入`float16`、`bfloat16`。默认值为`None`，代表输出Tensor数据类型为`float16`。

## 返回值
`Tensor`

一个Tensor类型的输出，公式中的输出$y$，数据类型支持`float16`、`bfloat16`，数据格式支持$ND$，输出维度与`x`一致。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式（PyTorch 2.1版本）。
- 有专家时，专家数据的总数需要与`x`的$M$保持一致。
- 激活层为`geglu/swiglu/reglu`时，仅支持无专家分组时的`float16`高性能场景（`float16`场景指类型为Tensor的必选参数数据类型都为`float16`的场景），且$N1=2*K2$。
- 激活层为`gelu/fastgelu/relu/silu`时，支持有专家或无专家分组的`float16`高精度及高性能场景，`bfloat16`场景，量化场景及伪量化场景，且$N1=K2$。
- 所有场景下需满足$K1=N2、K1<65536、K2<65536、M$轴在32Byte对齐后小于`int32`的最大值。
- 非量化场景不能输入量化参数和伪量化参数，量化场景不能输入伪量化参数，伪量化场景不能输入量化参数。
- 量化场景参数类型：`x`为`int8`、`weight`为`int8`、`bias`为`int32`、`scale`为`float32`、`offset`为`float32`，其余参数类型根据`y`不同分两种情况：
    - `y`为`float16`，`deq_scale`支持数据类型`uint64`、`int64`、`float32`。
    - `y`为`bfloat16`，`deq_scale`支持数据类型`bfloat16`。
    - 要求`deq_scale1`与`deq_scale2`的数据类型保持一致。

- 量化场景支持`scale`的per-channel模式参数类型：`x`为`int8`、`weight`为`int8`、`bias`为`int32`、`scale`为`float32`、`offset`为`float32`，其余参数类型根据`y`不同分两种情况：
    - `y`为`float16`，`deq_scale`支持数据类型`uint64`、`int64`。
    - `y`为`bfloat16`，`deq_scale`支持数据类型`bfloat16`。
    - 要求`deq_scale1`与`deq_scale2`的数据类型保持一致。

- 伪量化场景支持两种不同参数类型：
    - `y`为`float16`、`x`为`float16`、`bias`为`float16`、`antiquant_scale`为`float16`、`antiquant_offset`为`float16`、`weight`支持数据类型`int8`。
    - `y`为`bfloat16`、`x`为`bfloat16`、`bias`为`float32`、`antiquant_scale`为`bfloat16`、`antiquant_offset`为`bfloat16`、`weight`支持数据类型`int8`。

## 支持的型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> 
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> 

## 调用示例

- 单算子模式调用

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> cpu_x = torch.randn((1, 1280), device='npu', dtype=torch.float16)
    >>> cpu_weight1 = torch.randn(1280, 10240, device='npu', dtype=torch.float16)
    >>> cpu_weight2 = torch.randn(10240, 1280, device='npu', dtype=torch.float16)
    >>> activation = "fastgelu"
    >>> npu_out = torch_npu.npu_ffn(cpu_x.npu(), cpu_weight1.npu(), cpu_weight2.npu(), activation, inner_precise=1)
    >>>
    >>> npu_out
    tensor([[ 1474.0000,  2000.0000,  1683.0000,  ...,  1938.0000, -1353.0000,
            207.8750]], device='npu:0', dtype=torch.float16)
    >>>
    >>> npu_out.shape
    torch.Size([1, 1280])
    ```

- 图模式调用

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

    os.environ["ENABLE_ACLNN"] = "true"
    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, weight1, weight2, activation, expert):
            return torch_npu.npu_ffn(x, weight1, weight2, activation,  expert_tokens=expert, inner_precise=1)

    cpu_model = MyModel()
    cpu_x = torch.randn((1954, 2560),device='npu',dtype=torch.float16)
    cpu_weight1 = torch.randn((16, 2560, 5120),device='npu',dtype=torch.float16)
    cpu_weight2 = torch.randn((16, 5120, 2560),device='npu',dtype=torch.float16)
    activation = "fastgelu"
    expert = [227, 62, 78, 126, 178, 27, 122, 1, 19, 182, 166, 118, 66, 217, 122, 243]
    model = cpu_model.npu()
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)

    npu_out = model(cpu_x.npu(), cpu_weight1.npu(), cpu_weight2.npu(), activation, expert)
    print(npu_out.shape)
    print(npu_out)

    # 执行上述代码的输出类似如下
    torch.Size([1954, 2560])
    tensor([[  736.5000,  2558.0000,  3806.0000,  ..., -4180.0000,  -707.5000,
            1692.0000],
            [  113.0000,  1471.0000,  2492.0000,  ...,   404.5000, -1629.0000,
            -881.0000],
            [-3046.0000,  -401.0000,  3780.0000,  ...,  -518.5000,  -151.1250,
            3962.0000],
            ...,
            [ 2694.0000, -4648.0000,   -23.4844,  ..., -2624.0000, -2112.0000,
            -1070.0000],
            [ -438.0000, -3500.0000,  -941.0000,  ..., -2626.0000, -3878.0000,
            -2076.0000],
            [-2194.0000, -1583.0000, -1336.0000,  ...,  3906.0000,  -222.7500,
            -58.9688]], device='npu:0', dtype=torch.float16)

    ```

