# torch_npu.npu_quant_matmul_gelu

## 产品支持情况

| 产品                                            | 是否支持 |
|-----------------------------------------------|:----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>  |  √   |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>  |  √   |

## 功能说明

- API功能：完成量化的矩阵乘和GELU激活函数的融合计算，支持A8W8和A4W4量化。该接口融合了量化矩阵乘和GELU激活，减少内存访问，提升性能。

- 计算公式：

  - x1Scale， bias int32（此场景无offset）：

    $$
    qbmmout = (x1@x2 + bias) * x2Scale * x1Scale
    $$

  - x1Scale， bias BFLOAT16/FLOAT16/FLOAT32（此场景无offset）：

    $$
    qbmmout = x1@x2 * x2scale * x1Scale + bias
    $$

  - x1Scale无bias：

    $$
    qbmmout = x1@x2 * x2Scale * x1Scale
    $$

  - gelu类型由approximate输入指定，默认为"gelu_erf"，支持如下：

    - gelu_tanh运算：

      $$
      out = gelu\_tanh(qbmmout)
      $$

    - gelu_erf运算：

      $$
      out = gelu\_erf(qbmmout)
      $$

## 函数原型

```
npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, *, bias=None, approximate="gelu_erf") -> Tensor
```

## 参数说明

- **x1** (`Tensor`)：必选参数，输入张量，表示矩阵乘法中的左矩阵（激活值），数据格式支持$ND$，shape需要在2-6维范围。
    - 数据类型支持`int8`（A8W8量化）、`int32`（A4W4量化，每个`int32`数据存放8个`int4`数据）和`int4`（A4W4量化，直接int4类型）。

- **x2** (`Tensor`)：必选参数，输入张量，表示矩阵乘法中的右矩阵（权重），其与`x1`的数据类型须保持一致。数据格式支持$ND$或$NZ$（昇腾亲和排布格式），shape需要在2-6维范围。
    - 数据类型支持`int8`（A8W8量化）、`int32`（A4W4量化，每个`int32`数据存放8个`int4`数据）和`int4`（A4W4量化，直接int4类型）。
    - 支持昇腾亲和的NZ数据排布格式，可通过`torch_npu.npu_format_cast`转换为NZ格式以提升性能（仅A8W8场景）。

- **x1_scale** (`Tensor`)：必选参数，`x1`的量化缩放因子，数据格式支持$ND$。
    - 数据类型支持`float32`。
    - shape需要是1维$(m,)$，其中$m$与`x1`的$m$一致。采用per-token量化方式，每个token（行）有一个独立的scale值。

- **x2_scale** (`Tensor`)：必选参数，`x2`的量化缩放因子，数据格式支持$ND$。
    - 数据类型支持`float32`、`bfloat16`。
    - shape需要是1维$(n,)$或$(1,)$，其中$n$与`x2`的$n$一致。采用per-channel量化方式，每个输出通道有一个独立的scale值，或使用per-tensor量化（shape为$(1,)$）。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

- **bias** (`Tensor`)：可选参数，默认值为`None`，偏置项，数据格式支持$ND$。
    - 数据类型支持`int32`、`float32`、`bfloat16`、`float16`。
    - A4W4量化场景下：shape仅支持1维$(n,)$，$n$与`x2`的$n$一致。
    - A8W8量化场景下：shape支持1维$(n,)$或3维$(batch, 1, n)$，$n$与`x2`的$n$一致，同时$batch$值需要等于`x1`和`x2` broadcast后推导出的$batch$值。

- **approximate** (`str`)：可选参数，默认值为`"gelu_erf"`。指定GELU激活函数的类型。
    - 支持`"gelu_tanh"`（GELU的tanh近似版本）和`"gelu_erf"`（GELU的erf精确版本）。

## 返回值说明

`Tensor`

代表量化矩阵乘融合GELU激活的计算结果。
- 输出数据类型的确定规则：
  - 如果`x2_scale`的数据类型为`float32`，输出的数据类型为`float16`。
  - 如果`x2_scale`的数据类型为`bfloat16`，输出的数据类型为`bfloat16`。
  - 如果`bias`的数据类型为`bfloat16`，输出的数据类型强制为`bfloat16`（优先级高于`x2_scale`）。
- 输出shape为$(batch, m, n)$，其中$batch$根据`x1`和`x2`的batch通过broadcast得到。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持图模式。
- 传入的`x1`、`x2`、`x1_scale`、`x2_scale`不能是空。
- `x1`、`x2`、`bias`、`x1_scale`、`x2_scale`的数据类型和数据格式需要在支持的范围之内。
- `x1`与`x2`最后一维的shape大小不能超过65535。
- `approximate`必须为`"gelu_tanh"`或`"gelu_erf"`。

- **A4W4量化（int4/int32类型输入）的额外约束**：

    A4W4量化场景支持两种输入类型：
    - **int4类型**：直接使用int4数据类型
    - **int32类型**：每个`int32`数据存放8个`int4`数据（打包存储）
    
    当使用`int32`类型时，输入的`int32` shape需要将数据原本`int4`类型时shape的最后一维缩小8倍。`int4`数据的shape最后一维应为8的倍数。

    - `x1`和`x2`的内轴（k轴）必须为偶数。
    - 当`x2`为`int32`类型时，`x2`的shape为$(k, n//8)$，$n$必须是8的倍数。
    - 当`x2`为`int4`类型时，`x2`的shape为$(k, n)$，$n$必须是8的倍数。
    - A4W4量化仅支持ND格式，不支持NZ格式。
    - 转置信息由算子内部根据tensor的stride自动推导，无需手动指定。

- **A8W8量化的约束**：

    - 支持ND格式和NZ格式。
    - 如果需要使用NZ格式以提升性能，可以手动调用`torch_npu.npu_format_cast`完成输入`x2`（weight）的NZ格式转换。
    - 转置信息由算子内部根据tensor的stride自动推导，无需手动指定。

- 输入参数间支持的数据类型组合情况如下：

    **表 1** <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>

    | x1    | x2    | x1_scale | x2_scale  | bias                                | 输出数据类型    |
    |-------|-------|----------|-----------|-------------------------------------|-----------|
    | int8  | int8  | float32  | float32   | int32/float32/bfloat16/float16/None | float16   |
    | int8  | int8  | float32  | bfloat16  | int32/float32/bfloat16/float16/None | bfloat16  |
    | int32 | int32 | float32  | float32   | int32/None                          | float16   |
    | int32 | int32 | float32  | bfloat16  | int32/None                          | bfloat16  |
    | int4  | int4  | float32  | float32   | int32/None                          | float16   |
    | int4  | int4  | float32  | bfloat16  | int32/None                          | bfloat16  |

## 调用示例

- 单算子调用（A8W8量化，ND格式，gelu_tanh激活）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- 单算子调用（A8W8量化，ND格式，gelu_erf激活，带bias）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>> bias = torch.randn(n, dtype=torch.float32).npu() * 0.1
    >>>
    >>> # 使用gelu_erf激活并添加bias
    >>> output = torch_npu.npu_quant_matmul_gelu(
    ...     x1, x2, x1_scale, x2_scale, bias=bias, approximate="gelu_erf"
    ... )
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- 单算子调用（A8W8量化，NZ格式，gelu_tanh激活）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>>
    >>> # 将x2转换为NZ格式以提升性能
    >>> x2_nz = torch_npu.npu_format_cast(x2.contiguous(), 29)  # 29为ACL_FORMAT_FRACTAL_NZ
    >>>
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # 自动识别NZ格式并调用对应接口
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2_nz, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- 单算子调用（A8W8量化，bfloat16输出）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 64, 128, 256
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.bfloat16).abs().npu() * 0.01  # bfloat16 scale
    >>>
    >>> # 输出数据类型由x2_scale的类型决定，此处输出为bfloat16
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale, approximate="gelu_tanh")
    >>> print(output.dtype)  # torch.bfloat16
    ```

- 单算子调用（A4W4量化）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 128, 256, 512
    >>> # 生成int4数据（以int32格式存储）
    >>> # 注意：实际使用时需要通过量化接口将float32数据量化为int4并打包为int32
    >>> x1 = torch.randint(-8, 8, (m, k // 8), dtype=torch.int32).npu()
    >>> x2 = torch.randint(-8, 8, (k, n // 8), dtype=torch.int32).npu()
    >>>
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # A4W4量化仅支持ND格式，不支持NZ格式
    >>> # 转置信息由算子内部根据tensor的stride自动推导
    >>> output = torch_npu.npu_quant_matmul_gelu(
    ...     x1, x2, x1_scale, x2_scale, 
    ...     approximate="gelu_tanh"
    ... )
    >>> print(output.shape)  # torch.Size([128, 512])
    >>> print(output.dtype)  # torch.float16
    ```

- 单算子调用（使用默认approximate="gelu_erf"）

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> m, k, n = 64, 128, 256
    >>> x1 = torch.randint(-5, 5, (m, k), dtype=torch.int8).npu()
    >>> x2 = torch.randint(-5, 5, (k, n), dtype=torch.int8).npu()
    >>> x1_scale = torch.randn(m, dtype=torch.float32).abs().npu() * 0.01
    >>> x2_scale = torch.randn(n, dtype=torch.float32).abs().npu() * 0.01
    >>>
    >>> # 不指定approximate参数，使用默认值"gelu_erf"
    >>> output = torch_npu.npu_quant_matmul_gelu(x1, x2, x1_scale, x2_scale)
    >>> print(output.dtype)  # torch.float16
    ```

- 图模式调用（A8W8量化，ND格式）

    ```python
    >>> import torch
    >>> import torch_npu
    >>> import torchair as tng
    >>> from torchair.configs.compiler_config import CompilerConfig
    >>>
    >>> os.environ["ENABLE_ACLNN"] = "true"
    >>> config = CompilerConfig()
    >>> npu_backend = tng.get_npu_backend(compiler_config=config)
    >>>
    >>> class MyModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...
    ...     def forward(self, x1, x2, x1_scale, x2_scale, bias):
    ...         return torch_npu.npu_quant_matmul_gelu(
    ...             x1, x2, x1_scale, x2_scale, 
    ...             bias=bias, approximate="gelu_erf"
    ...         )
    >>>
    >>> cpu_model = MyModel()
    >>> model = cpu_model.npu()
    >>> cpu_x1 = torch.randint(-1, 1, (15, 1, 512), dtype=torch.int8)
    >>> cpu_x2 = torch.randint(-1, 1, (15, 512, 128), dtype=torch.int8)
    >>> x1_scale = torch.randn(15, dtype=torch.float32).abs() * 0.01
    >>> x2_scale = torch.randn(128, dtype=torch.float32).abs() * 0.01
    >>> bias = torch.randint(-1, 1, (15, 1, 128), dtype=torch.int32)
    >>>
    >>> model = torch.compile(model, backend=npu_backend, dynamic=True)
    >>> npu_out = model(
    ...     cpu_x1.npu(), cpu_x2.npu(), 
    ...     x1_scale.npu(), x2_scale.npu(), 
    ...     bias.npu()
    ... )
    >>> print(npu_out.shape)  # torch.Size([15, 1, 128])
    >>> print(npu_out.dtype)  # torch.float16
    ```
