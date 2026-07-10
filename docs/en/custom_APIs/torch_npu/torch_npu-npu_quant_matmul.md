# torch_npu.npu_quant_matmul

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>| √   |
|<term>Atlas inference accelerator cards</term>| √   |

## Function

- Description: Performs quantized matrix multiplication, supporting at least 2D and at most 6D input.

- Formulas:
    - Without `bias`:
    $$
    out = x1 \mathbin{@} x2 * \text{scale} + \text{offset}
    $$
    - When `bias` is `int32`:
    $$
    out = (x1 \mathbin{@} x2 + \text{bias}) * \text{scale} + \text{offset}
    $$
    - When `bias` is `bfloat16` or `float32` (without `offset`):
    $$
    out = x1 \mathbin{@} x2 * \text{scale} + \text{bias}
    $$

## Prototype

```python
torch_npu.npu_quant_matmul(x1, x2, scale, *, offset=None, pertoken_scale=None, bias=None, output_dtype=None, group_sizes=None) -> Tensor
```

## Parameters

- **`x1`** (`Tensor`): Required. Input tensor representing the left matrix in matrix multiplication. The data layout can be ND. The shape must have 2 to 6 dimensions.
    - Atlas inference accelerator cards: The data type can be `int8`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `int8` or `int32`. `int32` indicates `int4` matrix multiplication, where each `int32` element stores eight `int4` values.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `int8` or `int32`. `int32` indicates `int4` matrix multiplication, where each `int32` element stores eight `int4` values.

- **`x2`** (`Tensor`): Required. Input tensor representing the right matrix in matrix multiplication. Its data type must be identical to that of `x1`. The data layout can be ND. The shape must have 2 to 6 dimensions.
    - Atlas inference accelerator cards: The data type can be `int8`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `int8` or `int32`. The meaning of `int32` is identical to that of `x1`, which represents `int4` computation.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `int8` or `int32`. The meaning of `int32` is identical to that of `x1`, which represents `int4` computation.

- **`scale`** (`Tensor`): Required. Scaling factor used in quantization. This parameter can be 1D with shape `(t,)`, where `t = 1` or `n`. This parameter can also be 2D with shape `(CeilDiv(k, k_group_size), n)` **only when both `x1` and `x2` are of the `int32` type**. The dimensions of $k$ and $n$ are the same as those of `x2`. If an `int64` `scale` is required, call `torch_npu.npu_trans_quant_param` in advance to obtain the `int64` `scale`.
    - Atlas inference accelerator cards: The data type can be `float32` or `int64`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `float32`, `int64`, or `bfloat16`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `float32`, `int64`, or `bfloat16`.

- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`offset`** (`Tensor`): Required only when `scale` is a 2D tensor. The data type must be `float16`. This parameter must be a 2D tensor and its shape must be identical to that of `scale`. In all other scenarios, this parameter is optional and is used to adjust the quantized value offsets. The data type can be `float32`. The data layout can be ND. This parameter must be 1D with shape `(t,)`, where `t` equals `1` or `n`, and `n` must match the n-dimension of `x2`.
- **`pertoken_scale`** (`Tensor`): Required only when `scale` is 2D. This parameter must be 2D with shape `(m, 1)`, where `m` is identical to the `m` dimension of `x1`. In other scenarios, this parameter is optional and is used to scale the original values to match the quantized range. The data type can be `float32`. The data layout can be ND. This parameter must be 1D with shape `(m,)`, where `m` matches the `m` dimension of `x1`. Atlas inference accelerator cards: Currently, this parameter is not supported.
- **`bias`** (`Tensor`): Optional. Bias item. The data layout can be ND. This parameter must be1D with shape `(n,)` or 3D with shape `(batch, 1, n)`, where `n` must match the n dimension of `x2`. In addition, the `batch` value must equal the `batch` value derived after broadcasting `x1` and `x2`. In scenarios where the output has 2, 4, 5, or 6 dimensions, `bias` must be a 1D tensor. In scenarios where the output has 3 dimensions, `bias` must be a 1D or 3D tensor.
    - Atlas inference accelerator card: The data type can be `int32`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `int32`, `bfloat16`, `float16`, or `float32`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `int32`, `bfloat16`, `float16`, or `float32`.

- **`output_dtype`** (`int`): Optional. Data type of the output tensor. The default value is `None`, indicating that the data type of the output tensor is `int8`.
    - Atlas inference accelerator cards: The data type can be `int8` or `float16`.
    - Atlas A2 training products/Atlas A2 inference products: The data type can be `int8`, `float16`, `bfloat16`, or `int32`.
    - Atlas A3 training products/Atlas A3 inference products: The data type can be `int8`, `float16`, `bfloat16`, or `int32`.

- **`group_sizes`** (`list[int]`): Optional. Group quantization granularity.

## Return Values

`Tensor`

Output tensor representing the computation result of quantized matrix multiplication.

- If `output_dtype` is `"float16"`, the output data type is `float16`.
- If `output_dtype` is `"int8"` or `None`, the output data type is `int8`.
- If `output_dtype` is `"bfloat16"`, the output data type is `bfloat16`.
- If `output_dtype` is `"int32"`, the output data type is `int32`.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- The input parameters `x1`, `x2`, and `scale` must not be empty tensors.
- Data types and data layouts of `x1`, `x2`, `bias`, `scale`, `offset`, `pertoken_scale`, and `output_dtype` must be within the supported ranges.
- The size of the last dimension of `x1` and `x2` must be less than or equal to `65535`.
- Currently, when the output data type is `int8` or `float16` and `pertoken_scale` is not provided, graph capture mode does not support passing `scale` directly as a `float32` tensor.
- If this API is used in PyTorch graph capture mode and the environment variable `ENABLE_ACLNN=false` is set, `x2` with shape `(n, k // 8)` must be transposed before the API is called. The transpose operation must be included in the graph.
- Transposing `x2` into an Ascend-optimized data layout is supported to improve data transfer efficiency. Call `torch_npu.npu_format_cast` to convert the input `x2` (`weight`) into the Ascend-optimized data layout.
    - Atlas inference accelerator cards: `x2` must be transposed before being converted into the Ascend-optimized format.
    - Atlas A2 training products/Atlas A2 inference products: Converting `x2` directly into the Ascend-optimized format without transposition is recommended.
    - Atlas A3 training products/Atlas A3 inference products: Converting `x2` directly into the Ascend-optimized format without transposition is recommended.

- Additional constraints for `int4` computation:

    When the data types of both `x1` and `x2` are `int32`, each `int32` element stores eight `int4` values. For input `int32` tensors, the size of the last dimension must be one-eighth of that of the original `int4` tensors. The size of the last dimension of the original `int4` tensors must be a multiple of `8`. For example, when performing `int4` matrix multiplication with input shapes `(m, k)` and `(k, n)`, the inputs must be `int32` tensors with shapes `(m, k//8)` and `(k, n//8)`, where both `k` and `n` must be multiples of `8`. `x1` accepts only tensors with shape `(m, k // 8)` and a contiguous data layout, while `x2` accepts either a tensor with shape `(k, n // 8)` and a contiguous data layout, or a tensor with shape `(k // 8, n)` obtained by transposing a contiguous tensor with shape `(n, k // 8)`.

    > [!NOTE]  
    > A contiguous data layout means that all adjacent elements in a tensor are stored in contiguous memory locations, including across row boundaries. If `Tensor.is_contiguous()` returns `True`, the tensor layout is considered contiguous.

- The following tables describe the supported data type combinations among input parameters.

    **Table 1** Atlas inference accelerator cards

    |x1|x2|scale|offset|bias|pertoken_scale|output_dtype|
    |---------|--------|--------|--------|--------|--------|--------|
    |int8|int8|int64/float32|None|int32/None|None|float16|
    |int8|int8|int64/float32|float32/None|int32/None|None|int8|

    **Table 2** Atlas A2 training products/Atlas A2 inference products and Atlas A3 training products/Atlas A3 inference products

    |x1|x2|scale|offset|bias|pertoken_scale|output_dtype|
    |---------|--------|--------|--------|--------|--------|--------|
    |int8|int8|int64/float32|None|int32/None|None|float16|
    |int8|int8|int64/float32|float32/None|int32/None|None|int8|
    |int8|int8|float32/bfloat16|None|int32/bfloat16/float32/None|float32/None|bfloat16|
    |int8|int8|float32|None|int32/bfloat16/float32/None|float32|float16|
    |int32|int32|int64/float32|None|int32/None|None|float16|
    |int32|int32|float32|float16|None|float32|bfloat16/float16|
    |int8|int8|float32/bfloat16|None|int32/None|None|int32|

## Examples

- Single-operator call
    - Scenarios with `int8` inputs:

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
        >>> # Method 1: You can directly call npu_quant_matmul
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

- Graph mode (ND layout)
    - Scenarios with `float16` outputs:

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

        # ENABLE_ACLNN specifies whether to use ACLNN. Valid values: true (uses ACLNN execution) or false (uses online compilation).
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
        # If pertoken_scale is not specified and the output data type is float16, call npu_trans_quant_param first to convert scale (offset) from float to int64.
        scale_1 = torch_npu.npu_trans_quant_param(scale.npu(), None)
        bias = torch.randint(-1, 1, (15, 1, 128), dtype=torch.int32)
        # dynamic=True: dynamic graph mode; dynamic=False: static graph mode
        model = torch.compile(model, backend=npu_backend, dynamic=True)
        npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale_1, None, bias.npu())
        print(npu_out.shape)
        print(npu_out)
    
        # Expected output of the preceding code sample:
        torch.Size([15, 1, 128])
        [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
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
    
    - Code sample for scenarios with `bfloat16` outputs (supported on the following products):
    
        - Atlas A2 training products/Atlas A2 inference products
        - Atlas A3 training products/Atlas A3 inference products
    
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
    
        # Expected output of the preceding code sample:
        torch.Size([15, 6912])
        [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
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

- Graph mode call (high-performance data layout)
    - Transpose `x2` to shape `(batch, n, k)` before performing the format conversion. The following code sample applies only to Atlas inference accelerator cards.

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
    
        # Expected output of the preceding code sample:
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
    
    - Convert `x2` to the target format without transposition, while keeping its shape as `(batch, k, n)` before performing the format conversion. The following code sample applies only to the following products:
    
        - Atlas A2 training products/Atlas A2 inference products
        - Atlas A3 training products/Atlas A3 inference products
    
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
    
        # Expected output of the preceding code sample:
        torch.Size([15, 6912])
        [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
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
