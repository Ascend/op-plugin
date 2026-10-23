# torch_npu.npu_rotate_quant

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->

## Function

`npu_rotate_quant` is a fused computation method that combines rotation and quantization. It applies to scenarios where the input data needs to be rotated before quantization. The fused operator can execute part of the process in parallel at the lower layer to optimize performance.

## Prototype

```python
torch_npu.npu_rotate_quant(x, rotation, *, alpha=None, dst_dtype=None, axis=-1, round_mode="rint", scale_alg=0, dst_type_max=0.0, transpose_y=False) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input Tensor. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. Non-contiguous Tensors are supported.
- **`rotation`** (`Tensor`): Required. Rotation matrix Tensor. The shape can be 2D. The data type can be `torch.bfloat16` or `torch.float16`. The data layout can be ND. Non-contiguous Tensors are supported.
- **`alpha`** (`Tensor`): Optional. Scaling factor for the rotation angle. It must be a 1D `Tensor`. The default value is None.
- **`dst_dtype`** (`int`): Optional. Specifies the type of the quantization output. When None is passed, it is processed as `torch.int8`. The supported quantization output types are `torch.int8`, `torch.quint4x2`, `torch_npu.float4_e2m1fn_x2`, `torch.float8_e5m2`, and `torch.float8_e4m3fn`.
- **`axis`** (`int`): Optional. Specifies the axis of the quantization output. The default value is -1.
- **`round_mode`** (`str`): Optional. Specifies the rounding mode. The value can be "rint", "round", or "floor". The default value is "rint".
- **`scale_alg`** (`int`): Optional. Specifies the scale algorithm. The default value is 0.
- **`dst_type_max`** (`float`): Optional. Specifies the maximum value of the quantization output. The default value is 0.0.
- **`transpose_y`** (`bool`): Optional. Specifies whether to transpose the output. The default value is False. Only False is supported in the current version.

## Return Values

- **`y`** (`Tensor`): Quantization output. The data type is determined by `dst_dtype`. The data layout can be ND. For integer scenarios, non-contiguous Tensors are supported.
- **`scale`** (`Tensor`): Output quantization factor. The data type is determined by `dst_dtype`. When `dst_dtype` is an MX type (`torch_npu.float4_e2m1fn_x2`/`torch.float8_e5m2`/`torch.float8_e4m3fn`), `scale` is an MX-format quantization factor (`torch.uint8` representing `torch.float8_e8m0fnu`). For other types, `scale` is a pertoken quantization factor (`torch.float32`).

## Constraints

- This API supports inference and training scenarios.
<!-- npu="A3,910b" id20 -->
- This API supports graph mode only on Atlas A2 products and Atlas A3 products.
<!-- end id20 -->
<!-- npu="950" id21 -->
- This API does not support graph mode on Ascend 950PR&950DT products.
<!-- end id21 -->
- The following constraints apply to the shape of the input data:
    <!-- npu="A3,910b" id5 -->
    - On Atlas A2 products and Atlas A3 products, the shape of `rotation` supports only `[K, K]`, where the value range of `K` is [16, 1024].
    <!-- end id5 -->
    <!-- npu="950" id6 -->
    - On Ascend 950PR&950DT products, the shape of `rotation` supports `[K, K]` and `[block_num, K, K]`. `K` can be only 32, 64, or 128, where `block_num = N/K` and `N` is the size of the last dimension of the input `x`.
    <!-- end id6 -->
    - The last dimension of the input `x` must be divisible by `K`.
      <!-- npu="A3,910b" id7 -->
      - On Atlas A2 products and Atlas A3 products, the shape of `x` supports only 2D, and the size of the last dimension must be in [128, 16000] and must be divisible by 8.
      <!-- end id7 -->
      <!-- npu="950" id8 -->
      - On Ascend 950PR&950DT products, the shape of `x` supports 1 to 7 dimensions. When `dst_dtype` is `torch_npu.float4_e2m1fn_x2`, the last dimension of the input `x` must also be divisible by 2.
      <!-- end id8 -->
- The following constraints apply to the value range of the input data:   
    <!-- npu="A3,910b" id9 -->
    - On Atlas A2 products and Atlas A3 products, `dst_dtype` supports only `torch.int8` and `torch.quint4x2`.
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - On Ascend 950PR&950DT products, `dst_dtype` supports only `torch_npu.float4_e2m1fn_x2`, `torch.float8_e5m2`, and `torch.float8_e4m3fn`.
    <!-- end id10 -->
    <!-- npu="A3,910b" id11 -->
    - On Atlas A2 products and Atlas A3 products, `alpha` supports only None and has no actual function.
    <!-- end id11 -->
    <!-- npu="950" id12 -->
    - On Ascend 950PR&950DT products, the value range of `alpha` is (0.0, 1.0). If None is passed or the value is out of the valid range, no processing is performed.
    <!-- end id12 -->
    <!-- npu="A3,910b" id13 -->
    - On Atlas A2 products and Atlas A3 products, `axis` supports only -1 or 1.
    <!-- end id13 -->
    <!-- npu="950" id14 -->
    - On Ascend 950PR&950DT products, `axis` supports only -1 or D-1, where D is the number of dimensions of the input `x`.
    <!-- end id14 -->
    <!-- npu="950" id15 -->
    - On Ascend 950PR&950DT products, when `dst_dtype` is `torch.float8_e5m2` or `torch.float8_e4m3fn`, `round_mode` supports only "rint".
    <!-- end id15 -->
    <!-- npu="A3,910b" id16 -->
    - On Atlas A2 products and Atlas A3 products, `scale_alg` supports only 0.
    <!-- end id16 -->
    <!-- npu="950" id17 -->
    - On Ascend 950PR&950DT products, `scale_alg` can be 0, 1, or 2. When `dst_dtype` is `torch.float8_e5m2` or `torch.float8_e4m3fn`, only 0 and 1 are supported. When `dst_dtype` is `torch_npu.float4_e2m1fn_x2`, only 0 and 2 are supported.
    <!-- end id17 -->
    <!-- npu="A3,910b" id18 -->
    - On Atlas A2 products and Atlas A3 products, `dst_type_max` supports only 0.0.
    <!-- end id18 -->
    <!-- npu="950" id19 -->
    - On Ascend 950PR&950DT products, when `scale_alg` is 2, the value range of `dst_type_max` is [6.0, 12.0]. In other scenarios, only 0.0 is supported.
    <!-- end id19 -->
- The data types of `x` and `rotation` must be the same.

## Examples

- Single-operator call

    ```python
    import numpy as np
    import torch
    import torch_npu

    def gen_input_data(M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        return x, rotation

    M = 512
    N = 1024
    K = 128
    x, rotation = gen_input_data(M, N, K)
    # int8 quantization, only supported on Atlas A2/A3
    output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), dst_dtype=torch.int8)
    # mxfp4 quantization, only supported on Ascend 950PR&950DT products
    output0_npu, output1_npu = torch_npu.npu_rotate_quant(
        x.npu(), rotation.npu(), dst_dtype=torch_npu.float4_e2m1fn_x2, axis=-1, round_mode="rint"
    )
    ```

<!-- npu="A3,910b" id4 -->
- Graph mode call (supported only on Atlas A2 products and Atlas A3 products):

    ```python
    import numpy as np
    import torch
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, rotation):
            output = torch_npu.npu_rotate_quant(x, rotation, dst_dtype=torch.int8)
            return output

    def gen_input_data(M, N, K):
        x = torch.randn(M, N, dtype=torch.bfloat16)
        rotation = torch.randn(K, K, dtype=torch.bfloat16)
        return x, rotation

    M = 512
    N = 1024
    K = 1024
    x, rotation = gen_input_data(M, N, K)

    model = Model().npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    y = model(x.npu(), rotation.npu())
    ```
<!-- end id4 -->
