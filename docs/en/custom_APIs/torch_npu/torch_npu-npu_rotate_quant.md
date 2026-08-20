# torch_npu.npu_rotate_quant

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>       |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>       |    √     |

## Function

Description: Implements a fused computation method that combines rotation and quantization. This method is applicable to scenarios where input data needs to be rotated and then quantized. The fused operator can parallelize some processes at the bottom layer to optimize performance.

## Prototype

```python
torch_npu.npu_rotate_quant(x, rotation, *, alpha=0.0, dst_dtype=None) -> (Tensor, Tensor)
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor. This parameter must be 2D with shape `[m, n]`. The data type can be `bfloat16` or `float16`. The data layout can be ND. Non-contiguous tensors are supported.
- **`rotation`** (`Tensor`): Required. Rotation matrix tensor. This parameter must be 2D with shape `[k, k]`. The data type can be `bfloat16` or `float16`. The data layout can be ND. Non-contiguous tensors are supported.
- **`alpha`** (`float`): Optional. Scaling factor for the rotation angle. The data type can be `float`. The default value is `0.0`.
- **`dst_dtype`** (`int`): Optional. Data type of the quantization output. Processed as `torch.int8` if `None` is passed.

## Return Values

- **`y`** (`Tensor`): Output quantization result. This parameter must be 2D with shape `[m, n]` and the shape must be identical to that of `x`. The data type can be `int4` or `int8`. The data layout can be ND. Non-contiguous tensors are supported.
- **`scale`** (`Tensor`): Output quantization factor. This parameter must be 1D with shape `[m]`. The data type can be `float32`. The data layout can be ND. Non-contiguous tensors are supported.

## Constraints

- This API can be used in both inference and training scenarios.
- This API supports graph mode.
- `n` must be in the range of `128` to `16000`, aligned to 8 bytes, and divisible by `k`.
- The following table lists the data type combinations supported by the input and output tensors.

    |x|rotation|dst_dtype|y|scale|
    |--------|--------|--------|--------|--------|
    |`bfloat16`|`bfloat16`|torch.qint8|`int8`|`float32`|
    |`bfloat16`|`bfloat16`|torch.quint4X2|`int4`|`float32`|
    |`float16`|`float16`|torch.qint8|`int8`|`float32`|
    |`float16`|`float16`|torch.quint4X2|`int4`|`float32`|

## Example

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
    K = 1024
    x, rotation = gen_input_data(M, N, K)
    output0_npu, output1_npu = torch_npu.npu_rotate_quant(x.npu(), rotation.npu(), alpha=0.0, dst_dtype=torch.int8)
    ```

- Graph mode call:

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
            output = torch_npu.npu_rotate_quant(x, rotation, alpha=0.0, dst_dtype=torch.int8)
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
