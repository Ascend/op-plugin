# torch_npu.npu_trans_quant_param

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>    |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>  |     √    |

## Function

- Description: Converts the data type of the quantization parameter `scale` by packing the bit representation of `float32` data into an `int64` value.
- Formulas:

    1. `out` is a 64-bit value and is initialized to `0`.

    2. If `round_mode` is `1`, `scale` is rounded before truncation to the upper 19 bits. If `round_mode` is `0`, no processing is performed.
        $$
        scale = Round(scale)
        $$

    3. The upper 19 bits of `scale` are truncated and stored starting at bit 32 of `out`, and bit 46 is set to `1`.
     
        $$
        out = out\ |\ (scale\ \&\ 0XFFFFE000)\ |\ (1\ll46)
        $$

    4. The subsequent computation is performed based on the value of `offset`.

        - If `offset` is not provided, no further computation is performed.
        - If `offset` is provided:
            1. Convert the `offset` value to an integer within the range [-256, 255].
                $$
                offset = Max(Min(INT(Round(offset)),255),-256)
                $$

            2. Retain the lower 9 bits of `offset` and store them in bits 37 through 45 of `out`.
                $$
                out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0X1FF)\ll37)
                $$

## Prototype

```python
torch_npu.npu_trans_quant_param(scale, offset=None, round_mode=0) -> Tensor
```

## Parameters

- **`scale`** (`Tensor`): Required. $scale$ in the formulas. The data type can be `float32`. The data layout can be ND. This parameter can be 1D or 2D. For details about the constraints, see [Constraints](#constraints). Non-contiguous tensors and empty tensors are not supported.
- **`offset`** (`Tensor`): Optional. $offset$ in the formula. The data type can be `float32`. The data layout can be ND. This parameter can be 1D or 2D. For details about the constraints, see [Constraints](#constraints). Non-contiguous tensors and empty tensors are not supported.
- **`round_mode`** (`int`): Optional. Data type conversion mode used during quantization. The default value is `0`. Valid values are `0` (truncation mode, retaining the upper 19 bits) or `1` (`R_INT` mode, which can improve computational precision).

## Return Values

`Tensor`

The final computation result of `trans_quant_param`, $out$ in the formulas. The data type can be `int64` (`uint64` in graph mode). The data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- On currently supported products, this API can be used together with `matmul` APIs (such as [torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)).
- If `offset` is not provided, the output shape is identical to that of `scale`.
  - If this output serves as the input to a `matmul` operator (such as [torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)), this parameter can be 1D with shape `(1,)` or `(n,)`, or 2D with shape `(1, n)`, where `n` matches the size of dimension `n` of the right matrix (`weight`, corresponding to the parameter `x2`) in the `matmul` computation.
  - If the output serves as the input to a `grouped matmul` operator (such as [torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)), it can be used only when the grouping mode is M-axis grouping (with `group_type` set to `0`). This parameter can be 1D with shape `(g,)`, or 2D with shape `(g, 1)` or `(g, n)`, where `n` matches the size of dimension `n` of the right matrix (corresponding to the parameter `weight`) in the `grouped matmul` computation, and `g` matches the number of groups (corresponding to the shape size of the parameter `group_list`) in the `grouped matmul` computation.
- If `offset` is provided, it can serve only as the input to a `matmul` operator (such as [torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)):
  - `scale`, `offset`, and `out` can be 1D with shape `(1,)` or `(n,)`, or 2D with shape `(1, n)`, where `n` matches the size of dimension `n` of the right matrix (`weight`, corresponding to the parameter `x2`) in the `matmul` computation.
  - If the input `scale` is 1D, `out` is also 1D, and its shape size is the maximum of the 1D shape sizes of `scale` and `offset`.
  - If the input `scale` is 2D, `out` has exactly the same dimensions and shape as the input `scale`.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> scale = torch.randn(16, dtype=torch.float32)
    >>> offset = torch.randn(16, dtype=torch.float32)
    >>> round_mode = 1
    >>> npu_out = torch_npu.npu_trans_quant_param(scale.npu(), offset.npu(), round_mode)
    >>>
    >>> npu_out
    tensor([ 70507248869376,  70509369614336,  70507209793536, 140463653937152,
            140603250524160, 140603257561088, 140603230814208,  70369813069824,
            70369794605056, 140463675252736,  70784266256384,  70507233009664,
            140601114345472,  70371966238720, 140603258257408, 140603254505472],
        device='npu:0')
    >>> npu_out.dtype
    torch.int64
    >>> npu_out.shape
    torch.Size([16])
    ```

- Graph mode call

    In graph mode, the result tensor computed by `npu_trans_quant_param` is of the `uint64` data type. PyTorch does not support this data type. This API must be used together with other APIs, such as `npu_quant_matmul` in the following code sample.

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
        def forward(self, x1, x2, scale, offset, bias, round_mode):
            scale_1 = torch_npu.npu_trans_quant_param(scale, offset, round_mode)
            return torch_npu.npu_quant_matmul(x1, x2, scale_1, offset=offset, bias=bias)

    cpu_model = MyModel()
    model = cpu_model.npu()

    cpu_x1 = torch.randint(-1, 1, (15, 1, 512), dtype=torch.int8)
    cpu_x2 = torch.randint(-1, 1, (15, 512, 128), dtype=torch.int8)
    scale = torch.randn(1, dtype=torch.float32)
    offset = torch.randn(1, dtype=torch.float32)
    round_mode = 1
    bias = torch.randint(-1,1, (15, 1, 128), dtype=torch.int32)
    model = torch.compile(model, backend=npu_backend, dynamic=True)
    
    npu_out = model(cpu_x1.npu(), cpu_x2.npu(), scale.npu(), offset.npu(), bias.npu(), round_mode)
    print(npu_out.shape)
    print(npu_out)

    # Expected output of the preceding code sample:
    torch.Size([15, 1, 128])
    tensor([[[62, 56, 58,  ..., 63, 55, 68]],

            [[61, 57, 58,  ..., 60, 50, 53]],

            [[64, 60, 64,  ..., 63, 61, 62]],

            ...,

            [[57, 63, 57,  ..., 63, 61, 62]],

            [[61, 57, 61,  ..., 58, 60, 65]],

            [[68, 62, 63,  ..., 61, 65, 69]]], device='npu:0', dtype=torch.int8)
    ```
