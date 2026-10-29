# torch_npu.npu_trans_quant_param

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
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

- Function: converts the data type of the quantization parameter `scale`, storing the `torch.float32` data bitwise into a `torch.int64` value.
- Formulas:

    1. `out` is in 64-bit format and is initially 0.

    2. Convert `scale` from `torch.float32` to FP19 according to `round_mode`:
        - If `round_mode` is 0 (truncation padding mode, the default value), no R_INT rounding is performed on `scale`.
        - If `round_mode` is 1 (R_INT mode), R_INT rounding is first performed on `scale`:
        $$
        scale = Round(scale)
        $$

    3. Truncate the upper 19 bits of `scale` and store them in the lower 32 bits of `out` (bit 0-31), and set bit 46 to 1.
     
        $$
        out = out\ |\ (scale\ \&\ 0XFFFFE000)\ |\ (1\ll46)
        $$

    4. Perform subsequent computation according to the value of `offset`:
        - If `offset` does not exist, no subsequent computation is performed.
        - If `offset` exists:
            1. Process the `offset` value as an int, ranging from [-256, 255].
     
                $$
                offset = Max(Min(INT(Round(offset)),255),-256)
                $$

            2. Keep 9 bits of `offset` and store them in bits 37 to 45 of `out`.
                $$
                out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0X1FF)\ll37)
                $$

## Prototype

```python
torch_npu.npu_trans_quant_param(scale, offset=None, round_mode=0) -> Tensor
```

## Parameters

- **`scale`** (`Tensor`): Required. $scale$ in the formulas. The data type can be `torch.float32`. The data layout can be ND. This parameter can be 1D or 2D. For details about the constraints, see [Constraints](#constraints). Non-contiguous tensors and empty tensors are not supported.
- **`offset`** (`Tensor`): Optional. $offset$ in the formula. The data type can be `torch.float32`. The data layout can be ND. This parameter can be 1D or 2D. For details about the constraints, see [Constraints](#constraints). Non-contiguous tensors and empty tensors are not supported.
- **`round_mode`** (`int`): Optional. Mode for converting `scale` from `torch.float32` to FP19. The default value is `0`. Valid values:
  - 0: Truncation padding mode, which directly truncates `scale` to FP19 and then packs it.
  - 1: R_INT mode, which first rounds `scale` to FP19 according to the R_INT rule and then truncates and packs it, improving computation accuracy.

## Return Values

`Tensor`

Represents the computation result of `trans_quant_param`, $out$ in the formulas. The data type can be `torch.int64` (`torch.uint64` in graph mode). The data layout can be $ND$.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- API combination:

  <!-- npu="A3,910b,310p" id5 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Atlas inference products</term>: can be used together with `matmul` APIs (such as [torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)).
  <!-- end id5 -->
  <!-- npu="950" id6 -->
  - <term>Ascend 950PR&950DT products</term>: can be used together with both [torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md) and [torch_npu.npu_grouped_matmul](torch_npu-npu_grouped_matmul.md).
  <!-- end id6 -->

- When `offset` is not passed, the output shape is the same as the `scale` shape.
  - If the output is used as the input of a `matmul` API (such as [torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)), the shape can be 1D $(1,)$, $(n,)$, or 2D $(1, n)$, where $n$ is the same as the $n$ in the shape of the right matrix (parameter x2, `weight`) of the `matmul` computation.
  - If the output is used as the input of a `grouped matmul` API (such as [torch_npu.npu_grouped_matmul](torch_npu-npu_grouped_matmul.md)), it is used only when the grouping mode is M-axis grouping (`group_type` is 0). The shape can be 1D $(g,)$ or 2D $(g, 1)$, $(g, n)$, where $n$ is the same as the $n$ in the shape of the right matrix (parameter `weight`) of the `grouped matmul` computation, and $g$ is the same as the number of groups (the shape size of parameter `group_list`) of the `grouped matmul` computation.
- When `offset` is passed, the output is used only as the input of a `matmul` API (such as [torch_npu.npu_quant_matmul](torch_npu-npu_quant_matmul.md)):
  - The shapes of `scale`, `offset`, and `out` can be 1D $(1,)$, $(n,)$, or 2D $(1, n)$, where $n$ is the same as the $n$ in the shape of the right matrix (parameter x2, `weight`) of the `matmul` computation.
  - When the shape of the input `scale` is 1D, the shape of `out` is also 1D, and its size is the maximum of the single-dimension shape sizes of `scale` and `offset`.
  - When the shape of the input `scale` is 2D, the shapes of `scale` and `offset` must be the same, and the shape of `out` is $(1, n)$.

## Example

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
    >>> print(npu_out)
    tensor([ 70507248869376,  70509369614336,  70507209793536, 140463653937152,
            140603250524160, 140603257561088, 140603230814208,  70369813069824,
            70369794605056, 140463675252736,  70784266256384,  70507233009664,
            140601114345472,  70371966238720, 140603258257408, 140603254505472],
        device='npu:0')
    >>> print(npu_out.dtype)
    torch.int64
    >>> print(npu_out.shape)
    torch.Size([16])
    ```

- Graph mode call

    In graph mode, the result tensor computed by `npu_trans_quant_param` has the `torch.uint64` data type. Because torch does not support this data type, it must be used together with other APIs, such as `npu_quant_matmul` in the example code.

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
    model = torch.compile(cpu_model, backend=npu_backend, dynamic=True)
    
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
