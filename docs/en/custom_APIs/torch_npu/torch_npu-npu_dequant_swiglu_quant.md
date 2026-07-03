# torch\_npu.npu_dequant\_swiglu\_quant

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Fuses dequantization, SwiGLU activation, and quantization operations on the tensor `x`, with support for grouped computation.
- Formulas:
    - `Grouping` (currently only `count` mode is supported):

        The input `x` is partitioned into groups for computation. `group_index` indicates the number of tokens in each group. Each group utilizes different quantization scales (such as `weight_scale`, `activation_scale`, and `quant_scale`).

        For example, if `x.shape = [128, 2H]` and `group_index = [2, 1, 3]`, there are three groups, and the corresponding scale dimension is `[3, 2H]`. Each group performs dequantization, SwiGLU activation, and quantization using its respective scale.

        - group0=x\[0:2, :\], scale0=scale\[0, :\]
        - group1=x\[2:3, :\], scale1=scale\[1, :\]
        - group2=x\[3:6, :\], scale2=scale\[2, :\]

    - Dequantization: Perform weight dequantization and activation dequantization:
        $$
        x=x*weight\_scale\\
        x=x*activation\_scale
        $$

    - SwiGLU activation: Apply SwiGLU to the dequantized `x` (the left and right activation paths are controlled by the attribute `activate_left`).

        - When `swiglu_mode = 0` (standard SwiGLU), using left activation as an example:
            $$
            swiglu(x)=swish(x[:,0:H])*x[:,H:2H]
            $$
            where
            $$
            swish(z)=z*sigmoid(z)
            $$

        - When `swiglu_mode = 1` (variant SwiGLU), the input `x` is split in an interleaved manner (odd-even):
            $$
            x\_{glu}=clamp(x\_{even},\ max=clamp\_limit)\\
            x\_{linear}=clamp(x\_{odd},-clamp\_limit,clamp\_limit)\\
            swiglu(x)=swish(x\_{glu},\alpha)*(x\_{linear}+bias)
            $$
            where
            $$
            swish(z,\alpha)=z*sigmoid(\alpha*z)
            $$

    - Quantization:
        1. (Optional) Perform smooth quantization first.
            $$
            out=out*quant\_scale
            $$

        2. Dynamic/Static quantization: Quantize the activated output results. Taking dynamic quantization as an example:
            $$
            out,scale=dynamicquant(out)
            $$

## Prototype

```python
torch_npu.npu_dequant_swiglu_quant(x, *, weight_scale=None, activation_scale=None, bias=None, quant_scale=None, quant_offset=None, group_index=None, activate_left=False, quant_mode=0, swiglu_mode=0, clamp_limit=7.0, float glu_alpha=1.702, float glu_bias=1.0) -> (Tensor, Tensor)
```

## Parameters

> [!NOTE]  
> Variables used in tensor shapes:
>
>- `TokensNum`: Number of transmitted tokens. The value is greater than or equal to 0.
>- `H`: Length of the embedding vector. The value is greater than 0.
>- `groupNum`: Length of the `group_index` input. The value is greater than 0.

- **`x`** (`Tensor`): Required. Target input tensor. This parameter must be 2D with shape `[TokensNum, 2H]`, and the last dimension must be an even number. The data type can be `int32` or `bfloat16`. The data layout can be ND.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`weight_scale`** (`Tensor`): Optional. Dequantization coefficient corresponding to weight quantization. This parameter must be 2D with shape `[groupNum, 2H]`. The data type can be `float32`. The data layout can be ND. When `x` is `int32`, provide `weight_scale` for dequantization.
- **`activation_scale`** (`Tensor`): Optional. Dequantization coefficient corresponding to `pertoken` weight quantization. This parameter must be 2D with shape `[TokensNum, 1]`, where the last dimension is `1` and the remaining dimensions match those of `x`. The data type can be `float32`. The data layout can be ND. When `x` is `int32`, this parameter must not be `None`, necessitating dequantization.
- **`bias`** (`Tensor`): Optional. Bias tensor for `x`. The data type can be `int32`. The data layout can be ND. When `group_index` is configured as a 2D tensor, `bias` must be `None`.
- **`quant_scale`** (`Tensor`): Optional. Smooth quantization coefficient. This parameter must be 2D with shape `[groupNum, H]`. The data type can be `float32`, `float16`, or `bfloat16`. The data layout can be ND.
  > **Note**: In static quantization, `quant_scale` supports the `float32` data type only.
- **`quant_offset`** (`Tensor`): Optional. Quantization offset. The data type can be `float32`, `float16`, or `bfloat16`. The data layout can be ND. When `group_index` is provided (non-`None`), this parameter does not take effect and must be `None`.
- **`group_index`** (`Tensor`): Optional. Number of tokens per specified group in `count` mode (values must be non-negative integers). Currently, only `count` mode is supported. This parameter must be 1D. The data type can be `int64`. The data layout can be ND.
- **`activate_left`** (`bool`): Optional. Specifies whether to apply Swish activation to the left or right half after evenly splitting the input along the last dimension. This parameter is valid only when `swiglu_mode = 0`. The default value is `False`.
    - `True`: `out=swish(split[x, -1, 2][0]) * split[x, -1, 2][1]`
    - `False`: `out=swish(split[x, -1, 2][1]) * split[x, -1, 2][0]`

- **`quant_mode`** (`int`): Optional. Quantization type specification. Valid values are `0` (enables static quantization) or `1` (enables dynamic quantization). The default value is `0`.
- **`swiglu_mode`** (`int`): Optional. SwiGLU computation mode. Valid values are `0` (enables standard SwiGLU) or `1` (enables variant SwiGLU with clamp, alpha, and bias support).
- **`clamp_limit`** (`float`): Optional. Input threshold limit for variant SwiGLU computation. The default value is `7.0`.
- **`glu_alpha`** (`float`): Optional. Activation function coefficient for GLU. The default value is `1.702`.
- **`glu_bias`** (`float`): Optional. Bias in SwiGLU computation. The default value is `1.0`.

## Return Values

- **`out`** (`Tensor`): Quantized output tensor. This parameter must be 2D with shape `[TokensNum, H]`. The data type can be `int8`. The data layout can be ND.
- **`scale`** (`Tensor`): Quantization scale. This parameter must be 1D with shape `[TokensNum]`. The data type can be `float32`. The data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports graph mode.
- In scenarios where `group_index` is provided (non-`None`), the following constraints apply:
    - Only `count` mode is supported for `group_index`. The calling network must ensure that the sum of all elements in `group_index` does not exceed the `TokensNum` dimension of `x`. Otherwise, out-of-bounds memory access will occur.
    - `H-axis size constraint`: $H \le 10496$ and must be aligned to 64. Configurations failing to meet these specifications will trigger an input validation error.
    - The portions of the output tensors `out` and `scale` that exceed the total sum of `group_index` are not cleared. These memory regions contain garbage data and may exhibit `inf` or `nan` anomalies. The network logic must account for this impact during deployment.
- When `x` is `int32`, provide `weight_scale` for dequantization.
- When `x` is of type `float16` or `bfloat16`, `weight_scale` is optional (typically `None`, but a valid tensor may also be provided), while `activation_scale` and `bias` must be `None`.
- The last dimension size of `x` must be an even number.
- When the activation dimension is not the last dimension of `x`, `group_index` must be `None`.
- When `group_index` is not `None` and dynamic quantization is enabled (`quant_mode = 1`), `bias` and `quant_offset` do not take effect.
- The `clamp_limit`, `glu_alpha`, and `glu_bias` parameters take effect only when `swiglu_mode = 1`.

## Examples

- Single-operator call

    ```python
    import os
    import shutil
    import unittest

    import torch
    import torch_npu
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SupportedDevices

    class TestNPUDequantSwigluQuant(TestCase):
        def test_npu_dequant_swiglu_quant(self, device="npu"):
            tokens_num = 4608
            hidden_size = 2048
            x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.int32)
            weight_scale = torch.randn((1, hidden_size), dtype=torch.float32)
            activation_scale = torch.randn((tokens_num, 1), dtype=torch.float32)
            quant_scale = torch.randn((1, hidden_size // 2), dtype=torch.float32)
            group_index = torch.tensor([tokens_num], dtype=torch.int64)
            bias = None
            out, scale = torch_npu.npu_dequant_swiglu_quant(
                x.npu(),
                weight_scale=weight_scale.npu(),
                activation_scale=activation_scale.npu(),
                bias=None,
                quant_scale=quant_scale.npu(),
                quant_offset=None,
                group_index=group_index.npu(),
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=7.0,
                glu_alpha=1.702,
                glu_bias=1.0
            )

    if __name__ == "__main__":
        run_tests()

    ```

- Graph mode call

    ```python
    import os
    import shutil
    import unittest

    import torch
    import torch_npu
    from torch_npu.testing.testcase import TestCase, run_tests
    from torch_npu.testing.common_utils import SupportedDevices
    from torchair.configs.compiler_config import CompilerConfig
    import torchair as tng

    class Model(torch.nn.Module):
        def forward(
            self,
            x, weight_scale, activation_scale, bias,
            quant_scale, quant_offset, group_index,
            activate_left, quant_mode, swiglu_mode, clamp_limit, glu_alpha, glu_bias
        ):
            return torch_npu.npu_dequant_swiglu_quant(
                x,
                weight_scale=weight_scale,
                activation_scale=activation_scale,
                bias=bias,
                quant_scale=quant_scale,
                quant_offset=quant_offset,
                group_index=group_index,
                activate_left=activate_left,
                quant_mode=quant_mode,
                swiglu_mode=swiglu_mode,
                clamp_limit=clamp_limit,
                glu_alpha=glu_alpha,
                glu_bias=glu_bias
            )

    class TestNPUDequantSwigluQuant(TestCase):
        def test_npu_dequant_swiglu_quant(self, device="npu"):
            tokens_num = 4608
            hidden_size = 2048
            x = torch.randint(-10, 10, (tokens_num, hidden_size), dtype=torch.int32)
            weight_scale = torch.randn((1, hidden_size), dtype=torch.float32)
            activation_scale = torch.randn((tokens_num, 1), dtype=torch.float32)
            quant_scale = torch.randn((1, hidden_size // 2), dtype=torch.float32)
            group_index = torch.tensor([tokens_num], dtype=torch.int64)
            bias = None
            quant_offset = None

            compiler_config = CompilerConfig()
            npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
            npu_mode = 1
            model = Model().npu()

            if npu_mode == 1:
                model = torch.compile(model, backend=npu_backend, dynamic=False)
            else:
                model = torch.compile(model, backend=npu_backend, dynamic=True)

            out, scale = model(
                x.npu(),
                weight_scale.npu(),
                activation_scale.npu(),
                bias,
                quant_scale.npu(),
                quant_offset,
                group_index.npu(),
                activate_left=True,
                quant_mode=1,
                swiglu_mode=1,
                clamp_limit=7.0,
                glu_alpha=1.702,
                glu_bias=1.0
            )

    if __name__ == "__main__":
        run_tests()

    ```
