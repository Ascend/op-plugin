# torch\_npu.npu_dequant\_swiglu\_quant

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

- Function: performs the dequant dequantization + swiglu activation + quant quantization operations on the tensor `x`, and also supports grouping.
- Formulas:
    - Group grouping (only the count mode is supported currently):

        Grouping computation is performed on the input `x`. `group_index` indicates the number of Tokens in each group, and each group uses different quantization scales (such as `weight_scale`, `activation_scale`, and `quant_scale`).

        Example: Assume that x.shape=[128, 2H] and group\_index=[2, 1, 3], indicating that there are 3 groups and the corresponding scale dimension is [3, 2H]. The data of each group is dequantized separately using different scales:

        - group0=x[0:2, :], scale0=scale[0, :]
        - group1=x[2:3, :], scale1=scale[1, :]
        - group2=x[3:6, :], scale2=scale[2, :]

    - Dequant: the dequantization of the weight quantization and the dequantization of the activation quantization are performed separately.
        $$
        x=x*weight\_scale\\
        x=x*activation\_scale
        $$

    - Swiglu activation: swiglu is performed on the dequantized `x` (the left and right activations are controlled by the attr activate\_left).

        - When swiglu_mode=0 (standard Swiglu), taking the left activation as an example:
            $$
            swiglu(x)=swish(x[:,0:H])*x[:,H:2H]
            $$
            Where:
            $$
            swish(z)=z*sigmoid(z)
            $$

        - When swiglu_mode=1 (variant Swiglu), the input `x` is split in an odd-even interleaved manner:
            $$
            x\_{glu}=clamp(x\_{even},\ max=clamp\_limit)\\
            x\_{linear}=clamp(x\_{odd},-clamp\_limit,clamp\_limit)\\
            swiglu(x)=swish(x\_{glu},\alpha)*(x\_{linear}+bias)
            $$
            Where:
            $$
            swish(z,\alpha)=z*sigmoid(\alpha*z)
            $$
        - When swiglu_mode=2 (the splitting method of standard Swiglu + the computation method of variant Swiglu), taking the left activation as an example:
            $$
            x\_{glu}=clamp(x[:,0:H],\ max=clamp\_limit)\\
            x\_{linear}=clamp(x[:,H:2H],-clamp\_limit,clamp\_limit)\\
            swiglu(x)=swish(x\_{glu},\alpha)*(x\_{linear}+bias)
            $$
            Where:
            $$
            swish(z,\alpha)=z*sigmoid(\alpha*z)
            $$
        - When swiglu_mode=3 (the splitting method of standard Swiglu + the computation method of variant Swiglu, with clamp after swish), taking the left activation as an example:
            $$
            x\_{glu}=clamp(swish(x[:,0:H]),\ max=clamp\_limit)\\
            x\_{linear}=clamp(x[:,H:2H],-clamp\_limit,clamp\_limit)\\
            swiglu(x)=x\_{glu}*x\_{linear}
            $$
            Where:
            $$
            swish(z)=z*sigmoid(z)
            $$

    - Quant quantization:
        1. (Optional) Perform smooth quantization first.
            $$
            out=out*quant\_scale
            $$

        2. Dynamic/static quantization operation: quantize the activated result, taking dynamic quantization as an example.
            $$
            out,scale=dynamicquant(out)
            $$

## Prototype

```python
torch_npu.npu_dequant_swiglu_quant(x, *, weight_scale=None, activation_scale=None, bias=None, quant_scale=None, quant_offset=None, group_index=None, activate_left=False, quant_mode=0, dst_type=None, round_mode=None, activate_dim=None, swiglu_mode=0, clamp_limit=7.0, glu_alpha=1.702, glu_bias=1.0) -> (Tensor, Tensor)
```

## Parameters

> [!NOTE]
> Description of the variables used in the shapes of the Tensor:
>
>- TokensNum: number of transmitted Tokens. The value is >= 0.
>- H: length of the embedding vector. The value is > 0.
>- groupNum: length of the `group_index` input. The value is > 0.

- **`x`** (`Tensor`): Required. Target input tensor. This parameter must be 2D with shape `[TokensNum, 2H]`, and the last dimension must be an even number. The data type can be `torch.int32`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND.
- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.
- **`weight_scale`** (`Tensor`): Optional. Dequantization coefficient corresponding to weight quantization. This parameter must be 2D with shape `[groupNum, 2H]`. The data type can be `torch.float32`. The data layout can be ND. When `x` is `torch.int32`, provide `weight_scale` for dequantization.
- **`activation_scale`** (`Tensor`): Optional. Dequantization coefficient corresponding to `pertoken` activation quantization. This parameter must be 2D with shape `[TokensNum, 1]`, where the last dimension is `1` and the remaining dimensions match those of `x`. The data type can be `torch.float32`. The data layout can be ND. When `x` is `torch.int32`, provide this parameter for dequantization.
- **`bias`** (`Tensor`): Optional. Bias tensor for `x`. The data type can be `torch.int32`. The data layout can be ND. When `group_index` is provided, `bias` must be `None`.
- **`quant_scale`** (`Tensor`): Optional. Smooth quantization coefficient. This parameter must be 2D with shape `[groupNum, H]`. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND.
  > **Note:** In static quantization, quant\_scale supports only the `torch.float32` type.
- **`quant_offset`** (`Tensor`): Optional. Quantization offset. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`. The data layout can be ND. When `group_index` is provided (non-`None`), this parameter does not take effect and must be `None`.
- **`group_index`** (`Tensor`): Optional. Number of tokens per specified group in `count` mode (values must be non-negative integers). Currently, only `count` mode is supported. This parameter must be 1D. The data type can be `torch.int64`. The data layout can be ND.
- **`activate_left`** (`bool`): Optional. Applies the swish activation to the left or right half of the input after splitting it evenly along the last dimension. This parameter takes effect only when `swiglu_mode` is `0`, `2`, or `3`. The default value is `False`.
    - When it is True, out=swish\(split\[x, -1, 2\]\[0\]\)\*split\[x, -1, 2\]\[1\]
    - When it is False, out=swish\(split\[x, -1, 2\]\[1\]\)\*split\[x, -1, 2\]\[0\]

- **`quant_mode`** (`int`): Optional. The quantization type. The default value is `0`. Valid values are `0` (static quantization) or `1` (dynamic quantization). When `group_index` is provided, this parameter supports dynamic quantization only, which requires setting `quant_mode` to `1`.
- **`dst_type`** (`int`): Optional. Data type of the output `out`. `1` indicates `torch.int8`, `290` indicates `torch_npu.hifloat8`, `291` indicates `torch.float8_e5m2`, `292` indicates `torch.float8_e4m3fn`, `296` indicates `torch_npu.float4_e2m1fn_x2`, and `297` indicates `torch_npu.float4_e1m2fn_x2`. The default value is `None`, indicating that the output is `torch.int8`. When the output is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the last dimension of `out` is half of the corresponding quantization result (two elements are packed into one byte for storage).
- **`round_mode`** (`int`): Optional. Rounding mode of the output `out`. The value range is $[0,4]$: `0` indicates `"rint"`, `1` indicates `"round"`, `2` indicates `"floor"`, `3` indicates `"ceil"`, and `4` indicates `"trunc"`. When the output data type is `torch.int8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`, only `0` is supported; when it is `torch_npu.hifloat8`, only `1` is supported. The default value is `0`.
- **`activate_dim`** (`int`): Optional. Splitting axis selected for the swish computation. The value range is $[-x.dim(),\ x.dim()-1]$, and a negative value means counting from the end. When `activate_dim` is not the last axis, `group_index` must be `None`, and `quant_mode` supports only static quantization (`0`). The default value is `-1`, indicating the last axis.
- **`swiglu_mode`** (`int`): Optional. SwiGLU computation mode. The default value is 0. `0` indicates standard SwiGLU, `1` indicates variant SwiGLU (clamp, alpha, and bias are supported), `2` indicates the standard SwiGLU split method combined with the variant SwiGLU computation method, and `3` indicates the standard SwiGLU split method combined with the variant SwiGLU computation method (clamp is applied after swish; alpha and bias are not supported).
- **`clamp_limit`** (`float`): Optional. Input threshold limit for variant SwiGLU computation. The default value is `7.0`.
- **`glu_alpha`** (`float`): Optional. Activation function coefficient for GLU. The default value is `1.702`.
- **`glu_bias`** (`float`): Optional. Bias in SwiGLU computation. The default value is `1.0`.

## Return Values

- **`out`** (`Tensor`): Quantized output tensor. This parameter must be 2D with shape `[TokensNum, H]`. The data type is determined by `dst_type` and defaults to `torch.int8`. The data layout can be ND. When `dst_type` is `torch_npu.float4_e2m1fn_x2` or `torch_npu.float4_e1m2fn_x2`, the last axis is half of H.
- **`scale`** (`Tensor`): Quantization scale. This parameter must be 1D with shape `[TokensNum]`. The data type can be `torch.float32`. The data layout can be ND.

## Constraints

- This API can be used in inference scenarios.
- This API supports single-operator mode and graph mode.
- Constraints in the `group_index` scenario (not None):
    - `group_index` supports only the count mode. The network must ensure that the sum of the `group_index` input does not exceed the TokensNum dimension of `x`; otherwise, out-of-bounds access occurs.
    - The H axis has a dimension size limit: H <= 10496 and 64-aligned. Checks are performed when the specification is not met.
    - The parts of the output `out` and `scale` that exceed the sum of `group_index` are not cleared. The memory in these parts is garbage data and may contain abnormal values such as inf/nan. The network must take this into account when using them.
- When x is `torch.int32`, providing weight_scale for dequantization is recommended.
- When x is `torch.float16` or `torch.bfloat16`, weight_scale is optional (usually None, but passing it is allowed), and activation_scale and bias must be None.
- The length of the last dimension of x must be an even number.
- When the activation dimension is not the last dimension of x, group_index must be None.
- When `group_index` is not None and dynamic quantization is used (that is, `quant_mode` is 1), bias and quant_offset do not take effect.
- clamp_limit, glu_alpha, and glu_bias take effect only when swiglu_mode=1/2, and only clamp_limit takes effect when swiglu_mode=3.

## Example

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
