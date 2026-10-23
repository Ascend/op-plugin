# torch_npu.npu_gelu_quant

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: Applies the GELU (Gaussian Error Linear Unit) activation to a Tensor and then performs static or dynamic quantization on the result.

- Formulas:

  - **First compute Gelu to obtain geluOut**

     - approximate = "tanh"

       $$
       geluOut = 0.5 * self * (1 + Tanh(\sqrt{2 / \pi} * (self + 0.044715 * self^{3})))
       $$

     - approximate = "none"

       $$
       geluOut = 0.5 * self * [1 + erf(self/\sqrt{2})]
       $$

  - **Then quantize geluOut**

     - quant_mode = "static"

       $$
       y = round\_to\_dstType(geluOut * input\_scale + input\_offset, round\_mode)
       $$

     - quant_mode = "dynamic"

       $$
       geluOut = geluOut * input\_scale \\
       Max = max(abs(geluOut)) \\
       out\_scale = Max / maxValue \\
       y = round\_to\_dstType(geluOut / out\_scale, round\_mode)
       $$

  - maxValue: Maximum value of the corresponding data type.

    | Data type | maxValue |
    | :---: | :---: |
    | `torch.int8` | 127 |
    | `torch.float8_e4m3fn` | 448 |
    | `torch.float8_e5m2` | 57344 |
    | `torch_npu.hifloat8` | 32768 |

## Prototype

```python
torch_npu.npu_gelu_quant(self, *, input_scale=None, input_offset=None, approximate="none", quant_mode="dynamic", dst_type=None, round_mode="rint") -> (Tensor, Tensor)
```

## Parameters

- **`self`** (`Tensor`): Required. Input Tensor to be activated by Gelu and quantized. The data type can be `torch.float16`, `torch.bfloat16`, or `torch.float32`. The data layout can be ND. Non-contiguous Tensors are supported. Empty Tensors are not supported.
  - When `quant_mode` is `"dynamic"`, the shape can have 2 to 8 dimensions.
  - When `quant_mode` is `"static"`, the shape can have 1 to 8 dimensions.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`input_scale`** (`Tensor`): Optional. Input quantization scale (static scenario) or smooth scaling scale (dynamic scenario). The data type must be the same as that of `self` or use a higher precision (for example, `self` is `torch.bfloat16` and `input_scale` is `torch.float32`). Only a 1D shape is supported, and the size can only be the last-axis dimension of `self` or 1. Non-contiguous Tensors are supported. The data layout can be ND. When `quant_mode` is `"static"`, this parameter is required and cannot be `None`. When `quant_mode` is `"dynamic"`, it is optional.
- **`input_offset`** (`Tensor`): Optional. Bias of the quantized input in the static quantization scenario. The data type and shape must be the same as those of `input_scale`, and only a 1D shape is supported. When `quant_mode` is `"dynamic"` and `input_scale` is `None`, `input_offset` must also be `None`.
- **`approximate`** (`str`): Optional. Mode of the Gelu activation function. The value can be `"none"` or `"tanh"`, corresponding to the erf and tanh modes respectively. The default value is `"none"`.
- **`quant_mode`** (`str`): Optional. Quantization mode. The value can be `"static"` (static quantization) or `"dynamic"` (dynamic quantization). The default value is `"dynamic"`.
- **`dst_type`** (`int`): Optional. Specifies the data type of the quantized output `y`. The data type can be `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The default value is `None`, which indicates `torch.int8`.
- **`round_mode`** (`str`): Optional. Data conversion mode. The default value is `"rint"`.
  - When `dst_type` is `torch.int8`, `torch.float8_e4m3fn`, or `torch.float8_e5m2`, only `"rint"` is supported.
  - When `dst_type` is `torch_npu.hifloat8`, `"round"` and `"hybrid"` are supported.

## Return Values

- **`y`** (`Tensor`): Result of quantization after Gelu activation. The data type is specified by `dst_type`, and the shape is the same as that of the input `self`. When `dst_type` is `torch_npu.hifloat8`, the data type of `y` is `torch.uint8` (which actually carries the `torch_npu.hifloat8` type).
- **`out_scale`** (`Tensor`): Quantization scale computed during dynamic quantization. The data type is `torch.float32`, and the shape is the shape of `self` with the last dimension removed. When `quant_mode` is `"static"`, `out_scale` is an empty Tensor.

## Constraints

- This API supports inference scenarios.
- This API supports single-operator mode and graph mode.
- When `dst_type` is `torch_npu.hifloat8`, `round_mode` must be `"round"` or `"hybrid"`. The default value `"rint"` is not supported.

## Examples

- Single-operator call

    ```python
  import torch
  import torch_npu

  x = torch.randn((32,128), dtype=torch.float32).npu()
  y, scale = torch_npu.npu_gelu_quant(x)
  ```

- Graph mode call

    ```python
  import torch
  import torch_npu
  import torchair
  import numpy as np

  class NetModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
      def forward(self, x, input_scale=None, input_offset=None, approximate='none', quant_mode='dynamic', dst_type=None, round_mode="rint"):
          return torch_npu.npu_gelu_quant(x, input_scale=input_scale, input_offset=input_offset, approximate=approximate, quant_mode=quant_mode, dst_type=dst_type, round_mode=round_mode)
  # generate inputs
  x = torch.randn((32,128), dtype=torch.float16).npu()
  model = NetModel()
  config = torchair.CompilerConfig()
  npu_backend = torchair.get_npu_backend(compiler_config=config)
  model = torch.compile(model, backend=npu_backend, dynamic=True)
  y, scale = model(x)
  ```
