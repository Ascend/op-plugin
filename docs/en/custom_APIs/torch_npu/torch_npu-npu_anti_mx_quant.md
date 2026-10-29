# torch_npu.npu_anti_mx_quant

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->

## Function

- API function: Dequantizes a `float4`/`float8` Tensor obtained by calling `npu_dynamic_mx_quant` into the `torch.float16`/`torch.bfloat16`/`torch.float32` format. This is the inverse process of `npu_dynamic_mx_quant`.

- Formula:
  $$
  X_{dq} = X_q \times 2^{sf - bias}
  $$

  Here, $sf$ is the scaling factor provided by the input `mxscale`; $bias$ is the exponent offset, and for the `torch.float8_e8m0fnu` format, $bias=127$; $X_q$ is the quantized `float4`/`float8` Tensor; and $X_{dq}$ is the dequantized `torch.float16`/`torch.bfloat16`/`torch.float32` Tensor.

## Prototype

```python
torch_npu.npu_anti_mx_quant(x, mxscale, *, axis=-1, dst_type=15, src_type=292) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input Tensor to be dequantized. The data type can be `torch_npu.float4_e2m1fn_x2`, `torch_npu.float4_e1m2fn_x2`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`. The data layout can be ND. The shape can have 1 to 7 dimensions. For `float8` input, the data type of `x` is the corresponding `float8` type, and non-contiguous Tensors are supported. For `float4` input, the actual data type of `x` is `torch.uint8` (every two `float4` values are packed into one `torch.uint8`, and the last dimension of the shape is half the number of actual elements), and non-contiguous Tensors are not supported.
- **`mxscale`** (`Tensor`): Required. Quantization scale involved in the dequantization computation, obtained from the output of `npu_dynamic_mx_quant`. The logical data type is `torch.float8_e8m0fnu` (the actual storage data type is `torch.uint8`). The data layout can be ND. The shape can have 2 to 8 dimensions.
- **`*`**: Variables before it are position-dependent and must be provided in order. Variables after it are optional and position-independent; they must be assigned by using key-value pairs. If not assigned, their default values are used.
- **`axis`** (`int`): Optional. Specifies the dequantization axis. Currently, only the last axis is supported, that is, the value is `-1` or `D-1`, where `D` is the number of dimensions of the input `x`. The default value is `-1`.
- **`dst_type`** (`int`): Optional. Specifies the data type of the output result. The supported values are `{5, 6, 15}`, corresponding to the output data types `{5: torch.float16, 6: torch.float32, 15: torch.bfloat16}`. The default value is `15` (`torch.bfloat16`). You can also directly pass `torch.float16`, `torch.float32`, or `torch.bfloat16`.
- **`src_type`** (`int`): Optional. Specifies the data type of the input `x`. The supported values are `{291, 292, 296, 297}`, corresponding to the input `x` data types `{291: torch.float8_e5m2, 292: torch.float8_e4m3fn, 296: torch_npu.float4_e2m1fn_x2, 297: torch_npu.float4_e1m2fn_x2}`. The default value is `292` (`torch.float8_e4m3fn`).

## Return Values

`Tensor`

Dequantization result. The data type is specified by `dst_type` and can be `torch.float16`/`torch.bfloat16`/`torch.float32`. The data layout can be ND. For `float8` input, the shape of the result is the same as that of the input `x`. For `float4` input, the last dimension of the result shape is twice the last dimension of the input `x` (stored as `torch.uint8`), and the other dimensions are the same as those of the input `x`.

## Constraints

- This API supports inference and training scenarios.
- This API supports single-operator mode and graph mode.
- The input `x` and `mxscale` must be used together. It is recommended that `mxscale` be obtained by quantization using `torch_npu.npu_dynamic_mx_quant` (`block_size` is 32).
- Shape constraints between the input `x` and `mxscale` (when `block_size` is 32):
  - rank(mxscale) = rank(x) + 1
  - axis_change = axis if axis >= 0 else axis + rank(x) (currently only the last axis is supported)
  - mxscale.shape[axis_change] = (ceil(x.shape[axis] / block_size) + 2 - 1) / 2
  - mxscale.shape[-1] = 2
  - Other dimensions are the same as those of the input x.

## Examples

- Single-operator call

    ```python
    import torch
    import torch_npu

    def anti_mx_quant_test(x_dtype, dst_type, src_type):
        # method 1: generate inputs manually
        x = torch.randn((16, 128), dtype=torch.float32).to(dtype=torch.float8_e5m2).npu()
        mxscale = torch.randint(120, 140, (16, 2, 2), dtype=torch.float32).to(dtype=torch.float8_e8m0fnu).npu()
        y = torch_npu.npu_anti_mx_quant(x, mxscale, axis=-1, dst_type=dst_type, src_type=src_type)
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("mxscale:\n", mxscale.cpu())
        print("y:\n", y.cpu())

        # method 2: generate inputs by npu_dynamic_mx_quant
        x_tmp = torch.randn((16, 128), dtype=x_dtype).npu()
        x, mxscale = torch_npu.npu_dynamic_mx_quant(x_tmp, axis=-1, round_mode="rint", dst_type=torch.float8_e5m2, block_size=32, scale_alg=0)
        y = torch_npu.npu_anti_mx_quant(x, mxscale, axis=-1, dst_type=dst_type, src_type=src_type)
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("mxscale:\n", mxscale.cpu())
        print("y:\n", y.cpu())

    if __name__ == "__main__":
        anti_mx_quant_test(torch.bfloat16, 15, 291)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair

    class AntiMxQuantModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, mxscale, axis=-1, dst_type=15, src_type=291):
            return torch_npu.npu_anti_mx_quant(x, mxscale, axis=axis, dst_type=dst_type, src_type=src_type)

    def anti_mx_quant_test(x_dtype, dst_type):
        # Construct the x tensor.
        x_tmp = torch.randn((16, 128), dtype=x_dtype).npu()
        x, mxscale = torch_npu.npu_dynamic_mx_quant(x_tmp, axis=-1, round_mode="rint", dst_type=torch.float8_e5m2, block_size=32, scale_alg=0)
        # In graph mode, mxscale must be restored from torch.uint8 storage to the torch_npu.float8_e8m0fnu type.
        mxscale = mxscale.view(torch.float8_e8m0fnu)
        model = AntiMxQuantModel()
        model.to('npu')
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        y = model(x, mxscale, axis=-1, dst_type=dst_type, src_type=291)
        y = y.cpu()
        print("AntiMxQuant result:")
        print("x:\n", x.cpu())
        print("y:\n", y)

    if __name__ == "__main__":
        anti_mx_quant_test(torch.bfloat16, 15)
    ```
