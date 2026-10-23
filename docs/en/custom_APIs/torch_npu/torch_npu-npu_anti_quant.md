# torch_npu.npu_anti_quant

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
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

- Function: dequantizes the tensor `x`, restoring the quantized low-precision data to floating-point numbers. Dequantization is the inverse process of quantization and is used to convert low-precision data (such as `torch.int8`, `torch_npu.int4`, `torch_npu.hifloat8`, `torch.float8_e4m3fn`, and `torch.float8_e5m2`) back to high-precision floating-point numbers.

- Formulas: `out` is the output tensor, `x` is the quantized input, `scale` is the scaling factor, `offset` is the offset, and `dst_dtype` specifies the output data type.
  
  - When `offset` is provided:

    $out = cast\_to\_dst\_dtype((x + offset) * scale)$
  
  - When `offset` is omitted:

    $out = cast\_to\_dst\_dtype(x * scale)$

## Prototype

```python
torch_npu.npu_anti_quant(x, scale, *, offset=None, dst_dtype=None, src_dtype=None) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor to be dequantized. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported. Up to 8 dimensions are supported.

  <!-- npu="310p" id5 -->
  - <term>Atlas inference products</term>: The data type can be `torch.int8`.
  <!-- end id5 -->
  <!-- npu="910b" id6 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.int8` or `torch.int32` (where each `torch.int32` element is a packed combination of eight `torch_npu.int4` values).
  <!-- end id6 -->
  <!-- npu="A3" id7 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.int8` or `torch.int32` (where each `torch.int32` element is a packed combination of eight `torch_npu.int4` values).
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.int8`, `torch.int32`, `torch_npu.hifloat8`, `torch.float8_e4m3fn`, or `torch.float8_e5m2`, where each `torch.int32` element is packed from eight `torch_npu.int4` values.
  <!-- end id8 -->

- **`scale`** (`Tensor`): Required. Scaling value used in dequantization. This parameter must be 1D with shape `(n,)`, where `n` can be 1. If `n` is not 1, when `x` is of type `torch.int8`, `n` must match the size of the last dimension of `x`; when `x` is of type `torch.int32`, `n` must be exactly 8 times the size of the last dimension of `x`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported.

  <!-- npu="310p" id9 -->
  - <term>Atlas inference products</term>: The data type can be `torch.float32`. `scale` supports only 1D and can only be equal to the size of the last axis of `x` or 1. When `x` is of the `torch.int32` type, it must be 8 times the size of the last axis of `x`.
  <!-- end id9 -->
  <!-- npu="910b" id10 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.float32` or `torch.bfloat16`. `scale` supports only 1D and can only be equal to the size of the last axis of `x` or 1; when `x` is of the `torch.int32` type, it must be 8 times the size of the last axis of `x`.
  <!-- end id10 -->
  <!-- npu="A3" id11 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.float32` or `torch.bfloat16`. `scale` supports only 1D and can only be equal to the size of the last axis of `x` or 1; when `x` is of the `torch.int32` type, it must be 8 times the size of the last axis of `x`.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float32` or `torch.bfloat16`. When `x` is of type `torch_npu.hifloat8`, `torch.float8_e4m3fn`, or `torch.float8_e5m2`, `scale` can only be `torch.float32`.
  <!-- end id12 -->

- **`*`**: Position delimiter used to distinguish positional arguments from keyword arguments. Variables before it are position-dependent and must be passed in order; variables after it are optional keyword arguments and can be passed in any order using key-value pairs. If not specified, their default values are used.

- **`offset`** (`Tensor`): Optional. Offset value used in dequantization. This parameter must be a 1D tensor. The data type and shape must be identical to those of `scale`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported.

- **`dst_dtype`** (`ScalarType`): Optional. Target data type of the output tensor. The default value is `torch.float16`.

  <!-- npu="310p" id13 -->
  - <term>Atlas inference products</term>: The data type can be `torch.float16`.
  <!-- end id13 -->
  <!-- npu="910b" id14 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id14 -->
  <!-- npu="A3" id15 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id15 -->
  <!-- npu="950" id16 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.float16` or `torch.bfloat16`.
  <!-- end id16 -->

- **`src_dtype`** (`ScalarType`): Optional. Source data type of the input tensor. The default value is `torch.int8`.

  <!-- npu="310p" id17 -->
  - <term>Atlas inference products</term>: The data type can be `torch.int8`.
  <!-- end id17 -->
  <!-- npu="910b" id18 -->
  - <term>Atlas A2 products</term>: The data type can be `torch.quint4x2` or `torch.int8`.
  <!-- end id18 -->
  <!-- npu="A3" id19 -->
  - <term>Atlas A3 products</term>: The data type can be `torch.quint4x2` or `torch.int8`.
  <!-- end id19 -->
  <!-- npu="950" id20 -->
  - <term>Ascend 950PR&950DT products</term>: The data type can be `torch.quint4x2`, `torch_npu.int4`, `torch.int8`, `torch_npu.hifloat8`, `torch.float8_e5m2`, or `torch.float8_e4m3fn`.
  <!-- end id20 -->

## Return Values

`Tensor`

Represents the computation result of `npu_anti_quant`, $out$ in the formulas. Non-contiguous tensors are supported. Empty tensors are supported.

## Constraints

- This API can be used in inference and training scenarios.
- This API supports single-operator mode and TorchAir graph mode.
- The inputs `x` and `scale` must not be `None`.

## Example

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int8).npu()
    >>> scale = torch.tensor([2.0], dtype=torch.float).npu()
    >>> offset = torch.tensor([2.0], dtype=torch.float).npu()
    >>> out = torch_npu.npu_anti_quant(x_tensor, scale, offset=offset, dst_dtype=torch.float16)
    >>> print(out)
    tensor([ 6.,  8., 10., 12.], device='npu:0', dtype=torch.float16)
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig
    
    config = CompilerConfig()
    config.debug.graph_dump.type = 'pbtxt'
    npu_backend = tng.get_npu_backend(compiler_config=config)
    x_tensor = torch.tensor([1,2,3,4], dtype=torch.int8).npu()
    scale = torch.tensor([2.0], dtype=torch.float).npu()
    offset = torch.tensor([2.0], dtype=torch.float).npu()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,x,scale,offset):
            return torch_npu.npu_anti_quant(x, scale, offset=offset, dst_dtype=torch.float16)

    cpu_model = Model()
    model = cpu_model.npu()
    model = torch.compile(model, backend=npu_backend, dynamic=False, fullgraph=True)
    output = model(x_tensor,scale,offset)
    print(output)

    # Expected output of the preceding code sample:
    tensor([ 6.,  8., 10., 12.], device='npu:0', dtype=torch.float16)
    ```
