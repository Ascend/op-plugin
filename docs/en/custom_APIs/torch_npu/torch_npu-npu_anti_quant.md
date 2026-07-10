# torch_npu.npu_anti_quant

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

- Description: Dequantizes tensor `x`, converting integers back into floating-point values.
- Formula:

  $out$ is the output, and $quant$ is the specified output data type `dst_dtype`.

  $$
  out = \text{quant}((x + \text{offset}) * \text{scale}) 
  $$

## Prototype

```python
torch_npu.npu_anti_quant(x, scale, *, offset=None, dst_dtype=None, src_dtype=None) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Input tensor to be dequantized. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported. Up to 8 dimensions are supported.
  - Atlas inference products: The data type can be `int8`.
  - Atlas A2 training products/Atlas A2 inference products: The data type can be `int8` or `int32` (where each `int32` element is a packed combination of eight `int4` values).
  - Atlas A3 training products/Atlas A3 inference products: The data type can be `int8` or `int32` (where each `int32` element is a packed combination of eight `int4` values).

- `scale` (`Tensor`): Required. Scaling value used in dequantization. This parameter must be 1D with shape `(n,)`, where `n` can be 1. If `n` is not 1, when `x` is of type `int8`, `n` must match the size of the last dimension of `x`; when `x` is of type `int32`, `n` must be exactly 8 times the size of the last dimension of `x`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported.
  - Atlas inference products: The data type can be `float32`.
  - Atlas A2 training products/Atlas A2 inference products: The data type can be `float32` or `bfloat16`.
  - Atlas A3 training products/Atlas A3 inference products: The data type can be `float32` or `bfloat16`.

- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).

- **`offset`** (`Tensor`): Optional. Offset value used in dequantization. This parameter must be a 1D tensor. The data type and shape must be identical to those of `scale`. The data layout can be ND. Non-contiguous tensors are supported. Empty tensors are supported.

- **`dst_dtype`** (`ScalarType`): Optional. Target data type of the output tensor. The default value is `float16`.
  - Atlas inference products: The data type can be `float16`.
  - Atlas A2 training products/Atlas A2 inference products: The data type can be `float16` or `bfloat16`.
  - Atlas A3 training products/Atlas A3 inference products: The data type can be `float16` or `bfloat16`.

- **`src_dtype`** (`ScalarType`): Optional. Source data type of the input tensor. The default value is `int8`.
  - Atlas inference products: The data type can be `int8`.
  - Atlas A2 training products/Atlas A2 inference products: The data type can be `quint4x2` or `int8`.
  - Atlas A3 training products/Atlas A3 inference products: The data type can be `quint4x2` or `int8`.

## Return Values

`Tensor`

Computation result of `npu_anti_quant`, $out$ in the formula. Non-contiguous tensors are supported. Empty tensors are supported.

## Constraints

- This API can be used in inference and training scenarios.
- This API supports graph mode.
- The input tensors `x` and `scale` must not be `None`.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.int8).npu()
    >>> scale = torch.tensor([2.0], dtype=torch.float).npu()
    >>> offset = torch.tensor([2.0], dtype=torch.float).npu()
    >>> out = torch_npu.npu_anti_quant(x_tensor, scale, offset=offset, dst_dtype=torch.float16)
    >>> out
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
