# torch_npu.npu_group_quant

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> | √   |

## Function

- Description: Performs group-wise quantization on the input tensor.
- Formula: In the formula, $offsetOptional$ corresponds to the optional parameter `offset`.
    $$
    y=round((x*scale)+offsetOptional)
    $$

## Prototype

```python
torch_npu.npu_group_quant(x, scale, group_index, *, offset=None, dst_dtype=None) -> Tensor
```

## Parameters

- **`x`** (`Tensor`): Required. Source data tensor to be quantized, $x$ in the formula. The data type can be `float32`, `float16`, or `bfloat16`. Empty tensors are supported. Non-contiguous tensors are supported. The data layout can be ND. This parameter must be a 2D tensor. If `dst_dtype` is `quint4x2`, the size of the last dimension of `x` must be divisible by `8`.
- **`scale`** (`Tensor`): Required. Scaling factor used in quantization, $scale$ in the formula. The data type can be `float32`, `float16`, or `bfloat16`. Empty tensors are supported. Non-contiguous tensors are supported. The data layout can be ND. This parameter must be a 2D tensor. The size of the 0th dimension must not be 0, and the size of the 1st dimension of `scale` must be identical to that of the 1st dimension of `x`.
- **`group_index`** (`Tensor`): Required. Group numbers used in group-wise quantization. The data type can be `int32` or `int64`. Empty tensors are supported. Non-contiguous tensors are supported. The data layout can be ND. This parameter must be a 1D tensor. The size of the 0th dimension of `group_index` must be identical to that of the 0th dimension of `scale`.
- **`*`**: Required. Positional argument separator. Arguments before this symbol are positional-only and must be passed in sequence. Arguments after this symbol are keyword-only, position-independent options that require key-value assignments (default values are used if no value is assigned).
- **`offset`** (`Tensor`): Optional. Offset value used in quantization, $offsetOptional$ in the formula. The data type can be `float32`, `float16`, or `bfloat16`, and must be identical to that of `scale`. Empty tensors are supported. Non-contiguous tensors are supported. The data layout can be ND. This parameter must be a 1D tensor, and contains exactly one element.
- **`dst_dtype`** (`ScalarType`): Optional. Valid values are `int8` or `quint4x2`. The default value is `int8`.

## Return Values

`Tensor`

Computation result of `npu_group_quant`, $y$ in the formula. If `dst_dtype` is `int8`, the output shape is identical to that of `x`. If `dst_dtype` is `quint4x2`, the output data type is `int32`. The size of the 0th dimension is identical to that of the 0th dimension of `x`, and the size of the last dimension is 1/8 of that of the last dimension of `x`. Empty tensors are supported. Non-contiguous tensors are supported.

## Constraints

- The input `group_index` must be a non-decreasing sequence. Its minimum value must be greater than or equal to `0`, and its maximum value must be identical to the size of the 0th dimension of `x`.
- This API supports graph mode.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> x = torch.randn(6, 4).to(torch.float16).npu()
    >>> print(x)
    tensor([[ 1.0029, -1.2373,  1.0107, -0.2681],
            [ 0.5791,  0.1101,  1.0059, -0.9658],
            [-1.7637,  1.7588, -1.3193,  0.3989],
            [ 1.3262,  0.4854,  1.9551,  0.9697],
            [-0.8770, -1.8828,  2.1777, -0.0050],
            [ 0.4722,  0.5605,  0.8267, -0.9810]], device='npu:0',
        dtype=torch.float16)
    >>>
    >>> scale = torch.randn(4, 4).to(torch.float32).npu()
    >>> print(scale)
    tensor([[-0.2710, -0.9381,  0.2850, -1.1230],
            [ 0.5217, -0.7233, -0.1730, -0.1245],
            [-1.5433, -0.9129, -2.2095,  1.7371],
            [-0.8253,  0.3973,  0.1430,  0.3885]], device='npu:0')
    >>>
    >>> group_index = torch.tensor([1, 4, 6, 6], dtype=torch.int32).npu()
    >>> print(group_index)
    tensor([1, 4, 6, 6], device='npu:0', dtype=torch.int32)
    >>>
    >>> offset = torch.randn(1).to(torch.float32).npu()
    >>> print(offset)
    tensor([-1.1658], device='npu:0')
    >>>
    >>> y = torch_npu.npu_group_quant(x, scale, group_index, offset=offset, dst_dtype=torch.int8)
    >>> print(y)
    tensor([[-1,  0, -1, -1],
            [-1, -1, -1, -1],
            [-2, -2, -1, -1],
            [ 0, -2, -2, -1],
            [ 0,  1, -6, -1],
            [-2, -2, -3, -3]], device='npu:0', dtype=torch.int8)    
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    attr_dst_type = 2
    attr_dst_type_torch = torch.int8 if attr_dst_type == 2 else torch.quint4x2

    x = torch.randn(6, 4).to(torch.float16).npu()
    scale = torch.randn(4, 4).to(torch.float32).npu()
    group_index = torch.tensor([1, 4, 6, 6], dtype=torch.int32).npu()
    offset = torch.randn(1).to(torch.float32).npu()
    
    class Network(torch.nn.Module):
        def __init__(self):
            super(Network, self).__init__()
    
        def forward(self, x, scale, group_index, offset, dst_type):
            return torch_npu.npu_group_quant(x, scale, group_index, offset=offset, dst_dtype=dst_type)

    model = Network()
    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)
    config.debug.graph_dump.type = 'pbtxt'
    model = torch.compile(model, fullgraph=True, backend=npu_backend, dynamic=True)
    
    output_data = model(x, scale, group_index, offset=offset, dst_type=attr_dst_type_torch)
    print(output_data)
    
    # Expected output of the preceding code sample:
    tensor([[ 2,  0,  0,  1],
            [-1,  1,  1,  0],
            [-1, -1,  0,  0],
            [ 2,  1,  1,  0],
            [ 1, -1,  0,  1],
            [ 0,  0, -1, -1]], device='npu:0', dtype=torch.int8)
    ```
