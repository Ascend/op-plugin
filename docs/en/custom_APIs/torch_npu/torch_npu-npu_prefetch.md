# torch_npu.npu_prefetch

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>     |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>           |    √     |

## Function

Provides a network `weight` prefetching feature to pre-load specified weight data into the L2 Cache before computation begins, reducing memory access wait time when operators access these weights. For example, performing a prefetch before operators such as MatMul allows the operator to read weights directly from the low-latency L2 Cache during execution, which improves operator data access and computation efficiency. The actual performance gain depends on the parallelism strategy and configurations.

## Prototype

```python
torch_npu.npu_prefetch(input, dependency, max_size, offset=0) -> None
```

## Parameters

- **`input`** (`Tensor`): Weights to be prefetched. No data processing is performed. This parameter is independent of data type and data layout. The input must not contain a null pointer.
- **`dependency`** (`Tensor`): Node that specifies the start of prefetching. This parameter does not take effect in single-operator mode and can be set to `None`. In graph capture mode, this parameter must not be `None`. No data processing is performed. This parameter is independent of data type and data layout.
- **`max_size`** (`int`): Maximum size of weights to prefetch. The value must be greater than 0. When the size of the weights exceeds this value, it is set to the maximum size of the weights. The data type can be `int32` or `int64`.
- **`offset`** (`int`): Memory address offset for weight prefetching. The value must not exceed the weight address range. The data type can be `int32` or `int64`.

## Return Values

None.

## Constraints

This API supports graph mode.

## Examples

- Single-operator multi-stream concurrent call

    ```python
    >>> import torch
    >>> import torch_npu
    >>>
    >>> s_cmo = torch.npu.Stream()
    >>> x = torch.randn(10000, 10000, dtype=torch.float32).npu()
    >>> y = torch.randn(10000, 1, dtype=torch.float32).npu()
    >>> add = torch.add(x, 1)
    >>>
    >>> with torch.npu.stream(s_cmo):
    ...     torch_npu.npu_prefetch(y, None, 10000000)
    ...
    >>> abs = torch.abs(add)
    >>> mul = torch.matmul(abs, abs)
    >>> out = torch.matmul(mul, y)
    >>> out
    [W compiler_depend.ts:133] Warning: Warning: Device do not support double dtype now, dtype cast replace with float. (function operator())
    tensor([[-946066.3750],
            [-945756.1875],
            [-953013.2500],
            ...,
            [-938365.2500],
            [-951188.7500],
            [-941926.4375]], device='npu:0')
    >>> out.shape
    torch.Size([10000, 1])
    ```

- Graph mode call

    ```python
    import torch
    import torch_npu
    import torchair as tng
    from torchair.ge_concrete_graph import ge_apis as ge
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    config.debug.graph_dump.type = "pbtxt"
    npu_backend = tng.get_npu_backend(compiler_config=config)

    x = torch.randn(10000, 10000, dtype=torch.float32).npu()
    y = torch.randn(10000, 1, dtype=torch.float32).npu()


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            add = torch.add(x, 1)
            torch_npu.npu_prefetch(y, add, 10000000)
            abs = torch.abs(add)
            mul = torch.matmul(abs, abs)
            out = torch.matmul(mul, y)
            return out


    npu_model = Model().npu()
    model = torch.compile(npu_model, backend=npu_backend, dynamic=False, fullgraph=True)
    output = model(x, y)
    print(output)
    print(output.shape)

    # Expected output of the preceding code sample:   
    tensor([[83962.5078],
            [87820.6328],
            [87498.8594],
            ...,
            [76254.0781],
            [87780.6484],
            [75411.2188]], device='npu:0')
    torch.Size([10000, 1])
    ```
