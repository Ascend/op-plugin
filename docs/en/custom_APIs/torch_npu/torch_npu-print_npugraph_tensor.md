# torch_npu.print_npugraph_tensor

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

In aclgraph mode, the native Python `print` function involves synchronous processing, which triggers graph breaks during `torch.compile` and cannot be captured by aclgraph. As a result, in graph mode, the `print` API cannot be used to observe the tensor data during aclgraph execution.

This API provides tensor printing capabilities similar to the native Python API without affecting aclgraph capture and replay, allowing the tensor data, data type, and shape of intermediate nodes of aclgraph to be printed directly, so that the user can observe the tensor data during aclgraph execution to quickly locate problems.

## Prototype

```python
torch_npu.print_npugraph_tensor(input, tensor_name=None) -> None
```

## Parameters

- **`input`** (`Tensor`): Required. Tensor to be printed.
- **`tensor_name`** (`str`): Optional. Name of the tensor to be printed, used to distinguish different tensors. The default value is `None`.
  - When tensor_name is None: the tensor data is output directly.
  - When tensor_name is not None: the format `{tensor_name}:` is used as the prefix, followed by the tensor data.

## Return Values

`None`

No return value.

## Constraints

This API can be used in eager mode and aclgraph mode.

## Example

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> a = torch.randn([5, 5]).npu()
    >>> torch_npu.print_npugraph_tensor(a, tensor_name = "a")
    ```

- Call based on `torch.npu.graph`

    ```python
    import torch
    import torch_npu

    x = torch.randn([5, 5]).npu()

    graph1 = torch.npu.NPUGraph()
    with torch.npu.graph(graph1):
        torch_npu.print_npugraph_tensor(x, tensor_name = "x")
        output = torch.square(x)
        torch_npu.print_npugraph_tensor(output, tensor_name = "output")

    graph1.replay()
    ```

- Call based on `torch.compile`

    ```python
    import torch
    import torch_npu

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            x = torch.add(x, x)
            torch_npu.print_npugraph_tensor(x, tensor_name ="x")
            x = torch.add(x, 2)
            torch_npu.print_npugraph_tensor(x, tensor_name="added_x")
            return x

    x = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model (x)
    ```
