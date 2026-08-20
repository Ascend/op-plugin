# torch_npu.print_npugraph_tensor

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>| √   |

## Function

In `aclgraph` mode, the native Python `print` function introduces synchronous operations that trigger graph breaks during `torch.compile`. These operations cannot be captured by `aclgraph`, preventing the use of standard `print` to inspect tensor data during graph execution.

This API provides a tensor printing capability similar to the native Python `print` interface without affecting `aclgraph` capture and replay. This API allows direct printing of tensor data, data types, and shape information for intermediate nodes within an `aclgraph`, enabling users to observe execution behavior and quickly locate issues.

## Prototype

```python
torch_npu.print_npugraph_tensor(input, tensor_name=None) -> None
```

## Parameters

- **`input`** (`Tensor`): Required. Tensor to be printed.
- **`tensor_name`** (`str`): Optional. Name of the tensor to be printed, used to distinguish different tensors. The default value is `None`.
  - When `tensor_name` is `None`: directly outputs the tensor data content.
  - When `tensor_name` is not `None`: uses the `{tensor_name}:` as the prefix, followed by the tensor data.

## Return Values

`None`

None

## Constraints

- This API can be used in `Eager` mode and `aclgraph` mode.

- Tensors with proprietary formats are not supported as inputs.

## Example

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> a = torch.randn([5, 5]).npu()
    >>> torch_npu.print_npugraph_tensor(a, tensor_name = "a")
    ```

- `torch.npu.graph`-based call

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

- `torch.compile`-based call

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
