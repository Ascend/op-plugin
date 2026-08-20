# torch_npu.save_npugraph_tensor

> [!NOTICE]  
> This API is updated in this version. For details about the specific changes, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>| √   |

## Function

In `aclgraph` mode, the native Python `torch.save` function introduces synchronous operations that trigger graph breaks during `torch.compile`. These operations cannot be captured by `aclgraph`, preventing the use of standard `print` to inspect tensor data during graph execution.

This API provides a tensor dumping capability similar to PyTorch `torch.save` without affecting `aclgraph` capture and replay. This API allows the tensor data, data types, and shape information of intermediate nodes within an `aclgraph` to be saved to a specified binary file, enabling users to inspect tensor data during `aclgraph` execution and quickly locate issues.

This API supports saving either a single tensor or multiple tensors provided in a tensor list.

The saved binary files can be read using the `torch.load()` API to reconstruct the tensors.

## Prototype

```python
torch_npu.save_npugraph_tensor(input, save_path=None, overwrite=False) -> None
```

## Parameters

- **`input`** (`Tensor`/`List[Tensor]`): Required. Tensor or tensor list to be saved.

- **`save_path`** (`str`): Optional. Full path for saving the file.
  - Absolute and relative paths are supported. The path must include a file name. The path configuration rules are the same as those of the native PyTorch `torch.save`. The caller is responsible for ensuring that the parent directory exists.
  - The supported file extensions are the same as those of the native `torch.save`, including `.pt`, `.pth`, and `.bin`.
  - If no file path is specified, the file is saved to the current working directory by default, using the naming format `tensor_<timestamp>_device_<device_id>_<counter>.pt`, such as `tensor_20260101_000000_000000_device_0_0.pt`.

- **`overwrite`** (`bool`): Optional. Specifies whether to overwrite an existing file with the same name. The default value is `False`.
  - When `overwrite=False`, a counter is automatically appended to the file name if a file with the same name already exists, for example, `tensor_device_0_0.pt` and `tensor_device_0_1.pt`.
  - When `overwrite=True`, a file with the same name is directly overwritten, and no counter is appended to the file name, for example, `tensor_device_0.pt`.

## Return Values

`None`

None.

## Constraints

- This API can be used in `Eager` mode and `aclgraph` mode.

- Input tensors in proprietary formats are not supported.

- The actual file path includes the device ID to distinguish files in multi-device scenarios. When `overwrite=False`, each file has a counter appended to the file name to distinguish multiple files with the same name. For example, when `"tensor.pt"` is written multiple times, multiple files are generated, such as `"tensor_device_0_0.pt"`, `"tensor_device_0_1.pt"`, and so on. When `overwrite=True`, the file is directly overwritten as `"tensor_device_0.pt"`.

## Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randn([5, 5]).npu()
    >>> torch_npu.save_npugraph_tensor(x, save_path = "./x.bin")
    >>> y = torch.add(x, x)
    >>> torch_npu.save_npugraph_tensor(y)
    ```

- `torch.npu.graph`-based call

    ```python
    import torch
    import torch_npu

    x = torch.randn([5, 5]).npu()

    graph1 = torch.npu.NPUGraph()
    with torch.npu.graph(graph1):
        torch_npu.save_npugraph_tensor(x, save_path="./x.pt")
        output = torch.square(x)
        torch_npu.save_npugraph_tensor(output, save_path="./output.pt")

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
            torch_npu.save_npugraph_tensor(x, save_path="/home/dump/output.pt")
            x = torch.add(x, 2)
            torch_npu.save_npugraph_tensor(x)
            return x

    x = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model (x)
    ```

- Load and inspect the saved file

    ```python
    >>> x = torch.load("./x_device_0.pt")
    >>> print(x)
    >>> print(x.shape)
    ```
