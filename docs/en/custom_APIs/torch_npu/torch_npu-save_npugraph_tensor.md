# torch_npu.save_npugraph_tensor

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

In aclgraph mode, the native PyTorch `torch.save` function involves synchronous processing, which triggers graph breaks during `torch.compile` and cannot be captured by aclgraph. As a result, in graph mode, the tensor data during aclgraph execution cannot be observed.

This API provides tensor dump capabilities similar to the native PyTorch `torch.save` API without affecting aclgraph capture and replay, allowing the tensor data, data type, and shape of intermediate nodes of aclgraph to be saved to a specified binary file so that the user can observe the tensor data during aclgraph execution to quickly locate problems.

Saving a single tensor and saving multiple tensors (a tensor list) are supported.

The saved file can be read and the tensor can be rebuilt using the `torch.load()` API.

## Prototype

```python
torch_npu.save_npugraph_tensor(input, save_path=None, overwrite=False) -> None
```

## Parameters

- **`input`** (`Tensor`/`List[Tensor]`): Required. Tensor or tensor list to be saved.

- **`save_path`** (`str`): Optional. Full path for saving the file.
  - Absolute and relative paths are supported. The path must include a file name. The path configuration rules are the same as those of the native PyTorch `torch.save`. The caller must ensure that the parent path exists.
  - The supported file extensions are the same as those of the native `torch.save`, including ".pt", ".pth", and ".bin".
  - If no file save path is specified, the file is saved in the current path by default, with the default naming format "tensor_\<timestamp>\_device\_<device\_index>\_\<counter>.pt", for example, "tensor_20260101_000000_000000_device_0_0.pt".
  
- **`overwrite`** (`bool`): Optional. Specifies whether to overwrite an existing file with the same name. The default value is `False`.
  - When `overwrite=False`, a counter is automatically added when a file with the same name is encountered, for example, "tensor_device_0_0.pt" and "tensor_device_0_1.pt".
  - When `overwrite=True`, the file with the same name is directly overwritten, and no counter is added to the file name, for example, "tensor_device_0.pt".

## Return Values

`None`

No return value.

## Constraints

- This API can be used in eager mode and aclgraph mode.

- The actual file path has the device index appended to distinguish between devices in multi-device scenarios. When `overwrite=False`, each file has a file counter to distinguish between multiple files with the same name. For example, when writing to "tensor.pt" multiple times, multiple files are generated, namely "tensor_device_0_0.pt", "tensor_device_0_1.pt", and so on. When `overwrite=True`, the file is directly overwritten as "tensor_device_0.pt".

## Example

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randn([5, 5]).npu()
    >>> torch_npu.save_npugraph_tensor(x, save_path = "./x.bin")
    >>> y = torch.add(x, x)
    >>> torch_npu.save_npugraph_tensor(y)
    ```

- Call based on `torch.npu.graph`

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

- Call based on `torch.compile`

    ```python
    import torch
    import torch_npu

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            x = torch.add(x, x)
            torch_npu.save_npugraph_tensor(x, save_path="./output.pt")
            x = torch.add(x, 2)
            torch_npu.save_npugraph_tensor(x)
            return x

    x = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model (x)
    ```

- Parsing the saved file

    ```python
    >>> x = torch.load("./x_device_0.pt")
    >>> print(x)
    >>> print(x.shape)
    ```
