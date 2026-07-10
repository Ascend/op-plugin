# torch_npu.save_npugraph_tensor

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>          |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term>| √   |
|<term>Atlas inference accelerator cards</term>| √   |

## Function

In `aclgraph` mode, the native Python `print` function introduces synchronous operations that trigger graph breaks during `torch.compile`. These operations cannot be captured by `aclgraph`, preventing the use of standard `print` to inspect tensor data during graph execution.

Provides a tensor dumping capability similar to the native print feature without affecting `aclgraph` replay. This API allows the tensor data, data types, and shape information of intermediate nodes within an `aclgraph` to be saved to a specified `.pt` or `.bin` file, enabling users to inspect tensor data during `aclgraph` execution and quickly locate issues.

This API supports saving either a single tensor or multiple tensors provided in a tensor list or tuple.

The saved files can be read using the `torch.load()` interface to reconstruct the tensors.

## Prototype

```python
torch_npu.save_npugraph_tensor(input, save_path=None, save_name=None, save_dir=None, suffix=None) -> None
```

## Parameters

- **`input`** (`Tensor`/`List[Tensor]`): Required. Tensor or tensor list to be saved. If a single tensor is provided, it is saved to the specified path. If a tensor list is provided, each tensor in the list is saved individually to the specified path.
- **`save_path`** (`str`): Optional. Full save path of the file, which can be used only when saving a single tensor.
  - Absolute and relative paths are supported, and a specific file name must be specified. If the path does not exist, the API preferentially attempts to create it.
  - The supported file name extensions are `.pt` or `.bin`.
  - If this parameter is omitted, the file is created in the current directory. The default naming format is `tensor_timestamp_device_deviceIndex_count.pt`, such as `tensor_2026010100000_device_0_0.pt`.
- **`save_name`** (`str`): Optional. Base file name used to construct the final file paths. This parameter is supported only when saving multiple tensors.
  - All tensors in the input tensor list share the same base file name and are distinguished by an appended index.
  - If `save_name` is provided, the file name format is `<save_name>_<input_index>_device_<device_id>_<count>`. For example, if `save_name` is `inputs`, the generated file names are `inputs_0_device_0_0`, `inputs_1_device_0_0`, and so on.
  - If `save_name` is omitted, the default file name format is `tensor_<timestamp>_<input_index>_device_<device_id>_<count>`, such as `tensor_1630000000_0_device_0_0`.
- **`save_dir`** (`str`): Optional. Directory used to construct the final file paths. This parameter is supported only when saving multiple tensors.
  - Absolute and relative paths are supported. If the path does not exist, the API preferentially attempts to create it.
  - If not specified, the current path is used by default.
- **`suffix`** (`str`): Optional. File name extension used to construct the final file paths. This parameter is supported only when saving multiple tensors.
  - Valid values are `.pt` or `.bin`. If omitted, the default value `.pt` is used.

## Return Values

`None`

None

## Constraints

- This API can be used in `Eager` mode and `aclgraph` mode.

- The actual file path will be appended with the device index to differentiate files in multi-device scenarios. Each file possesses a file count identifier to differentiate multiple files with the same name. For example, if multiple attempts are made to write to "tensor.pt", multiple files named `tensor_device_x_0.pt` and `tensor_device_x_1.pt` are generated.

## Examples

### Single-Tensor Dump Mode Call Examples

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
        torch_npu.save_npugraph_tensor(x, save_path = "./x.pt")
        output = torch.square(x)
        torch_npu.save_npugraph_tensor(output, save_path = "./output.pt")

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
            torch_npu.save_npugraph_tensor(x, save_path = "/home/dump/output.pt")
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

### Multi-Tensor Dump Mode Call Examples

- Single-operator call

    ```python
    >>> import torch
    >>> import torch_npu
    >>> x = torch.randn([5, 5]).npu()
    >>> y = torch.randn([5, 5]).npu()
    >>> z = torch.randn([5, 5]).npu()
    >>> torch_npu.save_npugraph_tensor.tensorlist((x, y, z), save_name="tensors", save_dir=".", suffix=".pt")
    ```

- `torch.npu.graph`-based call

    ```python
    import torch
    import torch_npu

    x = torch.randn([5, 5]).npu()
    y = torch.randn([5, 5]).npu()
    z = torch.randn([5, 5]).npu()

    graph1 = torch.npu.NPUGraph()
    with torch.npu.graph(graph1):
        torch_npu.save_npugraph_tensor.tensorlist((x, y, z), save_name="tensors", save_dir=".", suffix=".pt")
        output = torch.square(x)
        torch_npu.save_npugraph_tensor(output, save_path = "./output.pt")

    graph1.replay()
    ```

- `torch.compile`-based call

    ```python
    import torch
    import torch_npu

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x, y, z):
            torch_npu.save_npugraph_tensor((x, y, z), save_name="inputs", save_dir=".", suffix=".pt")
            output = torch.add(x, y)
            output2 = torch.add(output, z)
            torch_npu.save_npugraph_tensor(output2, save_name="outputs", save_dir=".", suffix=".pt")
            return output2

    x = torch.randn([5, 5]).npu()
    y = torch.randn([5, 5]).npu()
    z = torch.randn([5, 5]).npu()
    model = Model()
    model = torch.compile(model, backend="npugraph_ex", dynamic=False, fullgraph=True)
    model (x)
    ```

- Saved file parsing: same as that in single-tensor dump mode.
