# torch_npu.npu.NpuGraphOpHandler

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.0.0/docs/en/release_notes/release_notes.md#api-changes).

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
| <term>Atlas A3 training products/Atlas A3 inference products</term>| √   |
| <term>Atlas A2 training products/Atlas A2 inference products</term>| √   |
| <term>Atlas inference accelerator cards</term>| √   |

## Function

Registers a custom operator handler, enabling custom operators to support dynamic shape updates and replay in `NPUGraph` mode. In `NPUGraph` mode, when you call `g.update()` with new parameters, the Ascend Extension for PyTorch framework maps the data to operator inputs through the registered handler.

Core mechanism:

1. Capture preprocessing: defines input data preprocessing logic during operator capture.
2. Dynamic update: allows dynamic modification of operator input parameters (such as sequence length and batch size) during the graph replay phase without re-capturing the graph structure. The procedure is as follows:
    1. Before replay, the user calls `g.update(cpu_update_input=[...])` to pass new parameters.
    2. The framework traverses operators in the graph and searches for registered `NpuGraphOpHandler` instances.
    3. The framework calls the `update_args` method of the handler, passing `dispatch_record` and `update_input`.
    4. In `update_args`, the user directly modifies the value at the specified index in `dispatch_record.args`.

## Definition File

torch_npu/npu/_npugraph_handlers/npugraph_handler.py

## Prototype

### NpuGraphOpHandler Base Class

```python
class NpuGraphOpHandler:
    @classmethod
    def prepare_capture(cls, func, args, kwargs): ...
    @classmethod
    def postprocess_result(cls, result, kwargs): ...
    @classmethod
    def update_args(cls, dispatch_record, update_input): ...
    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names): ...
```

### register_npu_graph_handler Decorator

```python
def register_npu_graph_handler(op_names: str | list[str]): ...
```

## Parameters

### NpuGraphOpHandler Base Class

#### 1. prepare_capture

- **`func`** (`Callable`): Original operator function object.
- **`args`** (`Tuple`): Tuple of positional arguments.
- **`kwargs`** (`Dict`): Dictionary of keyword arguments.

#### 2. postprocess_result

- **`result`** (`Any`): Original result of operator execution.
- **`kwargs`** (`Dict`): Dictionary of keyword arguments.

#### 3. update_args

- **`dispatch_record`** (`DispatchRecord`): Record object containing operator runtime information. You can modify arguments through `dispatch_record.args`.
- **`update_input`** (`Dict`): Dictionary of update parameters passed by the user.

#### 4. record_wrap_kwarg

- **`key`** (`str`): Key of the keyword argument.
- **`value`** (`Any`): Value of the keyword argument.
- **`tensor_param_names`** (`List[str]`): List of tensor parameter names.

> [!CAUTION]<br>
> All methods must be declared as `classmethod`. Do not use `self` to store state. This class is designed to be stateless.

### register_npu_graph_handler Decorator

**`op_names`** (`str` or `list[str]`): Required. Operator name corresponding to `func.__name__`. You are advised to register both `op_name` and `op_name.default`.

## Return Values

### NpuGraphOpHandler Base Class

#### 1. prepare_capture

The processed `func`, `args`, and `kwargs`. It is used for output preallocation or operator variant replacement. By default, inputs are transparently transmitted. In the source code, this method is commonly used to switch `.default` to `.out` and preallocate the workspace.

#### 2. postprocess_result

Processed result. It is used to adjust the format of the return value. By default, `result` is returned. In the source code, this method is commonly used to convert a C++ `TensorList` to a Python `list`.

#### 3. update_args

None. It is used to modify positional arguments to respond to dynamic updates. The default implementation is empty. **This method implements the core update logic.**

#### 4. record_wrap_kwarg

The processed value. It is used to customize how `kwargs` are stored. By default, weak references are used for `Tensor` objects.

### register_npu_graph_handler Decorator

The original decorated class.

## Example

This example shows how to customize a handler that implements output preallocation (`prepare_capture`), return value formatting (`postprocess_result`), dynamic argument updates (`update_args`), and custom `kwargs` storage (`record_wrap_kwarg`).

```python
import torch
import torch_npu
from torch_npu.npu import (
    NpuGraphOpHandler,
    register_npu_graph_handle
)

@register_npu_graph_handler(["my_custom_op.default"])
class ComprehensiveOpHandler(NpuGraphOpHandler):
    # Map parameters where the third parameter index 2 corresponds to "seq_len"
    _OP_ARG_SPECS = {
        "my_custom_op.default": (2, "seq_len"),
    }

    @classmethod
    def prepare_capture(cls, func, args, kwargs):
        # 1. Switch to the out operator and preallocate the output
        func_out = torch.ops.my_namespace.my_custom_op.out
        if "out" not in kwargs:
            # Define the pre-allocation logic, such as allocating space based on the input shape
            kwargs["out"] = [torch.empty_like(args[0]) for _ in range(2)]
        return func_out, args, kwargs

    @classmethod
    def postprocess_result(cls, result, kwargs):
        # 2. Adjust the return value format and convert the underlying C++ TensorList to a Python list
        return kwargs["out"]

    @classmethod
    def update_args(cls, dispatch_record, update_input):
        # 3. Update arguments dynamically
        spec = cls._OP_ARG_SPECS.get(dispatch_record.op_cache_entry.__name__)
        if spec:
            arg_index, key = spec
            # Check whether the key exists and update args
            if key in update_input and len(dispatch_record.args) >= (arg_index + 1):
                dispatch_record.args[arg_index] = update_input[key]

    @classmethod
    def record_wrap_kwarg(cls, key, value, tensor_param_names):
        # 4. Customize kwargs storage. For example, store tensors as weak references.
        if key in tensor_param_names and isinstance(value, torch.Tensor):
            return torch._C._weak_ref(value)
        return value

# Usage process
g = torch.npu.NPUGraph()
with torch.npu.graph(g, auto_dispatch_capture=True):
    # Capture phase: Call the operator to trigger prepare_capture and record_wrap_kwarg
    out_list = torch.ops.my_namespace.my_custom_op(x, weight, seq_len)

# Replay phase: Pass new arguments to trigger update_args and then postprocess_result
g.update(cpu_update_input=[{"seq_len": new_seq_len}])
g.replay()
```
