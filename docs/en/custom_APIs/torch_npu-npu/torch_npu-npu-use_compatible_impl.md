# torch_npu.npu.use_compatible_impl

## Supported Products

| Product                                                     | Supported|
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                       |    √     |
| <term>Atlas A3 inference products</term>                       |    √     |
| <term>Atlas A2 training products</term>                       |    √     |
| <term>Atlas A2 inference products</term>                       |    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

Controls whether operator APIs are fully aligned with the PyTorch community implementations. This API is used only to switch the underlying operators called by operator APIs.
You can use `torch_npu.npu.are_compatible_impl_enabled` to check whether this feature is enabled.

## Prototype

```python
torch_npu.npu.use_compatible_impl(is_enable)
```

## Parameters

**`is_enable`** (`bool`): Specifies whether to enable compatible implementation.

 - `True`: enables compatible operator replacement.
 - `False`: disables compatible operator replacement.

## Return Values

None

## Constraints

Currently, `torch.nn.functional.gelu`, `torch.nn.LayerNorm`, `torch.distributed.all_to_all`, `torch.distributed.scatter`, `torch.distributed.gather`, `torch.nn.MaxPool2d`, and `torch.nn.functional.max_pool2d` are supported.

## Example

```python
import torch
import torch_npu

torch_npu.npu.use_compatible_impl(True)
shape = [100, 400]
mode = "none"
input = torch.rand(shape, dtype=torch.float16).npu()
output = torch.nn.functional.gelu(input, approximate=mode)
```
