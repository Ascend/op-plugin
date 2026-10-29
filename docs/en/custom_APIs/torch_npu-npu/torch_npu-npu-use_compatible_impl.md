# torch_npu.npu.use_compatible_impl

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: Supported
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas inference products</term>: Supported
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas training products</term>: Supported
<!-- end id7 -->

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

Currently, `torch.nn.functional.gelu`, `torch.nn.LayerNorm`, `torch.distributed.all_to_all`, `torch.distributed.scatter`, `torch.distributed.gather`, `torch.nn.MaxPool2d`, `torch.nn.functional.max_pool2d`, and `torch.nn.functional.scaled_dot_product_attention` are supported.

For `torch.nn.functional.scaled_dot_product_attention`, enabling this feature allows the MATH backend to be selected in the context of the SDPA backend.

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
