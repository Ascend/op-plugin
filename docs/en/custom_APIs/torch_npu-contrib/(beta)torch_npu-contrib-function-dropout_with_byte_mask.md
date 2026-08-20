# (beta) torch_npu.contrib.function.dropout_with_byte_mask

> [!NOTICE]  
> This API is planned for deprecation. The underlying operator kernel is no longer maintained, and performance and accuracy are not guaranteed. This API is not recommended.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Applies an NPU-compatible `dropout_with_byte_mask` operation. This function is supported exclusively on NPU devices. It generates a stateless random `uint8` mask and performs dropout based on that mask.

## Prototype

```python
torch_npu.contrib.function.dropout_with_byte_mask(input1, p=0.5, training=True, inplace=False)
```

## Parameters

- **`input1`** (`Tensor`): Required. Input tensor.
- **`p`** (`float`): Optional. Dropout probability. The default value is `0.5`.
- **`training`** (`bool`): Optional. Specifies whether to enable dropout. Valid values are `True` (enables dropout) or `False` (disables dropout). The default value is `True`.
- **`inplace`** (`bool`): Optional. Specifies whether to perform the operation in-place. Valid values are `True` (modifies the input tensor in-place) or `False` (does not modify the input tensor in-place). The default value is `False`.

## Constraints

Performance is improved only in 32-core device scenarios.

## Example

```python
import torch, torch_npu
from torch_npu.contrib.function import npu_functional as F
input = torch.randn(4,4).npu()
input = torch_npu.npu_format_cast(input, 2)
output = F.dropout_with_byte_mask(input, p=0.2, training=True)
print(output)
```
