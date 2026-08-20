# (beta) torch_npu.npu.get_amp_supported_dtype

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √    |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Obtains the data types supported by the NPU device. A device may support multiple data types.

## Prototype

```python
torch_npu.npu.get_amp_supported_dtype()
```

## Return Values

**List**(`torch.dtype`)

## Example

```python
import torch
import torch_npu

supported_dtypes = torch_npu.npu.get_amp_supported_dtype()
print (f"AMP data types supported by the NPU: {supported_dtypes}")

```
