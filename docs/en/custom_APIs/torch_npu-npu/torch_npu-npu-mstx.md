# torch_npu.npu.mstx

## Supported Products

| Product                                                     | Supported|
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                       |    √     |
| <term>Atlas A3 inference products</term>                       |    √     |
| <term>Atlas A2 training products</term>                       |    √     |
| <term>Atlas A2 inference products</term>|    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

Provides event instrumentation interfaces.

This API supports mstx instrumentation in [torch_npu.profiler._ExperimentalConfig](../torch_npu-profiler/torch_npu-profiler-_ExperimentalConfig.md).

## Prototype

```python
torch_npu.npu.mstx()
```

## Parameters

None

## Return Values

None

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy, compile, or run the code.

```python
import torch
import torch_npu
mstx_object = torch_npu.npu.mstx()
```
