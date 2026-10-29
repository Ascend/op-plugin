# torch_npu.npu.mstx

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

Provides event instrumentation interfaces.

This API supports mstx instrumentation in [torch_npu.profiler._ExperimentalConfig](../torch_npu-profiler/torch_npu-profiler-_ExperimentalConfig.md).

## Prototype

```python
torch_npu.npu.mstx()
```

## Parameters

None

## Return Values

Returns an instance of the `mstx` class. The instance itself does not store context state and is used to access MSTX instrumentation APIs.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
import torch
import torch_npu
mstx_object = torch_npu.npu.mstx()
```
