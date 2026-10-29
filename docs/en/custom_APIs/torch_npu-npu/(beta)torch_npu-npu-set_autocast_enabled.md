# (beta) torch_npu.npu.set_autocast_enabled

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas training products</term>: Supported
<!-- end id5 -->

## Function

Controls whether to enable AMP on the device.

## Prototype

```python
torch_npu.npu.set_autocast_enabled(bool)
```

## Parameters

**`bool`**: Valid values are `True` (enables AMP on the device) or `False` (disables AMP).

## Example

```python
import torch
import torch_npu
torch_npu.npu.set_autocast_enabled(True)
```
