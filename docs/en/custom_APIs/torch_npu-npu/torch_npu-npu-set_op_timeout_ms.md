# torch_npu.npu.set_op_timeout_ms

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

Sets the execution timeout for operators on the NPU, in milliseconds (ms).

## Prototype

```python
torch_npu.npu.set_op_timeout_ms(timeout)
```

## Parameters

**`timeout`** (`int`): Execution timeout for operators on the NPU, in milliseconds (ms).

## Return Values

None

## Constraints

None

## Example

```python
import torch
import torch_npu

torch_npu.npu.set_op_timeout_ms(1000)
```
