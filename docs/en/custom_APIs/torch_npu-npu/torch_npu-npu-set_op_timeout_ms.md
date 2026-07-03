# torch_npu.npu.set_op_timeout_ms

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products/Atlas A3 inference products</term>           |    √     |
|<term>Atlas A2 training products/Atlas A2 inference products</term> |    √     |
|<term>Atlas 200I/500 A2 inference products</term>|    √     |
|<term>Atlas inference products</term>|    √     |
|<term>Atlas training products</term>|    √     |

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
