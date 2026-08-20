# (beta) torch_npu.npu.Event().recorded_time()

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A2 training products</term> | √   |

## Function

Obtains the time when an NPU `Event` object is recorded on the device.

## Prototype

```python
torch_npu.npu.Event().recorded_time() -> int
```

## Parameters

This member function can be called only on NPU `Event` objects.

## Return Values

- An integer (`uint64`) representing the recorded timestamp in microseconds.

- If "INTERNALError" is returned, it indicates that the `Event` object must be recorded before its timestamp is obtained.

## Constraints

When an `Event` object is created, the `enable_timing` parameter must be set to `True`.

## Example

```python
import torch
import torch_npu
 
event = torch_npu.npu.Event(enable_timing=True)
event.record()
res = event.recorded_time()
```
