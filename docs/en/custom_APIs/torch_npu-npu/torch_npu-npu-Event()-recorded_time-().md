# (beta) torch_npu.npu.Event().recorded_time()

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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

## Function

Obtains the time when an NPU `Event` object is recorded on the device.

## Prototype

```python
torch_npu.npu.Event().recorded_time() -> int
```

## Parameters

This member function can be called only on NPU `Event` objects.

## Return Values

- An integer (`torch.uint64`) representing the recorded timestamp in microseconds.

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
