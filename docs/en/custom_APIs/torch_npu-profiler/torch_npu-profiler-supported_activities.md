# torch_npu.profiler.supported_activities

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
<!-- npu="910" id4 -->
- <term>Atlas training products</term>: Supported
<!-- end id4 -->

## Function

Queries the CPU or NPU events of the `activities` parameter supported for collection.

## Prototype

```python
torch_npu.profiler.supported_activities()
```

## Return Values

Returns a `set` containing the currently supported activity types. On success, `{<ProfilerActivity.CPU: 1>, <ProfilerActivity.NPU: 2>}` is returned, indicating that CPU and NPU events are currently supported. On failure, an empty set `set()` is returned.

## Example

The following code sample demonstrates the key steps and is for reference only. Do not directly copy or run the code.

```python
import torch
import torch_npu

...

torch_npu.profiler.supported_activities()
```
