# torch_npu.utils.set_thread_affinity

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

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

Sets the CPU pinning range for the current thread.

## Prototype

```python
torch_npu.utils.set_thread_affinity(core_range: list[int] | list[list[int]] | None = None)
```

## Parameters

 **`core_range`** (`list[int] | list[list[int]] | None`): Optional. Specifies the CPU core ranges to which the current thread is expected to be bound. The default value is `None`, indicating that the current thread is automatically bound to CPU cores as a non-primary thread. To configure the CPU core ranges, use either of the following formats:

- `list[int]`: Sets a single CPU core range for the current thread. For example, `[0, 3]` specifies that the current thread is to be bound to CPU cores 0, 1, 2, and 3.
- `list[list[int]]`: Sets multiple CPU core ranges for the current thread. For example, `[[0, 3], [5, 7]]` specifies that the current thread is to be bound to CPU cores 0, 1, 2, 3, 5, 6, and 7.

## Return Values

None

## Constraints

This API takes effect only when the mode of the environment variable `CPU_AFFINITY_CONF` is set to `1` or `2`. This API is generally used before a sub-thread is started to specify the CPU pinning method or CPU pinning range of the sub-thread. Use this API together with [torch_npu.utils.reset_thread_affinity](torch_npu-utils.reset_thread_affinity.md).

## Example

```python
>>> import torch_npu
>>> import threading
>>>
>>> def run_thread():
...     print("This is a child thread.")
>>>
>>> torch_npu.utils.set_thread_affinity([12, 19])             # Set a single CPU core range: [12, 19]
>>> torch_npu.utils.set_thread_affinity([[0, 10], [12, 19]])  # Set multiple CPU core ranges: [0, 10] and [12, 19]
>>> child_thread = threading.Thread(target=run_thread)
>>> child_thread.start()
>>> torch_npu.utils.reset_thread_affinity()
>>> child_thread.join()
```
