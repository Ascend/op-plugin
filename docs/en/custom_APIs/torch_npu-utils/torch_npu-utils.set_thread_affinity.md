# torch_npu.utils.set_thread_affinity

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>           |    √     |
|<term>Atlas A3 inference products</term>  | √  |
|<term>Atlas A2 training products</term> | √   |
|<term>Atlas A2 inference products</term>|    √     |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

Sets the CPU pinning range for the current thread.

## Prototype

```python
torch_npu.utils.set_thread_affinity(core_range: list[int] | list[list[int]] | None = None)
```

## Parameters

 **`core_range`** (`List[int] | list[list[int]] | None`): Optional. Expected CPU pinning range for the current thread. The default value is `None` (automatically binds the current thread as a non-main thread). To configure a CPU pinning range, use either of the following formats:

- `List[int]`: sets a single CPU pinning range for the current thread. For example, `[0, 3]` specifies that the CPU pinning range for the current thread consists of CPU cores 0, 1, 2, and 3.
- `List[List[int]]`: sets multiple CPU pinning range structures for the current thread. For example, `[[0, 3], [5, 7]]` specifies that the CPU pinning range for the current thread consists of CPU cores 0, 1, 2, 3, 5, 6, and 7.

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
...   print("This is a child thread.")
>>>
>>> torch_npu.utils.set_thread_affinity([12, 19]) # Sets a single CPU pinning range: [12, 19]
>>> torch_npu.utils.set_thread_affinity([[0, 10], [12, 19]]) # Sets multiple CPU pinning ranges: [0, 10] and [12, 19]
>>> child_thread = threading.Thread(target=run_thread)
>>> child_thread.start()
>>> torch_npu.utils.reset_thread_affinity()
>>> child_thread.join()
```
