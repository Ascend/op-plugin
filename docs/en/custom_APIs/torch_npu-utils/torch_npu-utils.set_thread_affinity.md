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

- **`core_range`** (`list[int] | list[list[int]] | None`): Optional. Specifies the CPU core ranges to which the current thread is expected to be bound. The default value is `None`, indicating that the current thread is automatically bound to CPU cores as a non-primary thread. To configure the CPU core ranges, use either of the following formats:

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
...   print("This is a child thread.")
>>>
>>> torch_npu.utils.set_thread_affinity([12, 19])             # Set a single CPU core range: [12, 19]
>>> torch_npu.utils.set_thread_affinity([[0, 10], [12, 19]])  # Set multiple CPU core ranges: [0, 10] and [12, 19]
>>> child_thread = threading.Thread(target=run_thread)
>>> child_thread.start()
>>> torch_npu.utils.reset_thread_affinity()
>>> child_thread.join()
```
