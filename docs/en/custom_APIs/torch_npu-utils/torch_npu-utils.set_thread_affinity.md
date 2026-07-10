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
torch_npu.utils.set_thread_affinity(core_range: List[int] = None)
```

## Parameters

 **`core_range`** (`List[int]`): Optional. Expected CPU pinning range for the current thread. The default value is `None`, indicating that the current thread is automatically bound to a core as a non-main thread. The default value is `None` (automatically binds the current thread as a non-main thread).

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
>>> torch_npu.utils.set_thread_affinity([12, 19])
>>> child_thread = threading.Thread(target=run_thread)
>>> child_thread.start()
>>> torch_npu.utils.reset_thread_affinity()
>>> child_thread.join()
```
