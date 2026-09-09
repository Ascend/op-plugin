# torch_npu.utils.reset_thread_affinity

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

Restores the CPU pinning range for the current thread to the main thread.

## Prototype

```python
torch_npu.utils.reset_thread_affinity()
```

## Parameters

None

## Return Values

None

## Constraints

This API takes effect only when the mode of the environment variable `CPU_AFFINITY_CONF` is set to `1` or `2`. This API is generally used after a sub-thread is started to restore the CPU pinning range for the current thread to the main thread range. Use this API together with [torch_npu.utils.set_thread_affinity](torch_npu-utils.set_thread_affinity.md).

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
