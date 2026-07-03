# torch_npu.profiler.profile.disable_profiler_in_child_thread

## Supported Products

| Product                                                     | Supported|
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                       |    √     |
| <term>Atlas A3 inference products</term>                       |    √     |
| <term>Atlas A2 training products</term>                       |    √     |
| <term>Atlas A2 inference products</term>|    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

Deregisters the profiler collection callback function.

Use it together with [torch_npu.profiler.profile.enable_profiler_in_child_thread](./torch_npu-profiler-profile-enable_profiler_in_child_thread.md).

## Prototype

```python
torch_npu.profiler.profile.disable_profiler_in_child_thread()
```

## Parameters

None

## Return Values

None

## Example

For details, see [torch_npu.profiler.profile.enable_profiler_in_child_thread](./torch_npu-profiler-profile-enable_profiler_in_child_thread.md).
