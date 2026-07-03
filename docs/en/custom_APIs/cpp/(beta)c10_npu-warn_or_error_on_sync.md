# (beta) c10_npu::warn_or_error_on_sync

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
void c10_npu::warn_or_error_on_sync()
```

## Function

Raises a warning or error during NPU synchronization, with no return value. This function raises an error or logs a warning based on the current warning level, which is identical to `void c10::cuda::warn_or_error_on_sync()`.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
