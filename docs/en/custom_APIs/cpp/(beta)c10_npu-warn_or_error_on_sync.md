# (beta) c10_npu::warn_or_error_on_sync

## Supported Products

<!-- npu="910" id1 -->
- <term>Atlas training products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas inference products</term>: Supported
<!-- end id4 -->

## Function

Raises a warning or error during NPU synchronization, with no return value. This function raises an error or logs a warning based on the current warning level, which is identical to `void c10::cuda::warn_or_error_on_sync()`.

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
void c10_npu::warn_or_error_on_sync()
```
