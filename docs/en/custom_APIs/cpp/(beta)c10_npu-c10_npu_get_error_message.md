# (beta) c10_npu::c10_npu_get_error_message

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

Obtains error messages. The return value type is `const char *`, which represents the obtained error message string.

## Definition File

torch_npu\csrc\core\npu\NPUException.h

## Prototype

```cpp
const char* c10_npu::c10_npu_get_error_message()
```
