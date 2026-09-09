# (beta) c10_npu::c10_npu_get_error_message

## Definition File

torch_npu\csrc\core\npu\NPUException.h

## Prototype

```cpp
const char* c10_npu::c10_npu_get_error_message()
```

## Function

Obtains error messages. The return value type is `const char *`, which represents the obtained error message string.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
