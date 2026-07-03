# (beta) at_npu::native::empty_with_format

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
at::Tensor at_npu::native::empty_with_format(c10::IntArrayRef sizes, const c10::TensorOptions& options, int64_t acl_format, bool keep_format = false)
```

## Function

Obtains an empty NPU tensor in a specified format. The return value type is `Tensor`, which represents the obtained empty tensor.

## Parameters

**`sizes`** (`IntArrayRef`): Shape dimensions of the tensor to be obtained.

**`options`** (`TensorOptions`): Optional configuration attributes of the tensor, such as `dtype` or `device`.

**`acl_format`** (`int64_t`): Format of the tensor.

**`keep_format`** (`bool`): Optional. Specifies whether to enforce the requested format. Valid values are `True` (enforces the specified format) or `False` (allows the tensor format to be adjusted based on the actual operator execution requirements).

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
