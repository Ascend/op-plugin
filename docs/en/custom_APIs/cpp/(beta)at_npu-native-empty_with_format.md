# (beta) at_npu::native::empty_with_format

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

Obtains an empty NPU tensor in a specified format. The return value type is `Tensor`, which represents the obtained empty tensor.

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
at::Tensor at_npu::native::empty_with_format(c10::IntArrayRef sizes, const c10::TensorOptions& options, int64_t acl_format, bool keep_format = false)
```

## Parameters

- **`sizes`** (`IntArrayRef`): Shape dimensions of the tensor to be obtained.

- **`options`** (`TensorOptions`): Optional configuration attributes of the tensor, such as `dtype` or `device`.

- **`acl_format`** (`int64_t`): Format of the tensor.

- **`keep_format`** (`bool`): Optional. Specifies whether to enforce the requested format. Valid values are `true` (enforces the specified format) or `false` (allows the tensor format to be adjusted based on the actual operator execution requirements).
