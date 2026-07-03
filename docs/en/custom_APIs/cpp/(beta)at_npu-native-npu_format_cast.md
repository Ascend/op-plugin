# (beta) at_npu::native::npu_format_cast

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
at::Tensor at_npu::native::npu_format_cast(const at::Tensor& self, int64_t acl_format)
```

## Function

Converts the format of an NPU tensor. The return value type is `Tensor`, which represents the converted tensor.

## Parameters

**`self`** (`Tensor`): Tensor whose format is to be converted.

**`acl_format`** (`int64_t`): Destination format for conversion.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
