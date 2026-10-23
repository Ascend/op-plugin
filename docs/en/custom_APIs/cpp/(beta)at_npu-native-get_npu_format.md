# (beta) at_npu::native::get_npu_format

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

Obtains the format information of an NPU tensor and returns an `int64_t` value.

> [!NOTICE]  
> This API is typically used together with `empty_with_format` when allocating memory in an NPU proprietary format.

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
int64_t at_npu::native::get_npu_format(const at::Tensor& self)
```

## Parameters

`self` (`Tensor`): Tensor whose format information is to be obtained.
