# (beta) at_npu::native::get_npu_format

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
int64_t at_npu::native::get_npu_format(const at::Tensor& self)
```

## Function

Obtains the format information of an NPU tensor. The return value type is `int64_t`, which represents the obtained NPU tensor format data.

> [!NOTICE]  
> This API is typically used together with `empty_with_format` when allocating memory in an NPU private format.

## Parameters

`self` (`Tensor`): Tensor whose format information is to be obtained.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
