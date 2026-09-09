# (beta) at_npu::native::get_npu_storage_sizes

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
std::vector<int64_t> at_npu::native::get_npu_storage_sizes(const at::Tensor& self)
```

## Function

Obtains the memory size of an NPU tensor. The return value type is `vector<int64_t>`, which represents the allocated memory size of the NPU tensor.

## Parameters

**`self`** (`Tensor`): Tensor whose allocated memory size is to be obtained.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
