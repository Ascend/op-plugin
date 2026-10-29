# (beta) at_npu::native::get_npu_storage_sizes

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

Obtains the memory size of an NPU tensor. The return value type is `vector<int64_t>`, which represents the allocated memory size of the NPU tensor.

## Definition File

torch_npu\csrc\core\npu\NPUFormat.h

## Prototype

```cpp
std::vector<int64_t> at_npu::native::get_npu_storage_sizes(const at::Tensor& self)
```

## Parameters

**`self`** (`Tensor`): Tensor whose allocated memory size is to be obtained.
