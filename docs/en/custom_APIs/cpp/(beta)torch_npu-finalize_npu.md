# (beta) torch_npu::finalize_npu

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

Deinitializes the NPU device, which releases NPU resources.

## Definition File

torch_npu\csrc\libs\init_npu.h

## Prototype

```cpp
void torch_npu::finalize_npu()
```
