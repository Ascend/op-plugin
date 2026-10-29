# (beta) c10::npu::current_device

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

Obtains the NPU device ID. The return value type is `DeviceIndex`. This function is identical to `c10::DeviceIndex c10::cuda::current_device()`.

## Definition File

torch_npu\csrc\libs\init_npu.h

## Prototype

```cpp
c10::DeviceIndex c10::npu::current_device()
```
