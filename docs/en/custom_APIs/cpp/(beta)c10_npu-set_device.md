# (beta) c10_npu::set_device

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

Sets the NPU device. This function is identical to `void c10::cuda::set_device(c10::DeviceIndex device)` in PyTorch 1.11.0. The main difference from `c10_npu::SetDevice` is that this function includes additional error checking.

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
void c10_npu::set_device(c10::DeviceIndex device)
```

## Parameters

**`device`** (`DeviceIndex`): NPU device ID to be set.
