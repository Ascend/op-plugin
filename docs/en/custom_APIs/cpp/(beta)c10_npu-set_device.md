# (beta) c10_npu::set_device

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
void c10_npu::set_device(c10::DeviceIndex device)
```

## Function

Sets the NPU device. This function is identical to `void c10::cuda::set_device(c10::DeviceIndex device)` in PyTorch 1.11.0. The main difference from `c10_npu::SetDevice` is that this function includes additional error checking.

## Parameters

**`device`** (`DeviceIndex`): NPU device ID to be set.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
