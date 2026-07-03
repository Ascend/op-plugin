# (beta) c10_npu::SetDevice

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
aclError c10_npu::SetDevice(c10::DeviceIndex device)
```

## Function

Sets the NPU device. The return value type is `aclError`, which is identical to `cudaError_t c10::cuda::CUDAGuard::set_device(Device device)` in PyTorch 1.11.0.

## Parameters

**`device`** (`DeviceIndex`): NPU device ID to be set.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
