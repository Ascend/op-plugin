# (beta) c10_npu::GetDevice

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
aclError c10_npu::GetDevice(c10::DeviceIndex* device)
```

## Function

Obtains the NPU device ID. The return value type is `aclError`, which is identical to the return value type `cudaError_t` of the `c10::cuda::GetDevice` function in PyTorch 1.11.0.

## Parameters

**`device`** (`DeviceIndex`): Stores the obtained device ID.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
