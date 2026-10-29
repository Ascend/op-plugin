# (beta) c10_npu::GetDevice

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

Obtains the NPU device ID. The return value type is `aclError`, which is identical to the return value type `cudaError_t` of the `c10::cuda::GetDevice` function in PyTorch 1.11.0.

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
aclError c10_npu::GetDevice(c10::DeviceIndex* device)
```

## Parameters

**`device`** (`DeviceIndex`): Stores the obtained device ID.
