# (beta) c10_npu::current_device

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
c10::DeviceIndex c10_npu::current_device()
```

## Function

Obtains the NPU device ID. The return value type is `DeviceIndex`, which represents the obtained device ID. This function is identical to `c10::DeviceIndex c10::cuda::current_device()` in PyTorch 1.11.0. The main difference from `c10_npu::GetDevice` is that this function includes additional error checking.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
