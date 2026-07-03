# (beta) c10::npu::current_device

## Definition File

torch_npu\csrc\libs\init_npu.h

## Prototype

```cpp
c10::DeviceIndex c10::npu::current_device()
```

## Function

Obtains the NPU device ID. The return value type is `DeviceIndex`. This function is identical to `c10::DeviceIndex c10::cuda::current_device()`.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
