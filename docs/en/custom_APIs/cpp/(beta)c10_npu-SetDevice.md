# (beta) c10_npu::SetDevice

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
aclError c10_npu::SetDevice(c10::DeviceIndex device)
```

## Function

Sets the NPU device. It specifies the NPU device to be used by the current thread. The return value is of type `aclError`.

## Parameters

**`device`** (`DeviceIndex`): NPU device ID to be set.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
