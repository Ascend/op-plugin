# (beta) c10_npu::SetDevice

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

Sets the NPU device. It specifies the NPU device to be used by the current thread. The return value is of type `aclError`.

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
aclError c10_npu::SetDevice(c10::DeviceIndex device)
```

## Parameters

**`device`** (`DeviceIndex`): NPU device ID to be set.
