# (beta) torch_npu::init_npu

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

Initializes the NPU device.

## Definition File

torch_npu\csrc\libs\init_npu.h

## Prototype

```cpp
void torch_npu::init_npu(const c10::DeviceIndex device_index = 0)
void torch_npu::init_npu(const std::string& device_str)
void torch_npu::init_npu(const at::Device& device)
```

## Parameters

- **`device_index`** (`DeviceIndex`): ID of the NPU device to be initialized. The default value is `0`.
- **`device_str`** (`string`): Name of the device to be initialized.
- **`device`** (`Device`): NPU device to be initialized.
