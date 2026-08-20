# (beta) torch_npu::init_npu

## Definition File

torch_npu\csrc\libs\init_npu.h

## Prototype

```cpp
void torch_npu::init_npu(const c10::DeviceIndex device_index = 0)
void torch_npu::init_npu(const std::string& device_str)
void torch_npu::init_npu(const at::Device& device)
```

## Function

Initializes the NPU device.

## Parameters

- **`device_index`** (`DeviceIndex`): ID of the NPU device to be initialized. The default value is `0`.
- **`device_str`** (`string`): Name of the device to be initialized.
- **`device`** (`Device`): NPU device to be initialized.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
