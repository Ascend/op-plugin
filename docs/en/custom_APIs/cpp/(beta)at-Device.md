# (beta) at::Device

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

After TorchNPU is installed, the `NPU` field is supported for the Device type, allowing a device to be specified using a string description.

## Prototype

```cpp
at::Device(const std::string &device_string)
```

## Parameters

**`device_string`** (`string`): Input string that must be provided in the format `(npu)[:<device-index>]`, where `npu` specifies the device type and `<device-index>` (optional) specifies the device index.
