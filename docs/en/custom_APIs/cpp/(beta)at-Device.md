# (beta) at::Device

## Prototype

```cpp
at::Device(const std::string &device_string)
```

## Function

After `torch_npu` is installed, the `NPU` field is supported for the Device type, allowing a device to be specified using a string description.

## Parameters

**`device_string`** (`string`): Input string that must be provided in the format `(npu)[:<device-index>]`, where `npu` specifies the device type and `<device-index>` (optional) specifies the device index.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
