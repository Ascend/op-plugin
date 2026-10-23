# (beta) c10_npu::device_count

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

Obtains the number of available NPUs. The return value type is `DeviceIndex`.

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
c10::DeviceIndex c10_npu::device_count()
```
