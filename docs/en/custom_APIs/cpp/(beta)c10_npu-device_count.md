# (beta) c10_npu::device_count

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
c10::DeviceIndex c10_npu::device_count()
```

## Function

Obtains the number of available NPUs. The return value type is `DeviceIndex`.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
