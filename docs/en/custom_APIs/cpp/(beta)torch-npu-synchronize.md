# (beta) torch::npu::synchronize

## Definition File

torch_npu\csrc\libs\init_npu.h

## Prototype

```cpp
void torch::npu::synchronize(int64_t device_index = -1)
```

## Function

Synchronizes the NPU device. This function blocks the current thread until all computation tasks submitted to the NPU device complete execution. This function is identical to `void torch::cuda::synchronize(int64_t device_index = -1)`.

## Parameters

**`device_index`** (`int64_t`): Optional. Device index used for synchronization. The default value is `-1`, which specifying to synchronize the current device.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
