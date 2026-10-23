# (beta) torch::npu::synchronize

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

Synchronizes the NPU device. This function blocks the current thread until all computation tasks submitted to the NPU device complete execution. This function is identical to `void torch::cuda::synchronize(int64_t device_index = -1)`.

## Definition File

torch_npu\csrc\libs\init_npu.h

## Prototype

```cpp
void torch::npu::synchronize(int64_t device_index = -1)
```

## Parameters

**`device_index`** (`int64_t`): Optional. Device index used for synchronization. The default value is `-1`, which specifies to synchronize the current device.
