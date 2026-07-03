# (beta) c10_npu::warning_state

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
c10_npu::WarningState& c10_npu::warning_state()
```

## Function

Obtains the current synchronization warning level. The return type is the `WarningState` enum class, which can be `L_DISABLED` (no warning), `L_WARN` (warning), or `L_ERROR` (error). This function is identical to `WarningState& c10::cuda::warning_state()` in PyTorch 1.11.0.

## Supported Products

- <term>Atlas training products</term>
- <term>Atlas A2 training products</term>
- <term>Atlas A3 training products</term>
- <term>Atlas inference products</term>
