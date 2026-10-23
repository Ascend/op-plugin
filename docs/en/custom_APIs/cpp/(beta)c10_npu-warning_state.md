# (beta) c10_npu::warning_state

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

Obtains the current runtime warning level. The return type is the `WarningState` enum class, which can be `L_DISABLED` (no warning), `L_WARN` (warning), or `L_ERROR` (error). This function is identical to `WarningState& c10::cuda::warning_state()` in PyTorch 1.11.0.

## Definition File

torch_npu\csrc\core\npu\NPUFunctions.h

## Prototype

```cpp
c10_npu::WarningState& c10_npu::warning_state()
```
