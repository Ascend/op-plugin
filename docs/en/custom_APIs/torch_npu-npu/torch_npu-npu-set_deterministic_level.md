# torch_npu.npu.set_deterministic_level

> [!NOTICE]  
> This API has changed in this version. For details about the changes, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="950" id1 -->
- <term>Ascend 950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 training products</term>: Supported
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 inference products</term>: Supported
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 training products</term>: Supported
<!-- end id4 -->
<!-- npu="910b" id5 -->
- <term>Atlas A2 inference products</term>: Supported
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas inference products</term>: Supported
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas training products</term>: Supported
<!-- end id7 -->

## Function

Controls the strong consistency feature on the CANN side by reconfiguring CANN parameters. For detailed parameter descriptions, see [aclSysParamOpt](https://gitcode.com/cann/runtime/blob/9.2.0/python/docs/en/api_ref/runtime/datatypes/aclSysParamOpt.md).

The following table describes the mapping between configuration levels and their actual behaviors:

| `torch_npu.npu.set_deterministic_level` Configuration | `aclrtSetSysParamOpt` Configuration | Actual Behavior | Correspondence with Native API `torch.use_deterministic_algorithms` |
| :----------------------------------------------------- | :----------------------------------- | :-------------- | :------------------------------------------------------------------- |
| 0 | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 0)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | Disables determinism | torch.use_deterministic_algorithms(False) |
| 1 | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | Enables determinism only | torch.use_deterministic_algorithms(True) |
| 2 | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1) | Enables strong consistency | Resolves inconsistencies in certain scenarios where some operators produce inconsistent results due to different accumulation orders. |
| 3 | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1) | Enables batch consistency | Ensures that the same output result is produced no matter how the batch size changes. |

Enabling strong consistency degrades operator execution performance. You are advised to enable this feature only when strictly identical computation results are required across different positions, or when performing precision tuning. This helps with model debugging and optimization.

## Prototype

```python
torch_npu.npu.set_deterministic_level(level)
```

## Parameters

**`level`** (`int`): Deterministic level. Valid values are `0`, `1`, or `2`. The default value is `0`.

## Return Values

None

## Constraints

- When the value of `level` is 2, this feature currently affects only `torch.mm` computation results.

- This API does not support dynamic modification. Changing the configuration during operator delivery or execution causes unexpected behavior.

- Calling this API overrides any prior `torch.use_deterministic_algorithms` configuration based on the specified `level`.

- After this API is called, `torch.use_deterministic_algorithms` must not be used to modify deterministic configurations.

- CANN 8.5.0 and later versions support `level` values of 0, 1, and 2. Only CANN 9.2.0 and later versions support a `level` value of 3.

## Example

```python
import torch
import torch_npu
torch_npu.npu.set_deterministic_level(2)
```
