# torch_npu.npu.set_deterministic_level

## Supported Products

| Product                                                     | Supported|
| --------------------------------------------------------- | :------: |
| <term>Atlas A3 training products</term>                       |    √     |
| <term>Atlas A3 inference products</term>                       |    √     |
| <term>Atlas A2 training products</term>                       |    √     |
| <term>Atlas A2 inference products</term>                       |    √     |
| <term>Atlas inference products</term>                          |    √     |
| <term>Atlas training products</term>                          |    √     |

## Function

Controls the strong consistency feature on the CANN side by reconfiguring CANN parameters. For detailed parameter descriptions, see [aclSysParamOpt](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/runtimeapi/aclcppdevg_03_1393.html).

The following table describes the mapping between configuration levels and their actual behaviors:

| `torch_npu.npu.set_deterministic_level` Configuration| `aclrtSetSysParamOpt` Configuration| Actual Behavior| Correspondence with Native API `torch.use_deterministic_algorithms`|
| :---------------------------------------- | :-------------------------------------------------------------------------------------------- | :--------- | :------------------------|
| 0                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 0)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | Disables determinism| torch.use_deterministic_algorithms(False) |
| 1                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 0) | Enables determinism only| torch.use_deterministic_algorithms(True) |
| 2                        | aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 1)<br>aclrtSetSysParamOpt(ACL_OPT_STRONG_CONSISTENCY, 1) | Enables strong consistency| Resolves inconsistencies caused by different accumulation orders in certain operators|

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

- This feature currently affects only `torch.mm` computations.

- This API does not support dynamic modification. Changing the configuration during operator delivery or execution causes unexpected behavior.

- Calling this API overrides any prior `torch.use_deterministic_algorithms` configuration based on the specified `level`.

- After this API is called, `torch.use_deterministic_algorithms` must not be used to modify deterministic configurations.

- Only CANN 8.5.0 and later versions are supported.

## Example

```python
import torch
import torch_npu
torch_npu.npu.set_deterministic_level(2)
```
