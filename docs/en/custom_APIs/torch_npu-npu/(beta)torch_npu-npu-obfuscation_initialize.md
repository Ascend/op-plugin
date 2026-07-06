
# (beta) torch_npu.npu.obfuscation_initialize

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>          |    √     |
|<term>Atlas inference products</term>| √   |

## Function

Initializes resources for the Privacy and Model Confidential Computing (PMCC) model obfuscation engine by establishing a socket connection with the Client Application (CA) module in the normal OS, initializing the CA and Trusted Application (TA) modules, and returning the socket file descriptor.
For the [PMCC](https://www-file.huawei.com/admin/asset/v1/pro/view/6812dab6dd4e4640b11619e401db1c47.pdf) service, either of the following results is expected:

* If the PMCC feature is deployed, this API returns a response result.
* If the PMCC feature is not deployed, test cases return error code `507018`.

The PMCC deployment workflow is as follows:

1. Ensure that the NPU driver and firmware are installed in the environment.
2. Install the AI obfuscation SDK and run the quick deployment script, which automatically performs the following tasks:
   * Configure `kmsAgent`.
   * Deliver the NPU certificate.
   * Generate the `obf_sdk` client certificate.
   * Generate the PSK private key bound to the obfuscation factor.
3. Execute the obfuscation factor registration script.

For detailed deployment instructions, refer to the corresponding deployment guide.

## Prototype

```python
torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type, model_obf_seed_id, data_obf_seed_id, thread_num, obf_coefficient) -> Tensor
```

## Parameters

- **`hidden_size`** (`int`): Required. Dimension of the hidden layer. The data type is `int32`. Supported input values range from `1` to `10000`. A valid value is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`tp_rank`** (`int`): Required. Tensor parallelism (TP) rank. The data type is `int32`. Supported input values range from `0` to `1024`. A valid value is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`cmd`** (`int`): Required. Instruction ID for resource initialization. The data type is `int32`. Valid values are:
    * `1`: initializes resources for floating-point inference mode.
    * `2`: initializes resources for quantized inference mode.
    * `3`: releases resources.
- **`data_type`** (`int`): Optional. Numeric ID representing the tensor data type. The data type is `int32`. A valid value is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
    Atlas inference products: The data type can be `float16`, `float32`, or `int8`.
    Atlas A2 training products/Atlas A2 inference products: The data type can be `float16`, `float32`, `bfloat16`, or `int8`.
- **`model_obf_seed_id`** (`int`): Optional. Model obfuscation factor ID used by the `TA` module to query the model obfuscation factor from the `TEE KMC`. The data type is `int32`. A registered and valid obfuscation factor ID is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`data_obf_seed_id`** (`int`): Required. Data obfuscation factor ID used by the `TA` module to query the data obfuscation factor from the `TEE KMC`. The data type is `int32`. A registered and valid obfuscation factor ID is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`thread_num`** (`int`): Optional. Number of threads used by the `CA` and `TA` modules for obfuscation processing. The data type is `int32`. Valid values are `{1, 2, 3, 4, 5, 6}`. A valid value is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`obf_coefficient`** (`float`): Optional. Obfuscation coefficient. The value range is (0.0, 1.0]. The default value is `1.0`.

## Return Values

`Tensor`

A 1D tensor with shape `(1)` and data type `int32`, representing the socket file descriptor.

## Example

```python
import torch
import torch_npu

device = "npu:0"
hidden_size = int(3584)
cmd = 1
data_type = torch.bfloat16
model_obf_seed = 0
data_obf_seed = 0
thread_num = 4
tp_rank = 0
i = 0
hidden_states = torch.randn((1024,3584), dtype=torch.bfloat16, device=device)
obf_cft = 1.0
fd = torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type=data_type, thread_num= thread_num, obf_coefficient=obf_cft)
```
