# (beta) torch_npu.npu.obfuscation_initialize

## Supported Products

<!-- npu="910b" id1 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id1 -->
<!-- npu="310p" id2 -->
- <term>Atlas inference products</term>: Supported
<!-- end id2 -->

## Function

This API initializes the resources of the PMCC (Privacy and Model Confidential Computing) model obfuscation engine. It establishes a socket connection with the PMCC obfuscation engine CA (client application in a common OS), initializes the CA and the TA (trusted application in the TEE OS), and returns the socket file descriptor.
For PMCC services, either of the following results is expected:

* If the PMCC feature is deployed, this API returns a response.
* If the PMCC feature is not deployed, running the test case returns error code 507018.

The deployment procedure of the PMCC feature is as follows:

1. The NPU driver and firmware exist in the environment.
2. Install the AI obfuscation SDK and run the one-click deployment script, which automatically performs the following tasks:
   * Configure kmsAgent.
   * Deliver the npu certificate.
   * Generate the obf_sdk client certificate.
   * Generate the psk private key bound to the obfuscation factor.
3. Run the obfuscation factor registration script.

For the detailed deployment procedure of the PMCC feature, see the corresponding deployment guide.

## Prototype

```python
torch_npu.npu.obfuscation_initialize(hidden_size, tp_rank, cmd, data_type, model_obf_seed_id, data_obf_seed_id, thread_num, obf_coefficient) -> Tensor
```

## Parameters

- **`hidden_size`** (`int`): Required. Dimension of the hidden layer. The data type is `torch.int32`. Supported input values range from `1` to `10000`. A valid value is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`tp_rank`** (`int`): Required. Tensor parallelism (TP) rank. The data type is `torch.int32`. Supported input values range from `0` to `1024`. A valid value is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`cmd`** (`int`): Required. Instruction ID for resource initialization. The data type is `torch.int32`. Valid values are:
    * `1`: initializes resources for floating-point inference mode.
    * `2`: initializes resources for quantized inference mode.
    * `3`: releases resources.
- **`data_type`** (`torch.dtype`): Optional. Tensor data type. A valid value is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
    <!-- npu="310p" id3 -->
    * <term>Atlas inference products</term>: The Tensor data type can be `torch.float16`, `torch.float32`, or `torch.int8`.
    <!-- end id3 -->
    <!-- npu="910b" id4 -->
    * <term>Atlas A2 products</term>: The Tensor data type can be `torch.float16`, `torch.float32`, `torch.bfloat16`, or `torch.int8`.
    <!-- end id4 -->
- **`model_obf_seed_id`** (`int`): Optional. Model obfuscation factor ID used by the `TA` module to query the model obfuscation factor from the `TEE KMC`. The data type is `torch.int32`. A registered and valid obfuscation factor ID is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`data_obf_seed_id`** (`int`): Optional. Data obfuscation factor ID used by the `TA` module to query the data obfuscation factor from the `TEE KMC`. The data type is `torch.int32`. A registered and valid obfuscation factor ID is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`thread_num`** (`int`): Optional. Number of threads used by the `CA` and `TA` modules for obfuscation processing. The data type is `torch.int32`. Valid values are `{1, 2, 3, 4, 5, 6}`. A valid value is required only when `cmd` is set to `1` or `2`. Otherwise, set this parameter to `0`.
- **`obf_coefficient`** (`float`): Optional. Obfuscation coefficient. The value range is (0.0, 1.0]. The default value is `1.0`.

## Return Values

`Tensor`

Returns a 1D Tensor with the shape (1) and the `torch.int32` data type. When `cmd` is 1 or 2 (resource initialization), the return value is the socket file descriptor, which is used for subsequent model obfuscation computation and resource release. When `cmd` is 3 (resource release), the API disconnects the socket connection to the obfuscation engine CA, and the return value is no longer used as a socket file descriptor.

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
