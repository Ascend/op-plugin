
# (beta) torch_npu.npu.obfuscation_calculate

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>          |    √     |
|<term>Atlas inference products</term>| √   |

## Function

Sends the input tensor `x` and configuration parameters (such as `param`) to the Privacy and Model Confidential Computing (PMCC) obfuscation engine. The Client Application (CA) module in the normal OS calls the Trusted Application (TA) module in the TEE OS to perform tensor obfuscation, and returns the final obfuscation result.
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
torch_npu.npu.obfuscation_calculate(fd, x, param, obf_coefficient) -> Tensor
```

## Parameters

- **`fd`** (`Tensor`): Required. Socket file descriptor. The data type is `int32`. Use the return value of the [obfuscation_initialize]((beta)torch_npu-npu-obfuscation_initialize.md) API.
- **`x`** (`Tensor`): Required. Input tensor to be obfuscated. The tensor supports arbitrary dimensions with shape `( , *, ..., hidden_size)`, where the last dimension size must match the `hidden_size` parameter of [obfuscation_initialize]((beta)torch_npu-npu-obfuscation_initialize.md). The data layout can be ND.
    Atlas inference products: The data type can be `float16`, `float32`, or `int8`.
    Atlas A2 training products/Atlas A2 inference products: The data type can be `float16`, `float32`, `bfloat16`, or `int8`.
- **`param`** (`Tensor`): Required. Size of the last dimension of tensor `x`. The data type is `int32`.
- **`obf_coefficient`** (`float`): Optional. Obfuscation coefficient in the range (0.0, 1.0]. The default value is `1.0`.

## Return Values

`Tensor`

The final computation result. The output tensor has the same data type and shape as the input tensor `x`.

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
param = torch.tensor([3584], device=device)
x_obf_out = torch_npu.npu.obfuscation_calculate(fd, hidden_states, param, obf_coefficient=obf_cft)
```
