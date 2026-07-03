# torch_npu.npu_attention_update

> [!NOTICE]  
> This API is a new feature introduced in this version. For details about the specific dependency requirements, see [API Changes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1/docs/zh/release_notes/release_notes.md#%E6%8E%A5%E5%8F%A3%E5%8F%98%E6%9B%B4%E8%AF%B4%E6%98%8E).

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |

## Function

- Description: Updates the local intermediate variables `lse` and `local_out` output by the PagedAttention (PA) operator across each Sequence Parallelism (SP) domain into global results.

- Formulas:

  $$
  lse_{max} = \text{max}lse_i
  $$

  $$
  lse = \sum_i \text{exp}(lse_i - lse_{max})
  $$

  $$
  lse_m = lse_{max} + \text{log}(lse)
  $$

  $$
  O = \sum_i O_i \cdot \text{exp}(lse_i - lse_m)
  $$

## Prototype

```python
torch_npu.npu_attention_update(lse, local_out, update_type) -> (Tensor, Tensor)
```

## Parameters

- **`lse`** (`List[Tensor]`): Required. Local `lse` values from each SP domain, $lse_i$ in the formulas. The length of the tensor list must equal `SP`. The shape of each tensor must be $(batch \times seqLen \times headNum)$. The data type can be `float32`. The data layout can be ND. Empty tensors are supported.
- **`local_out`** (`List[Tensor]`): Required. Local attention outputs from each SP domain, $O_i$ in the formula. The length of the tensor list must equal `SP`. The shape of each tensor must be $(batch \times seqLen \times headNum, head\_dim)$. The data type can be `float32`, `float16`, or `bfloat16`. The data layout can be ND. Empty tensors are supported.
- **`update_type`** (`int`): Required. Operation type to be executed. Valid values are `0` (outputs only the merged `out` tensor) or `1` (outputs both the merged `out` and `lse_out` tensors).

## Return Values

- **`out`** (`Tensor`): Output tensor, $O$ in the formula. The shape must be $(batch \times seqLen \times headNum, head\_dim)$. The data type must be identical to the tensors within `local_out`. The data layout can be ND.
- **`lse_out`** (`Tensor`): Optional output tensor, $lse_m$ in the formula. The shape must be $(batch \times seqLen \times headNum)$. The data type is `float32`. The data layout can be ND. This tensor is returned only when `update_type` is set to `1`.

## Constraints

- Deterministic computation: This API defaults to a deterministic implementation. For identical inputs, multiple execution passes generate identical outputs to guarantee repeatability.
- The value range of the parallel degree `SP` for sequence parallelism is [1, 16].
- The value range of `head_dim` is [8, 512] and must be a multiple of 8.
- The lengths of the tensor lists for `lse` and `local_out` must be identical.

## Example

```python
import torch
import torch_npu

dtype = torch.float32
N = 4
head_dim = 32

lse = [
    torch.randn(N, dtype=dtype, device='npu'),
    torch.randn(N, dtype=dtype, device='npu'),
]

local_out = [
    torch.randn(N, head_dim, dtype=dtype, device='npu'),
    torch.randn(N, head_dim, dtype=dtype, device='npu'),
]

# update_type=0: Only the merged out tensor is returned
out, lse_out = torch_npu.npu_attention_update(lse, local_out, 0)
print("out:", out)
print("out.shape:", out.shape)

# update_type=1: Both the merged out and lse_out tensors are returned
out, lse_out = torch_npu.npu_attention_update(lse, local_out, 1)
print("out:", out)
print("lse_out:", lse_out)
```
