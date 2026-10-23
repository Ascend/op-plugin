# torch_npu.batch_norm_reduce

> [!NOTICE]  
> This API is new in this version. For details about the dependency requirements, see "[API Change Description](https://gitcode.com/Ascend/pytorch/blob/v2.14.0-26.2.0/docs/en/release_notes.md#api-change-description)" in the *Release Notes*.

## Supported Products

<!-- npu="A3" id1 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id2 -->

## Function

- API function: This API reduces the input Tensor along the BatchNorm channel dimension and computes the element sum `sum` and the sum of squares `square_sum` over the dimensions other than the channel dimension.

- Formulas: For a 4D Tensor input `(N, C, H, W)`, the sum and sum of squares are computed over the N, H, and W axes other than the C axis.

    $$
    sum_i = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} input_{(n,i,h,w)}
    $$

    $$
    square\_sum_i = \sum_{n=0}^{N-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} input_{(n,i,h,w)}^2
    $$

## Prototype

```python
torch_npu.batch_norm_reduce(input, eps) -> (Tensor, Tensor)
```

## Parameters

- **`input`** (`Tensor`): Required. Input Tensor to be reduced. A 4D Tensor with shape `(N, C, H, W)`. The data type can be `torch.float32`, `torch.float16`, or `torch.bfloat16`.
- **`eps`** (`float`): Required. Reserved parameter. It does not participate in the computation of `sum` and `square_sum`. The recommended value is `1e-5`, which is the commonly used BatchNorm configuration.

## Return Values

- **`sum`** (`Tensor`): Element sum after reducing the input Tensor along the channel dimension. A 1D `torch.float32` Tensor with length equal to the channel dimension. When the input is a 4D Tensor with shape `(N, C, H, W)`, the output shape is `(C,)`.
- **`square_sum`** (`Tensor`): Sum of squares after reducing the input Tensor along the channel dimension. A 1D `torch.float32` Tensor with the same shape as `sum`.

## Examples

```python
import torch
import torch_npu

input = torch.randn(2, 3, 12, 12, dtype=torch.float32, device="npu")

sum_out, square_sum_out = torch_npu.batch_norm_reduce(input, 1e-5)

print("sum_out.shape:", sum_out.shape)
print("sum_out.dtype:", sum_out.dtype)
print("square_sum_out.shape:", square_sum_out.shape)
print("square_sum_out.dtype:", square_sum_out.dtype)
```
