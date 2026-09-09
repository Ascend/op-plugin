# (beta) torch_npu.npu_layer_norm_eval

> [!NOTICE]  
> This API is planned for deprecation. Use `torch.nn.functional.layer_norm` instead.

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 training products</term>          |    √     |
|<term>Atlas A2 training products</term>| √   |
|<term>Atlas training products</term>| √   |
|<term>Atlas inference products</term>| √   |

## Function

Computes the layer normalization result. The semantics are identical to those of `torch.nn.functional.layer_norm`, and the operation is accelerated on the NPU.

- Formula: Let $x$ be a slice vector consisting of the last several dimensions of `input`, with a total of $N$ elements. Let $\gamma_i$ and $\beta_i$ represent `weight` and `bias`, respectively; $E[x]$ be the mean of $x$; $Var[x]$ be the population variance of $x$ (with $N$ as the denominator); and $\varepsilon$ represent `eps`. Then
    $$
    \begin{cases}
    \hat{x}_i = \dfrac{x_i - E[x]}{\sqrt{Var[x] + \varepsilon}} \\[6pt]
    y_i = \gamma_i \cdot \hat{x}_i + \beta_i
    \end{cases}
    $$
    where
    $$
    E[x] = \frac{1}{N}\sum_{i=1}^{N} x_i, \qquad Var[x] = \frac{1}{N}\sum_{i=1}^{N}\left(x_i - E[x]\right)^2
    $$
    When the length of `normalized_shape` is $r$, $N = \prod_{j=1}^{r}\text{normalized\_shape}[j]$. Each slice corresponds to a fixed set of indices along the first several dimensions of `input`, and the slices are computed independently.

## Prototype

```python
torch_npu.npu_layer_norm_eval(input, normalized_shape, weight=None, bias=None, eps=1e-05) -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor. The data layout can be ND. Non-contiguous tensors are supported.
- **`normalized_shape`** (`List[int]`): Required. Shape of the dimensions to be normalized. Each dimension must match the corresponding dimension among the last several dimensions of `input`.
- **`weight`** (`Tensor`): Optional. Scaling parameter ($\gamma_i$) with a shape that must match `normalized_shape`. The default value is `None`.
- **`bias`** (`Tensor`): Optional. Bias parameter ($\beta_i$) with a shape that must match `normalized_shape`. The default value is `None`.
- **`eps`** (`float`): Optional. Value of $\varepsilon$ added to the denominator to ensure numerical stability. The default value is `1e-5`.

## Return Value

`Tensor`

Layer normalization result with the same shape and data type as `input`. The data layout can be ND. Non-contiguous tensors are supported.

## Constraints

- This API supports both inference and training.
- When `weight` and `bias` are not specified, their default values are `None`. During computation, `weight` and `bias` are treated as tensors of ones and zeros, respectively, with the same shape as `normalized_shape`.

## Example

```python
import torch
import torch_npu

input = torch.tensor(
    [[0.1863, 0.3755, 0.1115, 0.7308],
     [0.6004, 0.6832, 0.8951, 0.2087],
     [0.8548, 0.0176, 0.8498, 0.3703],
     [0.5609, 0.0114, 0.5021, 0.1242],
     [0.3966, 0.3022, 0.2323, 0.3914],
     [0.1554, 0.0149, 0.1718, 0.4972]],
    dtype=torch.float32,
).npu()
normalized_shape = input.size()[1:]
weight = torch.ones(normalized_shape, dtype=input.dtype, device=input.device)
bias = torch.zeros(normalized_shape, dtype=input.dtype, device=input.device)
output = torch_npu.npu_layer_norm_eval(input, normalized_shape, weight, bias, 1e-5)
print(output)
# Expected output of the preceding code sample:
# tensor([[-0.6879,  0.1022, -1.0002,  1.5859],
#         [ 0.0143,  0.3474,  1.1999, -1.5616],
#         [ 0.9422, -1.4361,  0.9280, -0.4341],
#         [ 1.1061, -1.2204,  0.8571, -0.7428],
#         [ 0.9685, -0.4173, -1.4434,  0.8922],
#         [-0.3078, -1.1025, -0.2151,  1.6255]], device='npu:0')
```
