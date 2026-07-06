# (beta) torch_npu.npu_rms_norm

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training products/Atlas A3 inference products</term>    |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>  |     √    |
|  <term>Atlas training products</term>  |     √    |

## Function

- Description: The RMSNorm operator is a normalization operation commonly used in foundation models. Compared with the LayerNorm operator, it removes the mean subtraction step.
- Formulas:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} g_i
  $$

  $$
  \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+eps}
  $$

## Prototype

```python
torch_npu.npu_rms_norm(self, gamma, epsilon=1e-06) -> (Tensor, Tensor) 
```

## Parameters

- **`self`** (`Tensor`): Required. Input for normalization computation, $x$ in the formulas. The shape must have 2 to 8 dimensions. The data layout can be ND. Non-contiguous tensors and empty tensors are supported.
- **`gamma`** (`Tensor`): Required. Scaling factor (weight) for normalization computation, $g$ in the computation formulas. The shape must have 2 to 8 dimensions. The data layout can be ND. The shape must satisfy `gamma_shape = self_shape[n:]`, where `n < self_shape.dims()`, which is typically the last dimension of `self`. Non-contiguous tensors and empty tensors are supported.
- **`epsilon`** (`double`): Optional. Used to prevent division-by-zero errors, $eps$ in the formula. The data type is `double`. The default value is `1e-6`.

## Return Values

- **`RmsNorm(x)`** (`Tensor`): Final output after normalization, $RmsNorm(x)$ in the formula. The data type and shape must be identical to those of the input `self`. Non-contiguous tensors and empty tensors are supported.
- **`rstd`** (`Tensor`): Reciprocal of the normalized standard deviation, the intermediate result of `rms_norm` used for backward computation, reciprocal of $Rms(x)$ in the formula. The data type is `float32`. The shape must be identical to the leading dimensions of the input parameter `self`. The leading dimensions refer to the dimensions of $x$ minus the dimensions of `gamma`, representing the dimensions that do not require normalization. Non-contiguous tensors and empty tensors are supported.

## Constraints

- Atlas inference products: The tail axis length of the `self` and `gamma` inputs must be greater than or equal to 32 bytes.
- The supported data types and their mappings for each product are as follows:
  - Atlas A3 training products/Atlas A3 inference products and Atlas A2 training products/Atlas A2 inference products:

    | self| gamma|
    | -------- | -------- |
    | `float16` | `float32` |
    | `bfloat16` | `float32` |
    | `float16` | `float16` |
    | `bfloat16` | `bfloat16` |
    | `float32` | `float32`  |

  - Atlas inference products and Atlas training products:

    | self| gamma|
    | -------- | -------- |
    | `float16` | `float16` |
    | `float32` | `float32` |

## Example

```python
>>> import torch, torch_npu
>>> x = torch.randn(24, 1, 128).npu()
>>> w = torch.randn(128).npu()
>>> out1 = torch_npu.npu_rms_norm(x, w, epsilon=1e-5)[0]
>>> print(out1)
tensor([[[-0.1875,  0.2383,  0.2334,  ...,  0.8555, -0.0908, -0.3574]],
        [[ 0.0747,  0.4668,  0.1074,  ...,  1.7500,  0.1953, -0.1992]],
        [[-0.0571, -0.4883,  0.5273,  ..., -2.1250, -0.0312,  2.3281]],
        ...,
        [[ 0.0503,  1.9453,  2.6094,  ..., -0.1357,  0.0869, -2.8906]],
        [[ 0.0195,  0.6680, -0.9336,  ..., -0.6641, -0.1904,  0.4336]],
        [[ 0.0972, -1.2344, -1.0078,  ..., -0.5195,  0.3145, -3.7656]]],
       device='npu:0')
```
