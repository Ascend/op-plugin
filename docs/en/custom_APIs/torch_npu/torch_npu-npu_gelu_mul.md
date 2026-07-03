# torch_npu.npu_gelu_mul

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>           |    √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>              | √   |
|<term>Atlas inference products</term>                                      |    √     |
|<term>Atlas training products</term>                                      |    √     |

## Function

- Description: Performs fused computation of GELU and MUL. When the last axis of `input` is 32-byte aligned, this API performs fused computation of GELU and MUL to improve performance. When the last axis is not 32-byte aligned, operator concatenation is recommended. That is, it performs computation step by step using the following formulas.
- Formulas:

    Given `input` with the last dimension length of $2d$, where $d$ is a positive integer, the computation process of $\text{GELUMUL}$ is as follows:

    1. Split the input tensor.
    
       Split `input` along the last dimension into tensors $x_1$ and $x_2$ with identical shapes.
         $$x_1 = \text{input}[..., :d], \quad x_2 = \text{input}[..., d:]$$
                 
         $x_1$ and $x_2$ have the same shape as that of $\text{input}$ except for the last dimension, which has length $d$.

    2. Apply the GELU activation function:

       Apply the GELU activation function to $x_1$, where the mode is controlled by `approximate`: $x_1 = \text{GELU}(x_1)$.
        - When `approximate="tanh"` (approximation mode with computation high efficiency):
        $$\text{GELU}(x) = 0.5 \cdot x \cdot \left[ 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \cdot \left( x + 0.044715 \cdot x^3 \right) \right) \right]$$
        
        - When `approximate="none"` (high-precision mode):
        $$\text{GELU}(x) = 0.5 \cdot x \cdot \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$

    3. Element-wise product:

       Multiply the activated $x_1$ and $x_2$ element-wise to obtain the final output tensor.
        $$\text{out} = x_1 \cdot x_2$$
                 
        The shape of $\text{out}$ is identical to that of `input`.

## Prototype

```python
torch_npu.npu_gelu_mul(input, *, approximate="none") -> Tensor
```

## Parameters

- **`input`** (`Tensor`): Required. Input tensor. The data type can be `bfloat16`, `float16`, or `float`. Non-contiguous tensors are supported. The data layout can be ND. This parameter must be 2D to 8D. The last dimension value must be an even number less than or equal to `1024`. The product of the other dimensions must be less than or equal to `200000`.
- **`approximate`** (`str`): Optional. Specifies the computation mode of the GELU activation function. Default value: `"none"`. Valid values:
  - `"none"`: enables error function (`erf`) mode, providing high computation precision, which is applicable to scenarios with strict precision requirements.
  - `"tanh"`: enables the hyperbolic tangent (`tanh`) approximation mode, providing high computational efficiency, which is applicable to large-scale training or inference acceleration.

## Return Values

`Tensor`

Output tensor, $\text{out}$ in the formulas. The data type can be `bfloat16`, `float16`, or `float`. This parameter must be 2D to 8D. Non-contiguous tensors are supported. The data layout can be ND. The output data type is identical to that of `input`. The last dimension length is half that of the input, and other dimensions remain unchanged. For example, when `input.shape` is `(100, 400)`, `output.shape` is `(100, 200)`.

## Examples

```python
>>> import torch, torch_npu
>>> shape = [100, 400]
>>> input = torch.rand(shape, dtype=torch.float16).npu()

# Configure high-precision mode with approximate="none": Use error function (erf) mode.
# This mode is applicable to scenarios with strict precision requirements.
>>> output_high_precision = torch_npu.npu_gelu_mul(input, approximate="none")

# Configure high-efficiency mode with approximate="tanh": Use hyperbolic tangent (tanh) approximation mode.
# It is applicable to large-scale training or inference acceleration.
>>> output_high_efficiency = torch_npu.npu_gelu_mul(input, approximate="tanh")

```
