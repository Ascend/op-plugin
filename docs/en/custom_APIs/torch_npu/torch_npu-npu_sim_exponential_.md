# torch_npu.npu_sim_exponential_

## Supported Products

| Product                                                        | Supported|
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 training products/Atlas A2 inference products</term>       |    √     |
|<term>Atlas A3 training products/Atlas A3 inference products</term>       |    √     |

## Function

- Description: Generates exponentially distributed random numbers based on the parameter `lambd` and performs in-place filling of the input tensor `input`.
- Formula:
    $$f(x) = -1/λ * ln(1-u), u ~ Uniform(0, 1]$$

## Prototype

```python
torch_npu.npu_sim_exponential_(input, lambd=1, *, generator=None) -> Tensor
```

## Parameters

**`input`** (`Tensor`): Required. Source data tensor, $f(x)$ in the formula. This parameter must be a continuous tensor. The data type can be `bfloat16`, `float16`, or `float32`. The data layout can be ND. The shape can have 0 to 8 dimensions.

**`lambd`** (`double`): Optional. Parameter of the exponential distribution, $λ$ in the formula. This parameter can be set to any positive real number. The default value is `1`.

**`generator`** (`Generator`): Optional. Used to generate seed and offset for the `aclnnSimThreadExponential` operator. The default value is `None`.

## Return Values

`Tensor`

The `input` tensor after in-place update, $f(x)$ in the formula.

## Example

```python
>>> import torch
>>> import torch_npu

>>> shape = [100, 400]
>>> gen = torch.Generator(device="npu")
>>> gen.manual_seed(0)
>>> input = torch.zeros(shape, dtype=torch.float32).npu()
>>> torch_npu.npu_sim_exponential_(input, lambd=1, generator=gen)

```
